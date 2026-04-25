const std = @import("std");
const test_support = @import("../../test_support.zig");
pub const paw_data = @import("paw_data.zig");
pub const PawData = paw_data.PawData;

pub const Format = enum {
    upf,
    psp8,
    custom,
};

pub const Spec = struct {
    element: []const u8,
    path: []const u8,
    format: Format,
};

pub const Header = struct {
    element: ?[]u8,
    z_valence: ?f64,
    l_max: ?i32,
    mesh_size: ?usize,
    is_paw: bool = false,
    number_of_proj: ?usize = null,
    number_of_wfc: ?usize = null,

    /// Free owned header strings.
    pub fn deinit(self: *Header, alloc: std.mem.Allocator) void {
        if (self.element) |value| {
            alloc.free(value);
        }
    }
};

pub const Beta = struct {
    l: ?i32,
    values: []f64,

    /// Free owned beta values.
    pub fn deinit(self: *Beta, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) {
            alloc.free(self.values);
        }
    }
};

/// Atomic wavefunction from PP_PSWFC > PP_CHI section.
pub const AtomicWfc = struct {
    l: i32,
    label: ?[]u8,
    occupation: f64,
    values: []f64,

    pub fn deinit(self: *AtomicWfc, alloc: std.mem.Allocator) void {
        if (self.label) |lbl| alloc.free(lbl);
        if (self.values.len > 0) alloc.free(self.values);
    }
};

pub const UpfData = struct {
    r: []f64,
    rab: []f64,
    v_local: []f64,
    beta: []Beta,
    dij: []f64,
    qij: []f64,
    nlcc: []f64,
    rho_atom: []f64,
    paw: ?PawData = null,
    atomic_wfc: []AtomicWfc = &[_]AtomicWfc{},

    /// Free owned UPF arrays.
    pub fn deinit(self: *UpfData, alloc: std.mem.Allocator) void {
        if (self.r.len > 0) alloc.free(self.r);
        if (self.rab.len > 0) alloc.free(self.rab);
        if (self.v_local.len > 0) alloc.free(self.v_local);
        for (self.beta) |*b| {
            b.deinit(alloc);
        }
        if (self.beta.len > 0) alloc.free(self.beta);
        if (self.dij.len > 0) alloc.free(self.dij);
        if (self.qij.len > 0) alloc.free(self.qij);
        if (self.nlcc.len > 0) alloc.free(self.nlcc);
        if (self.rho_atom.len > 0) alloc.free(self.rho_atom);
        if (self.paw) |*p| p.deinit(alloc);
        for (self.atomic_wfc) |*w| {
            @constCast(w).deinit(alloc);
        }
        if (self.atomic_wfc.len > 0) alloc.free(self.atomic_wfc);
    }
};

pub const Parsed = struct {
    /// Spec data is borrowed from config and not owned.
    spec: Spec,
    header: Header,
    upf: ?UpfData,

    /// Free owned header and UPF data.
    pub fn deinit(self: *Parsed, alloc: std.mem.Allocator) void {
        self.header.deinit(alloc);
        if (self.upf) |*data| {
            data.deinit(alloc);
        }
    }
};

/// Parse pseudopotential format string.
pub fn parse_format(value: []const u8) !Format {
    if (std.mem.eql(u8, value, "upf")) return .upf;
    if (std.mem.eql(u8, value, "psp8")) return .psp8;
    if (std.mem.eql(u8, value, "custom")) return .custom;
    return error.InvalidPseudopotentialFormat;
}

/// Return format name.
pub fn format_name(format: Format) []const u8 {
    return switch (format) {
        .upf => "upf",
        .psp8 => "psp8",
        .custom => "custom",
    };
}

/// Load and parse a pseudopotential file.
pub fn load(alloc: std.mem.Allocator, io: std.Io, spec: Spec) !Parsed {
    const content = try std.Io.Dir.cwd().readFileAlloc(
        io,
        spec.path,
        alloc,
        .limited(8 * 1024 * 1024),
    );
    defer alloc.free(content);

    var header = Header{
        .element = null,
        .z_valence = null,
        .l_max = null,
        .mesh_size = null,
    };
    var upf_data: ?UpfData = null;

    switch (spec.format) {
        .upf => {
            header = try parse_upf_header(alloc, content);
            errdefer header.deinit(alloc);

            upf_data = try parse_upf_data(alloc, content);
            errdefer if (upf_data) |*data| data.deinit(alloc);

            if (header.mesh_size) |mesh| {
                const data = upf_data.?;
                if (data.r.len != mesh or data.rab.len != mesh or data.v_local.len != mesh) {
                    return error.InvalidUpf;
                }
                if (data.rho_atom.len > 0 and data.rho_atom.len != mesh) {
                    return error.InvalidUpf;
                }
                if (data.nlcc.len > 0 and data.nlcc.len != mesh) {
                    return error.InvalidUpf;
                }
            }
        },
        .psp8 => return error.UnsupportedPseudopotentialFormat,
        .custom => return error.UnsupportedPseudopotentialFormat,
    }

    return Parsed{ .spec = spec, .header = header, .upf = upf_data };
}

/// Parse UPF header attributes for minimal metadata.
fn parse_upf_header(alloc: std.mem.Allocator, content: []const u8) !Header {
    const header_start = std.mem.indexOf(u8, content, "<PP_HEADER") orelse return error.InvalidUpf;
    const tag_end = std.mem.indexOfPos(u8, content, header_start, ">") orelse
        return error.InvalidUpf;
    const tag = content[header_start .. tag_end + 1];

    var header = Header{
        .element = null,
        .z_valence = null,
        .l_max = null,
        .mesh_size = null,
    };

    if (find_attribute_value(tag, "element")) |value| {
        header.element = try alloc.dupe(u8, value);
    } else if (find_attribute_value(tag, "atomic_symbol")) |value| {
        header.element = try alloc.dupe(u8, value);
    }

    if (find_attribute_value(tag, "z_valence")) |value| {
        header.z_valence = try parse_float_token(alloc, value);
    }

    if (find_attribute_value(tag, "l_max")) |value| {
        header.l_max = try parse_i32(value);
    } else if (find_attribute_value(tag, "lmax")) |value| {
        header.l_max = try parse_i32(value);
    }

    if (find_attribute_value(tag, "mesh_size")) |value| {
        header.mesh_size = try parse_usize(value);
    } else if (find_attribute_value(tag, "mesh")) |value| {
        header.mesh_size = try parse_usize(value);
    }

    if (find_attribute_value(tag, "is_paw")) |value| {
        header.is_paw = std.mem.eql(u8, std.mem.trim(u8, value, " \t\r\n"), "true") or
            std.mem.eql(u8, std.mem.trim(u8, value, " \t\r\n"), "T");
    }

    if (find_attribute_value(tag, "number_of_proj")) |value| {
        header.number_of_proj = try parse_usize(value);
    }

    if (find_attribute_value(tag, "number_of_wfc")) |value| {
        header.number_of_wfc = try parse_usize(value);
    }

    return header;
}

/// Collect all PP_BETA entries from the UPF content into an allocated slice.
fn collect_beta_list(alloc: std.mem.Allocator, content: []const u8) ![]Beta {
    var beta_list: std.ArrayList(Beta) = .empty;
    errdefer {
        for (beta_list.items) |*b| b.deinit(alloc);
        beta_list.deinit(alloc);
    }
    var pos: usize = 0;
    while (find_next_tag(content, pos, "PP_BETA")) |tag| {
        const l = parse_beta_l(tag.start_tag);
        const values = try parse_float_list(alloc, tag.body);
        try beta_list.append(alloc, .{ .l = l, .values = values });
        pos = tag.end_pos;
    }
    return beta_list.toOwnedSlice(alloc);
}

/// Parse the optional float list under the tag with `name`, returning an
/// empty slice if the tag is absent.
fn parse_optional_float_tag(
    alloc: std.mem.Allocator,
    content: []const u8,
    name: []const u8,
) ![]f64 {
    if (find_tag_by_name(content, name)) |tag| {
        return parse_float_list(alloc, tag.body);
    }
    return empty_f64_slice();
}

/// Collect PP_CHI entries from the optional PP_PSWFC section.
fn collect_atomic_wfc(alloc: std.mem.Allocator, content: []const u8) ![]AtomicWfc {
    var chi_list: std.ArrayList(AtomicWfc) = .empty;
    errdefer {
        for (chi_list.items) |*w| w.deinit(alloc);
        chi_list.deinit(alloc);
    }
    if (find_tag_by_name(content, "PP_PSWFC")) |pswfc_tag| {
        var chi_pos: usize = 0;
        while (find_next_tag(pswfc_tag.body, chi_pos, "PP_CHI")) |tag| {
            const chi_l = if (find_attribute_value(tag.start_tag, "l")) |v|
                try parse_i32(v)
            else
                0;
            const chi_occ = if (find_attribute_value(tag.start_tag, "occupation")) |v|
                try parse_float_token(alloc, v)
            else
                0.0;
            const chi_label: ?[]u8 = if (find_attribute_value(tag.start_tag, "label")) |v|
                try alloc.dupe(u8, std.mem.trim(u8, v, " \t\r\n"))
            else
                null;
            errdefer if (chi_label) |lbl| alloc.free(lbl);

            const chi_values = try parse_float_list(alloc, tag.body);
            try chi_list.append(alloc, .{
                .l = chi_l,
                .label = chi_label,
                .occupation = chi_occ,
                .values = chi_values,
            });
            chi_pos = tag.end_pos;
        }
    }
    return chi_list.toOwnedSlice(alloc);
}

/// Parse UPF body sections needed for DFT.
fn parse_upf_data(alloc: std.mem.Allocator, content: []const u8) !UpfData {
    const r_tag = find_tag_by_name(content, "PP_R") orelse return error.InvalidUpf;
    const rab_tag = find_tag_by_name(content, "PP_RAB") orelse return error.InvalidUpf;
    const local_tag = find_tag_by_name(content, "PP_LOCAL") orelse return error.InvalidUpf;

    const r = try parse_float_list(alloc, r_tag.body);
    errdefer alloc.free(r);

    const rab = try parse_float_list(alloc, rab_tag.body);
    errdefer alloc.free(rab);

    const v_local = try parse_float_list(alloc, local_tag.body);
    errdefer alloc.free(v_local);

    const beta = try collect_beta_list(alloc, content);
    errdefer {
        for (beta) |*b| b.deinit(alloc);
        alloc.free(beta);
    }

    const dij = try parse_optional_float_tag(alloc, content, "PP_DIJ");
    errdefer if (dij.len > 0) alloc.free(dij);

    const qij = try parse_optional_float_tag(alloc, content, "PP_QIJ");
    errdefer if (qij.len > 0) alloc.free(qij);

    const nlcc = try parse_optional_float_tag(alloc, content, "PP_NLCC");
    errdefer if (nlcc.len > 0) alloc.free(nlcc);

    const rho_atom = try parse_optional_float_tag(alloc, content, "PP_RHOATOM");
    errdefer if (rho_atom.len > 0) alloc.free(rho_atom);

    const atomic_wfc = try collect_atomic_wfc(alloc, content);

    // Parse PAW data if present
    const paw_result = try parse_paw_data(alloc, content);

    return UpfData{
        .r = r,
        .rab = rab,
        .v_local = v_local,
        .beta = beta,
        .dij = dij,
        .qij = qij,
        .nlcc = nlcc,
        .rho_atom = rho_atom,
        .paw = paw_result,
        .atomic_wfc = atomic_wfc,
    };
}

/// Parsed PP_AUGMENTATION attribute block.
const AugmentationAttrs = struct {
    lmax_aug: usize,
    cutoff_r: f64,
    cutoff_r_index: usize,
    q_with_l: bool,
};

fn parse_augmentation_attrs(alloc: std.mem.Allocator, content: []const u8) !AugmentationAttrs {
    var out = AugmentationAttrs{
        .lmax_aug = 0,
        .cutoff_r = -1.0,
        .cutoff_r_index = 0,
        .q_with_l = false,
    };
    if (find_tag_by_name(content, "PP_AUGMENTATION")) |aug_tag| {
        if (find_attribute_value(aug_tag.start_tag, "l_max_aug")) |value| {
            out.lmax_aug = try parse_usize(value);
        }
        if (find_attribute_value(aug_tag.start_tag, "cutoff_r")) |value| {
            out.cutoff_r = try parse_float_token(alloc, value);
        }
        if (find_attribute_value(aug_tag.start_tag, "cutoff_r_index")) |value| {
            out.cutoff_r_index = try parse_usize(value);
        }
        if (find_attribute_value(aug_tag.start_tag, "q_with_l")) |value| {
            const trimmed = std.mem.trim(u8, value, " \t\r\n");
            out.q_with_l = std.mem.eql(u8, trimmed, "true") or std.mem.eql(u8, trimmed, "T");
        }
    }
    return out;
}

/// Collect PP_QIJL entries into an owned slice.
fn collect_qijl_entries(alloc: std.mem.Allocator, content: []const u8) ![]paw_data.QijlEntry {
    var qijl_list: std.ArrayList(paw_data.QijlEntry) = .empty;
    errdefer {
        for (qijl_list.items) |*q| q.deinit(alloc);
        qijl_list.deinit(alloc);
    }
    var pos: usize = 0;
    while (find_next_tag(content, pos, "PP_QIJL")) |tag| {
        const first = if (find_attribute_value(tag.start_tag, "first_index")) |v|
            (try parse_usize(v)) - 1 // Convert 1-based to 0-based
        else
            0;
        const second = if (find_attribute_value(tag.start_tag, "second_index")) |v|
            (try parse_usize(v)) - 1
        else
            0;
        const ang_mom = if (find_attribute_value(tag.start_tag, "angular_momentum")) |v|
            try parse_usize(v)
        else
            0;
        const values = try parse_float_list(alloc, tag.body);
        try qijl_list.append(alloc, .{
            .first_index = first,
            .second_index = second,
            .angular_momentum = ang_mom,
            .values = values,
        });
        pos = tag.end_pos;
    }
    return qijl_list.toOwnedSlice(alloc);
}

/// Collect partial-wave entries for a given tag name (PP_AEWFC / PP_PSWFC) in
/// the provided content scope.
fn collect_partial_waves(
    alloc: std.mem.Allocator,
    content: []const u8,
    tag_name: []const u8,
) ![]paw_data.PawPartialWave {
    var list: std.ArrayList(paw_data.PawPartialWave) = .empty;
    errdefer {
        for (list.items) |*w| w.deinit(alloc);
        list.deinit(alloc);
    }
    var pos: usize = 0;
    while (find_next_tag(content, pos, tag_name)) |tag| {
        const l = if (find_attribute_value(tag.start_tag, "l")) |v|
            try parse_i32(v)
        else if (find_attribute_value(tag.start_tag, "angular_momentum")) |v|
            try parse_i32(v)
        else
            0;
        const values = try parse_float_list(alloc, tag.body);
        try list.append(alloc, .{ .l = l, .values = values });
        pos = tag.end_pos;
    }
    return list.toOwnedSlice(alloc);
}

/// Parse PAW-specific sections from UPF content.
/// Returns null if this is not a PAW pseudopotential.
fn parse_paw_data(alloc: std.mem.Allocator, content: []const u8) !?PawData {
    // Check if PP_PAW section exists
    const paw_tag = find_tag_by_name(content, "PP_PAW") orelse return null;

    const core_fields = try parse_paw_core_fields(alloc, paw_tag);
    errdefer core_fields.deinit(alloc);
    const aug_attrs = try parse_augmentation_attrs(alloc, content);
    const qijl = try collect_qijl_entries(alloc, content);
    errdefer free_qijl_entries(alloc, qijl);
    const partial_waves = try collect_paw_partial_waves(alloc, content);
    errdefer partial_waves.deinit(alloc);

    // D^0_ij: use PP_DIJ values as the initial reference (these are D^0 for PAW)
    // The actual D^0_ij from PP_PAW is stored in PP_DIJ for PAW UPF files
    const dij0 = try parse_optional_float_tag(alloc, content, "PP_DIJ");
    errdefer if (dij0.len > 0) alloc.free(dij0);

    const number_of_proj = partial_waves.ae_wfc.len;

    return PawData{
        .ae_wfc = partial_waves.ae_wfc,
        .ps_wfc = partial_waves.ps_wfc,
        .qijl = qijl,
        .lmax_aug = aug_attrs.lmax_aug,
        .cutoff_r = aug_attrs.cutoff_r,
        .cutoff_r_index = aug_attrs.cutoff_r_index,
        .q_with_l = aug_attrs.q_with_l,
        .dij0 = dij0,
        .ae_core_density = core_fields.ae_core_density,
        .ae_local_potential = core_fields.ae_local_potential,
        .occupations = core_fields.occupations,
        .core_energy = core_fields.core_energy,
        .number_of_proj = number_of_proj,
    };
}

const PawCoreFields = struct {
    occupations: []f64,
    ae_core_density: []f64,
    ae_local_potential: []f64,
    core_energy: f64,

    fn deinit(self: PawCoreFields, alloc: std.mem.Allocator) void {
        if (self.occupations.len > 0) alloc.free(self.occupations);
        if (self.ae_core_density.len > 0) alloc.free(self.ae_core_density);
        if (self.ae_local_potential.len > 0) alloc.free(self.ae_local_potential);
    }
};

fn parse_paw_core_fields(alloc: std.mem.Allocator, paw_tag: Tag) !PawCoreFields {
    var core_energy: f64 = 0.0;
    if (find_attribute_value(paw_tag.start_tag, "core_energy")) |value| {
        core_energy = try parse_float_token(alloc, value);
    }

    const occupations = try parse_optional_float_tag(alloc, paw_tag.body, "PP_OCCUPATIONS");
    errdefer if (occupations.len > 0) alloc.free(occupations);
    const ae_core_density = try parse_optional_float_tag(alloc, paw_tag.body, "PP_AE_NLCC");
    errdefer if (ae_core_density.len > 0) alloc.free(ae_core_density);
    const ae_local_potential = try parse_optional_float_tag(alloc, paw_tag.body, "PP_AE_VLOC");
    errdefer if (ae_local_potential.len > 0) alloc.free(ae_local_potential);
    return .{
        .occupations = occupations,
        .ae_core_density = ae_core_density,
        .ae_local_potential = ae_local_potential,
        .core_energy = core_energy,
    };
}

const PawPartialWaveSet = struct {
    ae_wfc: []paw_data.PawPartialWave,
    ps_wfc: []paw_data.PawPartialWave,

    fn deinit(self: PawPartialWaveSet, alloc: std.mem.Allocator) void {
        for (self.ae_wfc) |*w| w.deinit(alloc);
        alloc.free(self.ae_wfc);
        for (self.ps_wfc) |*w| w.deinit(alloc);
        if (self.ps_wfc.len > 0) alloc.free(self.ps_wfc);
    }
};

fn collect_paw_partial_waves(
    alloc: std.mem.Allocator,
    content: []const u8,
) !PawPartialWaveSet {
    const ae_wfc = try collect_partial_waves(alloc, content, "PP_AEWFC");
    errdefer {
        for (ae_wfc) |*w| w.deinit(alloc);
        alloc.free(ae_wfc);
    }

    const ps_wfc: []paw_data.PawPartialWave =
        if (find_tag_by_name(content, "PP_FULL_WFC")) |full_wfc_tag|
            try collect_partial_waves(alloc, full_wfc_tag.body, "PP_PSWFC")
        else
            &[_]paw_data.PawPartialWave{};
    errdefer {
        for (ps_wfc) |*w| w.deinit(alloc);
        if (ps_wfc.len > 0) alloc.free(ps_wfc);
    }
    return .{ .ae_wfc = ae_wfc, .ps_wfc = ps_wfc };
}

fn free_qijl_entries(alloc: std.mem.Allocator, qijl: []paw_data.QijlEntry) void {
    for (qijl) |*q| q.deinit(alloc);
    alloc.free(qijl);
}

const Tag = struct {
    name: []const u8,
    start_tag: []const u8,
    body: []const u8,
    end_tag: []const u8,
    end_pos: usize,
};

/// Find the first tag that matches an exact name.
fn find_tag_by_name(content: []const u8, name: []const u8) ?Tag {
    var pos: usize = 0;
    while (find_next_tag(content, pos, name)) |tag| {
        if (tag.name.len == name.len) return tag;
        pos = tag.end_pos;
    }
    return null;
}

/// Find the next tag whose name starts with prefix.
fn find_next_tag(content: []const u8, start: usize, prefix: []const u8) ?Tag {
    var pos = start;
    while (true) {
        const open = std.mem.indexOfPos(u8, content, pos, "<") orelse return null;
        const name_start = open + 1;
        if (name_start + prefix.len > content.len) return null;
        if (!std.mem.eql(u8, content[name_start .. name_start + prefix.len], prefix)) {
            pos = open + 1;
            continue;
        }
        var name_end = name_start + prefix.len;
        while (name_end < content.len) : (name_end += 1) {
            const c = content[name_end];
            if (c == ' ' or c == '\t' or c == '\r' or c == '\n' or c == '>') break;
        }
        const name = content[name_start..name_end];
        const tag_end = std.mem.indexOfPos(u8, content, name_end, ">") orelse return null;
        const start_tag = content[open .. tag_end + 1];
        const close = find_closing_tag(content, tag_end + 1, name) orelse return null;
        const body = content[tag_end + 1 .. close.start];
        const end_tag = content[close.start..close.end];
        return Tag{
            .name = name,
            .start_tag = start_tag,
            .body = body,
            .end_tag = end_tag,
            .end_pos = close.end,
        };
    }
}

const CloseTag = struct {
    start: usize,
    end: usize,
};

/// Find a closing tag that matches name.
fn find_closing_tag(content: []const u8, start: usize, name: []const u8) ?CloseTag {
    var pos = start;
    while (pos < content.len) {
        const close = std.mem.indexOfPos(u8, content, pos, "</") orelse return null;
        const name_start = close + 2;
        if (name_start + name.len <= content.len and
            std.mem.eql(u8, content[name_start .. name_start + name.len], name))
        {
            const end_pos = name_start + name.len;
            if (end_pos < content.len and content[end_pos] == '>') {
                return CloseTag{ .start = close, .end = end_pos + 1 };
            }
        }
        pos = close + 2;
    }
    return null;
}

/// Parse beta angular momentum from tag attributes.
fn parse_beta_l(tag: []const u8) ?i32 {
    if (find_attribute_value(tag, "l")) |value| {
        return parse_i32(value) catch null;
    }
    if (find_attribute_value(tag, "angular_momentum")) |value| {
        return parse_i32(value) catch null;
    }
    return null;
}

/// Parse a whitespace-separated list of floats.
fn parse_float_list(alloc: std.mem.Allocator, text: []const u8) ![]f64 {
    var list: std.ArrayList(f64) = .empty;
    errdefer list.deinit(alloc);

    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (it.next()) |token| {
        try list.append(alloc, try parse_float_token(alloc, token));
    }
    return try list.toOwnedSlice(alloc);
}

/// Parse float token allowing Fortran-style exponents.
fn parse_float_token(alloc: std.mem.Allocator, token: []const u8) !f64 {
    const trimmed = std.mem.trim(u8, token, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidFloat;
    if (std.mem.indexOfAny(u8, trimmed, "Dd") == null) {
        return std.fmt.parseFloat(f64, trimmed);
    }
    const buf = try alloc.alloc(u8, trimmed.len);
    defer alloc.free(buf);

    @memcpy(buf, trimmed);
    for (buf) |*c| {
        if (c.* == 'D' or c.* == 'd') c.* = 'E';
    }
    return std.fmt.parseFloat(f64, buf);
}

/// Return a zero-length mutable slice.
fn empty_f64_slice() []f64 {
    return @constCast(&[_]f64{});
}

/// Find attribute value inside a tag.
/// Checks word boundaries to avoid matching substrings of other attribute names.
fn find_attribute_value(tag: []const u8, key: []const u8) ?[]const u8 {
    var pos: usize = 0;
    while (pos < tag.len) {
        const found = std.mem.indexOfPos(u8, tag, pos, key) orelse return null;
        var i = found + key.len;
        if (i >= tag.len) return null;
        // Check that the character before key is a word boundary
        if (found > 0) {
            const before = tag[found - 1];
            if (before != ' ' and before != '\t' and before != '\r' and
                before != '\n' and before != '<')
            {
                pos = found + 1;
                continue;
            }
        }
        // Check that the character after key is '=' or whitespace (not part of a longer name)
        if (tag[i] != '=' and tag[i] != ' ' and tag[i] != '\t') {
            pos = i;
            continue;
        }
        while (i < tag.len and (tag[i] == ' ' or tag[i] == '\t')) : (i += 1) {}
        if (i >= tag.len or tag[i] != '=') {
            pos = i;
            continue;
        }
        i += 1;
        while (i < tag.len and (tag[i] == ' ' or tag[i] == '\t')) : (i += 1) {}
        if (i >= tag.len) return null;
        const quote = tag[i];
        if (quote != '"' and quote != '\'') {
            pos = i;
            continue;
        }
        i += 1;
        const start = i;
        while (i < tag.len and tag[i] != quote) : (i += 1) {}
        if (i >= tag.len) return null;
        return tag[start..i];
    }
    return null;
}

/// Parse signed integer value.
fn parse_i32(value: []const u8) !i32 {
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidInteger;
    return try std.fmt.parseInt(i32, trimmed, 10);
}

/// Parse unsigned integer value.
fn parse_usize(value: []const u8) !usize {
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidInteger;
    return try std.fmt.parseInt(usize, trimmed, 10);
}

test "parse NC UPF has no PAW data" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    try test_support.require_file(io, "pseudo/Si_ONCV_PBE-1.2.upf");
    var parsed = try load(alloc, io, .{
        .element = "Si",
        .path = "pseudo/Si_ONCV_PBE-1.2.upf",
        .format = .upf,
    });
    defer parsed.deinit(alloc);

    try std.testing.expect(!parsed.header.is_paw);
    try std.testing.expect(parsed.upf.?.paw == null);
}

test "parse PAW UPF" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    try test_support.require_file(io, "pseudo/Si.pbe-n-kjpaw_psl.1.0.0.UPF");
    var parsed = try load(alloc, io, .{
        .element = "Si",
        .path = "pseudo/Si.pbe-n-kjpaw_psl.1.0.0.UPF",
        .format = .upf,
    });
    defer parsed.deinit(alloc);

    // Header checks
    try std.testing.expect(parsed.header.is_paw);
    try std.testing.expectEqual(@as(?usize, 6), parsed.header.number_of_proj);
    try std.testing.expectEqual(@as(?usize, 2), parsed.header.number_of_wfc);
    try std.testing.expectEqual(@as(?i32, 2), parsed.header.l_max);

    // PAW data must be present
    const paw = parsed.upf.?.paw.?;

    // 6 projectors => 6 AE and 6 PS partial waves
    try std.testing.expectEqual(@as(usize, 6), paw.ae_wfc.len);
    try std.testing.expectEqual(@as(usize, 6), paw.ps_wfc.len);
    try std.testing.expectEqual(@as(usize, 6), paw.number_of_proj);

    // Check angular momenta: 0,0,1,1,2,2
    try std.testing.expectEqual(@as(i32, 0), paw.ae_wfc[0].l);
    try std.testing.expectEqual(@as(i32, 0), paw.ae_wfc[1].l);
    try std.testing.expectEqual(@as(i32, 1), paw.ae_wfc[2].l);
    try std.testing.expectEqual(@as(i32, 1), paw.ae_wfc[3].l);
    try std.testing.expectEqual(@as(i32, 2), paw.ae_wfc[4].l);
    try std.testing.expectEqual(@as(i32, 2), paw.ae_wfc[5].l);

    // PS partial waves should have same angular momenta
    try std.testing.expectEqual(@as(i32, 0), paw.ps_wfc[0].l);
    try std.testing.expectEqual(@as(i32, 1), paw.ps_wfc[2].l);
    try std.testing.expectEqual(@as(i32, 2), paw.ps_wfc[4].l);

    // Augmentation parameters
    try std.testing.expectEqual(@as(usize, 4), paw.lmax_aug);
    try std.testing.expect(paw.q_with_l);

    // QIJL entries: should have entries for all (i,j,L) combos
    try std.testing.expect(paw.qijl.len > 0);

    // Each QIJL entry should have mesh_size values (1141)
    try std.testing.expectEqual(@as(usize, 1141), paw.qijl[0].values.len);

    // Occupations: 6 projectors
    try std.testing.expectEqual(@as(usize, 6), paw.occupations.len);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), paw.occupations[0], 1e-10);

    // Core energy (Ry)
    try std.testing.expectApproxEqAbs(@as(f64, -5.333458649736e2), paw.core_energy, 1e-6);

    // AE core density and AE local potential should have mesh_size values
    try std.testing.expectEqual(@as(usize, 1141), paw.ae_core_density.len);
    try std.testing.expectEqual(@as(usize, 1141), paw.ae_local_potential.len);

    // D^0_ij: 6×6 = 36 values
    try std.testing.expectEqual(@as(usize, 36), paw.dij0.len);

    // Check first D^0 value
    try std.testing.expectApproxEqAbs(@as(f64, 5.856343899820748e-1), paw.dij0[0], 1e-10);

    // Partial wave arrays should have mesh_size values
    try std.testing.expectEqual(@as(usize, 1141), paw.ae_wfc[0].values.len);
    try std.testing.expectEqual(@as(usize, 1141), paw.ps_wfc[0].values.len);

    // Atomic wavefunctions (PP_CHI)
    const wfc = parsed.upf.?.atomic_wfc;
    try std.testing.expectEqual(@as(usize, 2), wfc.len);
    try std.testing.expectEqual(@as(i32, 0), wfc[0].l); // 3S
    try std.testing.expectEqual(@as(i32, 1), wfc[1].l); // 3P
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), wfc[0].occupation, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), wfc[1].occupation, 1e-10);
    try std.testing.expectEqual(@as(usize, 1141), wfc[0].values.len);
    try std.testing.expect(std.mem.eql(u8, wfc[0].label.?, "3S"));
    try std.testing.expect(std.mem.eql(u8, wfc[1].label.?, "3P"));
}

test "parse NC UPF with no atomic wavefunctions" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;
    try test_support.require_file(io, "pseudo/Si_ONCV_PBE-1.2.upf");
    var parsed = try load(alloc, io, .{
        .element = "Si",
        .path = "pseudo/Si_ONCV_PBE-1.2.upf",
        .format = .upf,
    });
    defer parsed.deinit(alloc);

    // ONCV 1.2 has number_of_wfc=0, empty PP_PSWFC
    try std.testing.expectEqual(@as(usize, 0), parsed.upf.?.atomic_wfc.len);
}
