const std = @import("std");
const math = @import("../math/math.zig");

/// PAW occupation matrix ρ_ij for each atom (m-resolved).
///
/// Stores the full (β_i,m_i) × (β_j,m_j) occupation matrix:
///   ρ_{(i,m_i),(j,m_j)} = Σ_{n,k} f_{n,k} w_k × <p̃_{i,m_i}|ψ̃_{n,k}> × <ψ̃_{n,k}|p̃_{j,m_j}>
///
/// For Si with 6 radial projectors (l=0,0,1,1,2,2), m_total = 2×1+2×3+2×5 = 18.
/// The matrix is 18×18 instead of the old m-summed 6×6.
///
/// This enables multi-L on-site density, which is required for accurate PAW
/// Hartree/XC corrections and QE-level agreement.
pub const RhoIJ = struct {
    /// ρ_ij values for each atom. Layout: [natom][m_total*m_total]
    values: [][]f64,
    /// Number of atoms
    natom: usize,
    /// m_total per atom (Σ_β (2l_β+1) for all β of that species)
    m_total_per_atom: []usize,
    /// Number of radial projectors per atom
    nbeta_per_atom: []usize,
    /// l value for each radial projector, per atom species (shared within species)
    /// Layout: [natom] -> slice of l values (len = nbeta)
    l_per_beta: [][]const i32,
    /// m offset for each radial projector (cumulative sum of 2l+1)
    /// Layout: [natom] -> slice of offsets (len = nbeta)
    m_offsets: [][]const usize,

    pub fn init(
        alloc: std.mem.Allocator,
        natom: usize,
        nbeta_list: []const usize,
        l_lists: []const []const i32,
    ) !RhoIJ {
        const values = try alloc.alloc([]f64, natom);
        errdefer {
            for (values[0..natom]) |v| {
                if (v.len > 0) alloc.free(v);
            }
            alloc.free(values);
        }
        const m_total_per_atom = try alloc.alloc(usize, natom);
        errdefer alloc.free(m_total_per_atom);
        const nbeta_per_atom = try alloc.alloc(usize, natom);
        errdefer alloc.free(nbeta_per_atom);
        const l_per_beta = try alloc.alloc([]const i32, natom);
        errdefer alloc.free(l_per_beta);
        const m_offsets_arr = try alloc.alloc([]const usize, natom);
        errdefer {
            for (m_offsets_arr[0..natom]) |o| {
                if (o.len > 0) alloc.free(@constCast(o));
            }
            alloc.free(m_offsets_arr);
        }

        for (0..natom) |a| {
            const nb = nbeta_list[a];
            const l_list = l_lists[a];
            nbeta_per_atom[a] = nb;
            l_per_beta[a] = l_list;

            // Compute m_total and offsets
            var mt: usize = 0;
            const offsets = try alloc.alloc(usize, nb);
            for (0..nb) |b| {
                offsets[b] = mt;
                mt += @as(usize, @intCast(2 * l_list[b] + 1));
            }
            m_total_per_atom[a] = mt;
            m_offsets_arr[a] = offsets;

            values[a] = try alloc.alloc(f64, mt * mt);
            @memset(values[a], 0.0);
        }

        return .{
            .values = values,
            .natom = natom,
            .m_total_per_atom = m_total_per_atom,
            .nbeta_per_atom = nbeta_per_atom,
            .l_per_beta = l_per_beta,
            .m_offsets = m_offsets_arr,
        };
    }

    /// Reset all ρ_ij to zero (before accumulation over k-points).
    pub fn reset(self: *RhoIJ) void {
        for (0..self.natom) |a| {
            @memset(self.values[a], 0.0);
        }
    }

    /// Accumulate m-resolved ρ_ij from projector overlaps for one band at one k-point.
    /// proj_overlaps[a] has length m_total for atom a: <p̃_{β,m}|ψ̃_{n,k}>.
    /// weight = f_nk * w_k (occupation × k-weight).
    pub fn accumulate(
        self: *RhoIJ,
        proj_overlaps: []const []const math.Complex,
        weight: f64,
    ) void {
        for (0..self.natom) |a| {
            const mt = self.m_total_per_atom[a];
            const overlaps = proj_overlaps[a];
            for (0..mt) |i| {
                for (0..mt) |j| {
                    const ci = overlaps[i];
                    const cj = overlaps[j];
                    self.values[a][i * mt + j] += weight * (ci.re * cj.re + ci.im * cj.im);
                }
            }
        }
    }

    /// Contract m-resolved rhoij to radial rhoij for a given atom.
    /// Result: radial_rhoij[i*nbeta+j] = Σ_m ρ_{(i,m),(j,m)} for l_i == l_j, 0 otherwise.
    /// This is what the old m-summed code produced.
    pub fn contract_to_radial(
        self: *const RhoIJ,
        atom_idx: usize,
        radial_rhoij: []f64,
    ) void {
        const nb = self.nbeta_per_atom[atom_idx];
        const mt = self.m_total_per_atom[atom_idx];
        const rhoij_m = self.values[atom_idx];
        const l_list = self.l_per_beta[atom_idx];
        const offsets = self.m_offsets[atom_idx];

        @memset(radial_rhoij[0 .. nb * nb], 0.0);

        for (0..nb) |i| {
            for (0..nb) |j| {
                if (l_list[i] != l_list[j]) continue;
                const m_count = @as(usize, @intCast(2 * l_list[i] + 1));
                var sum: f64 = 0.0;
                for (0..m_count) |mi| {
                    const idx_i = offsets[i] + mi;
                    const idx_j = offsets[j] + mi;
                    sum += rhoij_m[idx_i * mt + idx_j];
                }
                radial_rhoij[i * nb + j] = sum;
            }
        }
    }

    /// Clone this RhoIJ (deep copy of all data).
    pub fn clone(self: *const RhoIJ, alloc: std.mem.Allocator) !RhoIJ {
        const values = try alloc.alloc([]f64, self.natom);
        errdefer {
            for (values) |v| {
                if (v.len > 0) alloc.free(v);
            }
            alloc.free(values);
        }
        for (0..self.natom) |a| {
            values[a] = try alloc.alloc(f64, self.values[a].len);
            @memcpy(values[a], self.values[a]);
        }
        const m_total_per_atom = try alloc.alloc(usize, self.natom);
        @memcpy(m_total_per_atom, self.m_total_per_atom);
        const nbeta_per_atom = try alloc.alloc(usize, self.natom);
        @memcpy(nbeta_per_atom, self.nbeta_per_atom);
        const l_per_beta = try alloc.alloc([]const i32, self.natom);
        @memcpy(l_per_beta, self.l_per_beta); // shared pointers, not owned
        const m_offsets = try alloc.alloc([]const usize, self.natom);
        errdefer alloc.free(m_offsets);
        for (0..self.natom) |a| {
            const src = self.m_offsets[a];
            const dst = try alloc.alloc(usize, src.len);
            @memcpy(dst, src);
            m_offsets[a] = dst;
        }
        return .{
            .values = values,
            .natom = self.natom,
            .m_total_per_atom = m_total_per_atom,
            .nbeta_per_atom = nbeta_per_atom,
            .l_per_beta = l_per_beta,
            .m_offsets = m_offsets,
        };
    }

    /// Add scaled values from another RhoIJ: self += scale * other.
    /// Both must have the same atom layout.
    pub fn add_scaled(self: *RhoIJ, other: *const RhoIJ, scale: f64) void {
        for (0..self.natom) |a| {
            for (0..self.values[a].len) |i| {
                self.values[a][i] += scale * other.values[a][i];
            }
        }
    }

    pub fn deinit(self: *RhoIJ, alloc: std.mem.Allocator) void {
        for (0..self.natom) |a| {
            if (self.values[a].len > 0) alloc.free(self.values[a]);
            if (self.m_offsets[a].len > 0) alloc.free(@constCast(self.m_offsets[a]));
        }
        alloc.free(self.values);
        alloc.free(self.m_total_per_atom);
        alloc.free(self.nbeta_per_atom);
        alloc.free(self.l_per_beta);
        alloc.free(self.m_offsets);
    }
};
