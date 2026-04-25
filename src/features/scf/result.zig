const std = @import("std");
const apply = @import("apply.zig");
const energy_mod = @import("energy.zig");
const final_wavefunction = @import("final_wavefunction.zig");
const grid_mod = @import("pw_grid.zig");
const hamiltonian = @import("../hamiltonian/hamiltonian.zig");
const kpoint = @import("kpoint.zig");
const math = @import("../math/math.zig");
const paw_mod = @import("../paw/paw.zig");

const Grid = grid_mod.Grid;
const KpointCache = kpoint.KpointCache;
const WavefunctionData = final_wavefunction.WavefunctionData;

pub const ScfResult = struct {
    potential: hamiltonian.PotentialGrid,
    density: []f64,
    iterations: usize,
    converged: bool,
    energy: energy_mod.EnergyTerms,
    fermi_level: f64,
    potential_residual: f64,
    wavefunctions: ?WavefunctionData,
    vresid: ?hamiltonian.PotentialGrid,
    grid: Grid,
    kpoint_cache: ?[]KpointCache = null,
    apply_caches: ?[]apply.KpointApplyCache = null,
    vxc_r: ?[]f64 = null,
    // Spin-polarized fields (nspin=2 only)
    density_up: ?[]f64 = null,
    density_down: ?[]f64 = null,
    potential_down: ?hamiltonian.PotentialGrid = null,
    magnetization: f64 = 0.0,
    wavefunctions_down: ?WavefunctionData = null,
    vxc_r_up: ?[]f64 = null,
    vxc_r_down: ?[]f64 = null,
    fermi_level_down: f64 = 0.0,
    // PAW fields for band calculation
    paw_tabs: ?[]paw_mod.PawTab = null, // Owned PAW tables (one per species)
    paw_dij: ?[][]f64 = null, // Per-atom converged D_ij (radial): [natom][nbeta*nbeta]
    paw_dij_m: ?[][]f64 = null, // Per-atom converged D_ij (m-resolved): [natom][mt*mt]
    paw_dxc: ?[][]f64 = null, // Per-atom D^xc_ij (m-resolved): [natom][mt*mt]
    paw_rhoij: ?[][]f64 = null, // Per-atom rhoij: [natom][nbeta*nbeta]
    ionic_g: ?[]math.Complex = null, // Ionic potential in G-space (for PAW D^hat force)
    rho_core: ?[]f64 = null, // NLCC core density in real space (for stress)

    /// Free allocated SCF results.
    pub fn deinit(self: *ScfResult, alloc: std.mem.Allocator) void {
        self.potential.deinit(alloc);
        if (self.density.len > 0) alloc.free(self.density);
        if (self.wavefunctions) |*wf| wf.deinit(alloc);
        if (self.vresid) |*vresid| vresid.deinit(alloc);
        if (self.kpoint_cache) |cache| {
            for (cache) |*c| c.deinit();
            alloc.free(cache);
        }
        if (self.apply_caches) |caches| {
            for (caches) |*ac| ac.deinit(alloc);
            alloc.free(caches);
        }
        if (self.vxc_r) |v| alloc.free(v);
        if (self.density_up) |d| alloc.free(d);
        if (self.density_down) |d| alloc.free(d);
        if (self.potential_down) |*p| p.deinit(alloc);
        if (self.wavefunctions_down) |*wf| wf.deinit(alloc);
        if (self.vxc_r_up) |v| alloc.free(v);
        if (self.vxc_r_down) |v| alloc.free(v);
        if (self.paw_dij) |dij| {
            for (dij) |d| alloc.free(d);
            alloc.free(dij);
        }
        if (self.paw_dij_m) |dij| {
            for (dij) |d| alloc.free(d);
            alloc.free(dij);
        }
        if (self.paw_dxc) |dxc| {
            for (dxc) |d| alloc.free(d);
            alloc.free(dxc);
        }
        if (self.paw_rhoij) |rij| {
            for (rij) |r| alloc.free(r);
            alloc.free(rij);
        }
        if (self.paw_tabs) |tabs| {
            for (@constCast(tabs)) |*t| t.deinit(alloc);
            alloc.free(tabs);
        }
        if (self.ionic_g) |ig| alloc.free(ig);
        if (self.rho_core) |rc| alloc.free(rc);
    }
};

/// Params bag for building the final ScfResult.
pub const ScfResultSetup = struct {
    potential: hamiltonian.PotentialGrid,
    density: []f64,
    iterations: usize,
    converged: bool,
    energy: energy_mod.EnergyTerms,
    fermi_level: f64,
    potential_residual: f64,
    wavefunctions: ?WavefunctionData,
    vresid: ?hamiltonian.PotentialGrid,
    grid: Grid,
    kpoint_cache: []KpointCache,
    apply_caches: []apply.KpointApplyCache,
    vxc_r: ?[]f64,
    paw_tabs: ?[]paw_mod.PawTab,
    paw_dij: ?[][]f64,
    paw_dij_m: ?[][]f64,
    paw_dxc: ?[][]f64,
    paw_rhoij: ?[][]f64,
    ionic_g: ?[]math.Complex,
    rho_core_copy: ?[]f64,
};

pub fn build_scf_result(s: ScfResultSetup) ScfResult {
    return ScfResult{
        .potential = s.potential,
        .density = s.density,
        .iterations = s.iterations,
        .converged = s.converged,
        .energy = s.energy,
        .fermi_level = s.fermi_level,
        .potential_residual = s.potential_residual,
        .wavefunctions = s.wavefunctions,
        .vresid = s.vresid,
        .grid = s.grid,
        .kpoint_cache = s.kpoint_cache,
        .apply_caches = s.apply_caches,
        .vxc_r = s.vxc_r,
        .paw_tabs = s.paw_tabs,
        .paw_dij = s.paw_dij,
        .paw_dij_m = s.paw_dij_m,
        .paw_dxc = s.paw_dxc,
        .paw_rhoij = s.paw_rhoij,
        .ionic_g = s.ionic_g,
        .rho_core = s.rho_core_copy,
    };
}

/// Copy NLCC core density to a newly-allocated buffer for result, or null.
pub fn copy_rho_core_for_result(
    alloc: std.mem.Allocator,
    rho_core: ?[]const f64,
) !?[]f64 {
    if (rho_core) |rc| {
        const copy = try alloc.alloc(f64, rc.len);
        @memcpy(copy, rc);
        return copy;
    }
    return null;
}
