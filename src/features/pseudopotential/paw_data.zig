const std = @import("std");

/// A single radial partial wave function: r*phi(r) or r*tphi(r).
pub const PawPartialWave = struct {
    l: i32,
    values: []f64,

    pub fn deinit(self: *PawPartialWave, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
    }
};

/// Augmentation charge Q_ij^L(r) entry for a specific (i,j,L) triplet.
pub const QijlEntry = struct {
    first_index: usize,
    second_index: usize,
    angular_momentum: usize,
    values: []f64,

    pub fn deinit(self: *QijlEntry, alloc: std.mem.Allocator) void {
        if (self.values.len > 0) alloc.free(self.values);
    }
};

/// PAW-specific data from a UPF file.
pub const PawData = struct {
    /// PP_FULL_WFC > PP_AEWFC: all-electron partial waves phi_i(r)
    ae_wfc: []PawPartialWave,
    /// PP_FULL_WFC > PP_PSWFC: pseudo partial waves tphi_i(r)
    ps_wfc: []PawPartialWave,
    /// PP_AUGMENTATION > PP_QIJL: augmentation charge Q_ij^L(r)
    qijl: []QijlEntry,
    /// Maximum L for augmentation charges
    lmax_aug: usize,
    /// Augmentation sphere cutoff radius (from PP_AUGMENTATION cutoff_r)
    cutoff_r: f64,
    /// Augmentation sphere cutoff radius index
    cutoff_r_index: usize,
    /// q_with_l flag from PP_AUGMENTATION
    q_with_l: bool,
    /// D^0_ij from PP_PAW (atomic reference, Ry units)
    dij0: []f64,
    /// PP_AE_NLCC: all-electron core charge density ρ_core^AE(r)
    ae_core_density: []f64,
    /// PP_AE_VLOC: all-electron local potential
    ae_local_potential: []f64,
    /// PP_OCCUPATIONS: reference occupations for partial waves
    occupations: []f64,
    /// core_energy attribute from PP_PAW (Ry)
    core_energy: f64,
    /// Number of projectors (same as beta count)
    number_of_proj: usize,

    pub fn deinit(self: *PawData, alloc: std.mem.Allocator) void {
        for (self.ae_wfc) |*w| w.deinit(alloc);
        if (self.ae_wfc.len > 0) alloc.free(self.ae_wfc);

        for (self.ps_wfc) |*w| w.deinit(alloc);
        if (self.ps_wfc.len > 0) alloc.free(self.ps_wfc);

        for (self.qijl) |*q| q.deinit(alloc);
        if (self.qijl.len > 0) alloc.free(self.qijl);

        if (self.dij0.len > 0) alloc.free(self.dij0);
        if (self.ae_core_density.len > 0) alloc.free(self.ae_core_density);
        if (self.ae_local_potential.len > 0) alloc.free(self.ae_local_potential);
        if (self.occupations.len > 0) alloc.free(self.occupations);
    }
};
