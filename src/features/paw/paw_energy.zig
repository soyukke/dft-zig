const std = @import("std");
const paw_data = @import("../pseudopotential/paw_data.zig");
const paw_tab = @import("paw_tab.zig");
const paw_xc = @import("paw_xc.zig");
const gaunt_mod = @import("gaunt.zig");
const PawData = paw_data.PawData;
const PawTab = paw_tab.PawTab;
const GauntTable = gaunt_mod.GauntTable;
const xc = @import("../xc/xc.zig");

/// Compute total PAW on-site energy correction for one atom.
///
/// E_PAW = (E_H + E_xc)^AE - (E_H + E_xc)^PS + core_energy
///
/// The kinetic correction K_ij is NOT included here because it is already
/// contained in D^0 (PP_DIJ from UPF). In QE's atomic_paw.f90,
/// dion = kdiff + V_loc integrals, where kdiff = <φ^AE|T|φ^AE> - <φ̃^PS|T|φ̃^PS>.
/// Since D^0 enters the Hamiltonian, Σ_ij ρ_ij × K_ij is already in the band energy.
///
/// For pseudized augmentation charges (PSQ), E_H^AE - E_H^{PS+Q} ≠ 0.
/// Uses full multipole expansion (all L) with Gaunt coefficients.
pub fn compute_paw_onsite_energy(
    alloc: std.mem.Allocator,
    paw: PawData,
    tab: *const PawTab,
    rhoij_m: []const f64,
    m_total: usize,
    m_offsets: []const usize,
    r: []const f64,
    rab: []const f64,
    rho_core_ps: ?[]const f64,
    xc_func: xc.Functional,
    gaunt_table: *const GauntTable,
) !f64 {
    _ = tab; // kij no longer used here (already in D^0/PP_DIJ)

    // On-site XC energy (angular Lebedev quadrature with GGA gradients)
    const e_xc = try paw_xc.compute_paw_exc_onsite_angular(
        alloc,
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        r,
        rab,
        paw.ae_core_density,
        rho_core_ps,
        xc_func,
        gaunt_table,
    );

    // On-site Hartree energy (multi-L multipole expansion with Gaunt coefficients)
    const e_h = try paw_xc.compute_paw_eh_onsite_multi_l(
        alloc,
        paw,
        rhoij_m,
        m_total,
        m_offsets,
        r,
        rab,
        gaunt_table,
    );

    return e_xc + e_h + paw.core_energy;
}
