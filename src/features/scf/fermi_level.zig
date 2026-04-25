const std = @import("std");
const config = @import("../config/config.zig");
const kpoint_data = @import("kpoint/data.zig");

const KpointEigenData = kpoint_data.KpointEigenData;

pub fn smearing_occ(method: config.SmearingMethod, energy: f64, mu: f64, sigma: f64) f64 {
    if (sigma <= 0.0) return if (energy <= mu) 1.0 else 0.0;
    if (method == .none) return if (energy <= mu) 1.0 else 0.0;
    return 1.0 / (1.0 + std.math.exp((energy - mu) / sigma));
}

/// Compute entropy contribution for a given occupation number.
/// Returns -[f*ln(f) + (1-f)*ln(1-f)] which is always >= 0.
pub fn smearing_entropy(occ: f64) f64 {
    const eps = 1e-12;
    if (occ <= eps or occ >= 1.0 - eps) return 0.0;
    return -(occ * @log(occ) + (1.0 - occ) * @log(1.0 - occ));
}

pub fn electron_count_for_mu(
    mu: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data: []const KpointEigenData,
) f64 {
    return electron_count_for_mu_spin(mu, sigma, method, data, 2.0);
}

/// Electron count with configurable spin factor
/// (1.0 for spin-polarized per-channel, 2.0 for unpolarized).
pub fn electron_count_for_mu_spin(
    mu: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data: []const KpointEigenData,
    spin_factor: f64,
) f64 {
    var count: f64 = 0.0;
    for (data) |entry| {
        const weight = spin_factor * entry.kpoint.weight;
        for (entry.values[0..entry.nbands]) |energy| {
            count += weight * smearing_occ(method, energy, mu, sigma);
        }
    }
    return count;
}

pub fn find_fermi_level(
    nelec: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data: []const KpointEigenData,
) f64 {
    return find_fermi_level_spin(nelec, sigma, method, data, null, 2.0);
}

/// Find Fermi level for spin-polarized calculation.
/// Both spin channels' eigendata are passed; spin_factor should be 1.0.
/// If data_down is null, only data_up is used (equivalent to unpolarized with given spin_factor).
pub fn find_fermi_level_spin(
    nelec: f64,
    sigma: f64,
    method: config.SmearingMethod,
    data_up: []const KpointEigenData,
    data_down: ?[]const KpointEigenData,
    spin_factor: f64,
) f64 {
    var min_energy = std.math.inf(f64);
    var max_energy = -std.math.inf(f64);
    for (data_up) |entry| {
        for (entry.values[0..entry.nbands]) |energy| {
            min_energy = @min(min_energy, energy);
            max_energy = @max(max_energy, energy);
        }
    }
    if (data_down) |down| {
        for (down) |entry| {
            for (entry.values[0..entry.nbands]) |energy| {
                min_energy = @min(min_energy, energy);
                max_energy = @max(max_energy, energy);
            }
        }
    }
    var padding = @max(10.0 * sigma, 1e-3);
    var low = min_energy - padding;
    var high = max_energy + padding;
    var count_low = electron_count_for_mu_spin(low, sigma, method, data_up, spin_factor);
    var count_high = electron_count_for_mu_spin(high, sigma, method, data_up, spin_factor);
    if (data_down) |down| {
        count_low += electron_count_for_mu_spin(low, sigma, method, down, spin_factor);
        count_high += electron_count_for_mu_spin(high, sigma, method, down, spin_factor);
    }

    var expand: usize = 0;
    while ((count_low > nelec or count_high < nelec) and expand < 12) : (expand += 1) {
        padding *= 2.0;
        low = min_energy - padding;
        high = max_energy + padding;
        count_low = electron_count_for_mu_spin(low, sigma, method, data_up, spin_factor);
        count_high = electron_count_for_mu_spin(high, sigma, method, data_up, spin_factor);
        if (data_down) |down| {
            count_low += electron_count_for_mu_spin(low, sigma, method, down, spin_factor);
            count_high += electron_count_for_mu_spin(high, sigma, method, down, spin_factor);
        }
    }
    var left = low;
    var right = high;
    var iter: usize = 0;
    while (iter < 80) : (iter += 1) {
        const mid = 0.5 * (left + right);
        var count = electron_count_for_mu_spin(mid, sigma, method, data_up, spin_factor);
        if (data_down) |down| {
            count += electron_count_for_mu_spin(mid, sigma, method, down, spin_factor);
        }
        if (@abs(count - nelec) < 1e-8) return mid;
        if (count > nelec) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return 0.5 * (left + right);
}
