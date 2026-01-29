//! Auxiliary basis sets for density fitting (RI-J/K).
//!
//! Provides:
//!   - Even-tempered auxiliary basis (for testing)
//!   - def2-universal-JKFIT (Weigend 2008) for H, C, N, O, F

const std = @import("std");
const math = @import("../math/math.zig");
const gaussian = @import("gaussian.zig");

const PrimitiveGaussian = gaussian.PrimitiveGaussian;
const ContractedShell = gaussian.ContractedShell;

pub const MAX_AUX_SHELLS: usize = 64;

pub const AuxBasisResult = struct {
    shells: [MAX_AUX_SHELLS]ContractedShell,
    count: usize,

    pub fn slice(self: *const AuxBasisResult) []const ContractedShell {
        return self.shells[0..self.count];
    }
};

/// Build an even-tempered auxiliary basis set for testing.
/// Generates n_per_l uncontracted shells for each l from 0 to l_max.
/// Exponents form a geometric series: alpha_min * ratio^i.
pub fn buildEvenTemperedAux(
    center: math.Vec3,
    l_max: u32,
    n_per_l: usize,
    alpha_min: f64,
    ratio: f64,
) AuxBasisResult {
    var result = AuxBasisResult{ .shells = undefined, .count = 0 };

    var l: u32 = 0;
    while (l <= l_max) : (l += 1) {
        for (0..n_per_l) |i| {
            if (result.count >= MAX_AUX_SHELLS) break;
            const alpha = alpha_min * std.math.pow(f64, ratio, @floatFromInt(i));
            const prim_storage = &even_tempered_prims[result.count];
            prim_storage.* = .{ .alpha = alpha, .coeff = 1.0 };
            result.shells[result.count] = .{
                .center = center,
                .l = l,
                .primitives = @as(*const [1]PrimitiveGaussian, prim_storage),
            };
            result.count += 1;
        }
    }

    return result;
}

// Static storage for even-tempered primitives (one per shell)
var even_tempered_prims: [MAX_AUX_SHELLS]PrimitiveGaussian = undefined;

// ============================================================================
// def2-universal-JKFIT data (Weigend 2008, from Basis Set Exchange)
// ============================================================================

/// Build def2-universal-JKFIT auxiliary basis for element z at given center.
/// Returns null if element is not supported.
/// Uses Cartesian Gaussians for all angular momenta.
pub fn buildDef2UniversalJkfit(z: u32, center: math.Vec3) ?AuxBasisResult {
    return switch (z) {
        1 => buildFromData(center, &h_data),
        6 => buildFromData(center, &c_data),
        7 => buildFromData(center, &n_data),
        8 => buildFromData(center, &o_data),
        9 => buildFromData(center, &f_data),
        else => null,
    };
}

const ShellData = struct {
    l: u32,
    primitives: []const PrimitiveGaussian,
};

fn buildFromData(center: math.Vec3, data: []const ShellData) AuxBasisResult {
    var result = AuxBasisResult{ .shells = undefined, .count = 0 };
    for (data) |sd| {
        if (result.count >= MAX_AUX_SHELLS) break;
        result.shells[result.count] = .{
            .center = center,
            .l = sd.l,
            .primitives = sd.primitives,
        };
        result.count += 1;
    }
    return result;
}

// ============================================================================
// H (Z=1): 6 shells
// ============================================================================

const h_s1 = [_]PrimitiveGaussian{
    .{ .alpha = 22.0683430000, .coeff = 0.0530339860 },
    .{ .alpha = 4.3905712000, .coeff = 0.3946522022 },
    .{ .alpha = 1.0540787000, .coeff = 0.9172987712 },
};
const h_s2 = [_]PrimitiveGaussian{.{ .alpha = 0.2717874000, .coeff = 1.0 }};
const h_p1 = [_]PrimitiveGaussian{.{ .alpha = 1.8529979000, .coeff = 1.0 }};
const h_p2 = [_]PrimitiveGaussian{.{ .alpha = 0.3881034000, .coeff = 1.0 }};
const h_d1 = [_]PrimitiveGaussian{.{ .alpha = 2.5579933000, .coeff = 1.0 }};
const h_d2 = [_]PrimitiveGaussian{.{ .alpha = 0.3292649000, .coeff = 1.0 }};

const h_data = [_]ShellData{
    .{ .l = 0, .primitives = &h_s1 },
    .{ .l = 0, .primitives = &h_s2 },
    .{ .l = 1, .primitives = &h_p1 },
    .{ .l = 1, .primitives = &h_p2 },
    .{ .l = 2, .primitives = &h_d1 },
    .{ .l = 2, .primitives = &h_d2 },
};

// ============================================================================
// C (Z=6): 25 shells
// ============================================================================

const c_s1 = [_]PrimitiveGaussian{
    .{ .alpha = 384.44382410, .coeff = 0.18257867580 },
    .{ .alpha = 167.56266310, .coeff = 0.11970433650 },
    .{ .alpha = 76.619620300, .coeff = 0.51617376790 },
    .{ .alpha = 36.679529300, .coeff = 0.53561565140 },
    .{ .alpha = 18.338091700, .coeff = 0.60189974570 },
};
const c_s2 = [_]PrimitiveGaussian{.{ .alpha = 9.5470634000, .coeff = 0.19165883840 }};
const c_s3 = [_]PrimitiveGaussian{.{ .alpha = 5.1584143000, .coeff = 1.0 }};
const c_s4 = [_]PrimitiveGaussian{.{ .alpha = 2.8816701000, .coeff = 1.0 }};
const c_s5 = [_]PrimitiveGaussian{.{ .alpha = 1.6573522000, .coeff = 1.0 }};
const c_s6 = [_]PrimitiveGaussian{.{ .alpha = 0.97681020000, .coeff = 1.0 }};
const c_s7 = [_]PrimitiveGaussian{.{ .alpha = 0.58702000000, .coeff = 1.0 }};
const c_s8 = [_]PrimitiveGaussian{.{ .alpha = 0.35779270000, .coeff = 1.0 }};
const c_s9 = [_]PrimitiveGaussian{.{ .alpha = 0.21995500000, .coeff = 1.0 }};
const c_s10 = [_]PrimitiveGaussian{.{ .alpha = 0.13560770000, .coeff = 1.0 }};

const c_p1 = [_]PrimitiveGaussian{
    .{ .alpha = 62.067956100, .coeff = 0.25747330090 },
    .{ .alpha = 20.012750700, .coeff = 0.58566021840 },
    .{ .alpha = 10.205359400, .coeff = 0.51756621670 },
};
const c_p2 = [_]PrimitiveGaussian{.{ .alpha = 5.6367045000, .coeff = 0.56818555000 }};
const c_p3 = [_]PrimitiveGaussian{.{ .alpha = 2.8744918000, .coeff = 1.0 }};
const c_p4 = [_]PrimitiveGaussian{.{ .alpha = 1.4513791000, .coeff = 1.0 }};
const c_p5 = [_]PrimitiveGaussian{.{ .alpha = 0.73283270000, .coeff = 1.0 }};
const c_p6 = [_]PrimitiveGaussian{.{ .alpha = 0.37002560000, .coeff = 1.0 }};
const c_p7 = [_]PrimitiveGaussian{.{ .alpha = 0.18683600000, .coeff = 1.0 }};
const c_p8 = [_]PrimitiveGaussian{.{ .alpha = 0.095282100000, .coeff = 1.0 }};

const c_d1 = [_]PrimitiveGaussian{
    .{ .alpha = 11.282005700, .coeff = 0.13434185260 },
    .{ .alpha = 4.3789486000, .coeff = 0.41999086490 },
};
const c_d2 = [_]PrimitiveGaussian{.{ .alpha = 1.7517699000, .coeff = 0.89752991040 }};
const c_d3 = [_]PrimitiveGaussian{.{ .alpha = 0.71648520000, .coeff = 1.0 }};
const c_d4 = [_]PrimitiveGaussian{.{ .alpha = 0.29704020000, .coeff = 1.0 }};
const c_d5 = [_]PrimitiveGaussian{.{ .alpha = 0.12370870000, .coeff = 1.0 }};

const c_f1 = [_]PrimitiveGaussian{
    .{ .alpha = 1.8786227000, .coeff = 0.62371094050 },
    .{ .alpha = 0.68603580000, .coeff = 0.78165507910 },
};

const c_g1 = [_]PrimitiveGaussian{.{ .alpha = 1.1430602000, .coeff = 1.0 }};

const c_data = [_]ShellData{
    .{ .l = 0, .primitives = &c_s1 },
    .{ .l = 0, .primitives = &c_s2 },
    .{ .l = 0, .primitives = &c_s3 },
    .{ .l = 0, .primitives = &c_s4 },
    .{ .l = 0, .primitives = &c_s5 },
    .{ .l = 0, .primitives = &c_s6 },
    .{ .l = 0, .primitives = &c_s7 },
    .{ .l = 0, .primitives = &c_s8 },
    .{ .l = 0, .primitives = &c_s9 },
    .{ .l = 0, .primitives = &c_s10 },
    .{ .l = 1, .primitives = &c_p1 },
    .{ .l = 1, .primitives = &c_p2 },
    .{ .l = 1, .primitives = &c_p3 },
    .{ .l = 1, .primitives = &c_p4 },
    .{ .l = 1, .primitives = &c_p5 },
    .{ .l = 1, .primitives = &c_p6 },
    .{ .l = 1, .primitives = &c_p7 },
    .{ .l = 1, .primitives = &c_p8 },
    .{ .l = 2, .primitives = &c_d1 },
    .{ .l = 2, .primitives = &c_d2 },
    .{ .l = 2, .primitives = &c_d3 },
    .{ .l = 2, .primitives = &c_d4 },
    .{ .l = 2, .primitives = &c_d5 },
    .{ .l = 3, .primitives = &c_f1 },
    .{ .l = 4, .primitives = &c_g1 },
};

// ============================================================================
// N (Z=7): 25 shells
// ============================================================================

const n_s1 = [_]PrimitiveGaussian{
    .{ .alpha = 502.86084920, .coeff = 0.17948662110 },
    .{ .alpha = 209.96500460, .coeff = 0.16394914980 },
    .{ .alpha = 92.434008900, .coeff = 0.55297971800 },
    .{ .alpha = 42.817583500, .coeff = 0.58582103980 },
    .{ .alpha = 20.817615400, .coeff = 0.53192624460 },
};
const n_s2 = [_]PrimitiveGaussian{.{ .alpha = 10.591289400, .coeff = 0.094798946400 }};
const n_s3 = [_]PrimitiveGaussian{.{ .alpha = 5.6186543000, .coeff = 1.0 }};
const n_s4 = [_]PrimitiveGaussian{.{ .alpha = 3.0952689000, .coeff = 1.0 }};
const n_s5 = [_]PrimitiveGaussian{.{ .alpha = 1.7624885000, .coeff = 1.0 }};
const n_s6 = [_]PrimitiveGaussian{.{ .alpha = 1.0319737000, .coeff = 1.0 }};
const n_s7 = [_]PrimitiveGaussian{.{ .alpha = 0.61783520000, .coeff = 1.0 }};
const n_s8 = [_]PrimitiveGaussian{.{ .alpha = 0.37593430000, .coeff = 1.0 }};
const n_s9 = [_]PrimitiveGaussian{.{ .alpha = 0.23100830000, .coeff = 1.0 }};
const n_s10 = [_]PrimitiveGaussian{.{ .alpha = 0.14242240000, .coeff = 1.0 }};

const n_p1 = [_]PrimitiveGaussian{
    .{ .alpha = 44.607497200, .coeff = 0.51530130010 },
    .{ .alpha = 22.523067100, .coeff = 0.19383347500 },
    .{ .alpha = 11.372330300, .coeff = 0.81042183410 },
};
const n_p2 = [_]PrimitiveGaussian{.{ .alpha = 5.7421344000, .coeff = 0.20027382490 }};
const n_p3 = [_]PrimitiveGaussian{.{ .alpha = 2.8993384000, .coeff = 1.0 }};
const n_p4 = [_]PrimitiveGaussian{.{ .alpha = 1.4639486000, .coeff = 1.0 }};
const n_p5 = [_]PrimitiveGaussian{.{ .alpha = 0.73918610000, .coeff = 1.0 }};
const n_p6 = [_]PrimitiveGaussian{.{ .alpha = 0.37323510000, .coeff = 1.0 }};
const n_p7 = [_]PrimitiveGaussian{.{ .alpha = 0.18845670000, .coeff = 1.0 }};
const n_p8 = [_]PrimitiveGaussian{.{ .alpha = 0.095157100000, .coeff = 1.0 }};

const n_d1 = [_]PrimitiveGaussian{
    .{ .alpha = 46.844040600, .coeff = 0.039944270800 },
    .{ .alpha = 14.882456100, .coeff = 0.27216475910 },
    .{ .alpha = 5.3437066000, .coeff = 0.96142123920 },
};
const n_d2 = [_]PrimitiveGaussian{.{ .alpha = 2.1154199000, .coeff = 1.0 }};
const n_d3 = [_]PrimitiveGaussian{.{ .alpha = 0.89232180000, .coeff = 1.0 }};
const n_d4 = [_]PrimitiveGaussian{.{ .alpha = 0.38480430000, .coeff = 1.0 }};

const n_f1 = [_]PrimitiveGaussian{.{ .alpha = 2.4229788000, .coeff = 0.69222939090 }};
const n_f2 = [_]PrimitiveGaussian{.{ .alpha = 0.83354410000, .coeff = 0.72167753900 }};

const n_g1 = [_]PrimitiveGaussian{.{ .alpha = 1.4586791000, .coeff = 1.0 }};

const n_data = [_]ShellData{
    .{ .l = 0, .primitives = &n_s1 },
    .{ .l = 0, .primitives = &n_s2 },
    .{ .l = 0, .primitives = &n_s3 },
    .{ .l = 0, .primitives = &n_s4 },
    .{ .l = 0, .primitives = &n_s5 },
    .{ .l = 0, .primitives = &n_s6 },
    .{ .l = 0, .primitives = &n_s7 },
    .{ .l = 0, .primitives = &n_s8 },
    .{ .l = 0, .primitives = &n_s9 },
    .{ .l = 0, .primitives = &n_s10 },
    .{ .l = 1, .primitives = &n_p1 },
    .{ .l = 1, .primitives = &n_p2 },
    .{ .l = 1, .primitives = &n_p3 },
    .{ .l = 1, .primitives = &n_p4 },
    .{ .l = 1, .primitives = &n_p5 },
    .{ .l = 1, .primitives = &n_p6 },
    .{ .l = 1, .primitives = &n_p7 },
    .{ .l = 1, .primitives = &n_p8 },
    .{ .l = 2, .primitives = &n_d1 },
    .{ .l = 2, .primitives = &n_d2 },
    .{ .l = 2, .primitives = &n_d3 },
    .{ .l = 2, .primitives = &n_d4 },
    .{ .l = 3, .primitives = &n_f1 },
    .{ .l = 3, .primitives = &n_f2 },
    .{ .l = 4, .primitives = &n_g1 },
};

// ============================================================================
// O (Z=8): 25 shells
// ============================================================================

const o_s1 = [_]PrimitiveGaussian{
    .{ .alpha = 625.28298110, .coeff = 0.18479249890 },
    .{ .alpha = 253.93274180, .coeff = 0.19224605780 },
    .{ .alpha = 109.04929550, .coeff = 0.59372043000 },
    .{ .alpha = 49.423005600, .coeff = 0.60593463970 },
    .{ .alpha = 23.580521100, .coeff = 0.45741933600 },
};
const o_s2 = [_]PrimitiveGaussian{.{ .alpha = 11.807759100, .coeff = 1.0 }};
const o_s3 = [_]PrimitiveGaussian{.{ .alpha = 6.1827814000, .coeff = 1.0 }};
const o_s4 = [_]PrimitiveGaussian{.{ .alpha = 3.3709061000, .coeff = 1.0 }};
const o_s5 = [_]PrimitiveGaussian{.{ .alpha = 1.9042805000, .coeff = 1.0 }};
const o_s6 = [_]PrimitiveGaussian{.{ .alpha = 1.1085447000, .coeff = 1.0 }};
const o_s7 = [_]PrimitiveGaussian{.{ .alpha = 0.66098860000, .coeff = 1.0 }};
const o_s8 = [_]PrimitiveGaussian{.{ .alpha = 0.40108140000, .coeff = 1.0 }};
const o_s9 = [_]PrimitiveGaussian{.{ .alpha = 0.24597690000, .coeff = 1.0 }};
const o_s10 = [_]PrimitiveGaussian{.{ .alpha = 0.15139390000, .coeff = 1.0 }};

const o_p1 = [_]PrimitiveGaussian{
    .{ .alpha = 77.687483800, .coeff = 0.39010104350 },
    .{ .alpha = 22.415388400, .coeff = 0.83793482660 },
    .{ .alpha = 9.8906463000, .coeff = 0.38168888150 },
};
const o_p2 = [_]PrimitiveGaussian{.{ .alpha = 5.4848863000, .coeff = 1.0 }};
const o_p3 = [_]PrimitiveGaussian{.{ .alpha = 2.9732983000, .coeff = 1.0 }};
const o_p4 = [_]PrimitiveGaussian{.{ .alpha = 1.4735260000, .coeff = 1.0 }};
const o_p5 = [_]PrimitiveGaussian{.{ .alpha = 0.73603410000, .coeff = 1.0 }};
const o_p6 = [_]PrimitiveGaussian{.{ .alpha = 0.36974140000, .coeff = 1.0 }};
const o_p7 = [_]PrimitiveGaussian{.{ .alpha = 0.18637210000, .coeff = 1.0 }};
const o_p8 = [_]PrimitiveGaussian{.{ .alpha = 0.094990600000, .coeff = 1.0 }};

const o_d1 = [_]PrimitiveGaussian{
    .{ .alpha = 37.707107400, .coeff = 0.077860015600 },
    .{ .alpha = 14.775254300, .coeff = 0.31355206270 },
    .{ .alpha = 5.8470900000, .coeff = 0.94637356360 },
};
const o_d2 = [_]PrimitiveGaussian{.{ .alpha = 2.3304365000, .coeff = 1.0 }};
const o_d3 = [_]PrimitiveGaussian{.{ .alpha = 0.93282670000, .coeff = 1.0 }};
const o_d4 = [_]PrimitiveGaussian{.{ .alpha = 0.37392850000, .coeff = 1.0 }};

const o_f1 = [_]PrimitiveGaussian{.{ .alpha = 3.0293422000, .coeff = 0.76154791140 }};
const o_f2 = [_]PrimitiveGaussian{.{ .alpha = 0.92484900000, .coeff = 0.64810861640 }};

const o_g1 = [_]PrimitiveGaussian{.{ .alpha = 1.6934809000, .coeff = 1.0 }};

const o_data = [_]ShellData{
    .{ .l = 0, .primitives = &o_s1 },
    .{ .l = 0, .primitives = &o_s2 },
    .{ .l = 0, .primitives = &o_s3 },
    .{ .l = 0, .primitives = &o_s4 },
    .{ .l = 0, .primitives = &o_s5 },
    .{ .l = 0, .primitives = &o_s6 },
    .{ .l = 0, .primitives = &o_s7 },
    .{ .l = 0, .primitives = &o_s8 },
    .{ .l = 0, .primitives = &o_s9 },
    .{ .l = 0, .primitives = &o_s10 },
    .{ .l = 1, .primitives = &o_p1 },
    .{ .l = 1, .primitives = &o_p2 },
    .{ .l = 1, .primitives = &o_p3 },
    .{ .l = 1, .primitives = &o_p4 },
    .{ .l = 1, .primitives = &o_p5 },
    .{ .l = 1, .primitives = &o_p6 },
    .{ .l = 1, .primitives = &o_p7 },
    .{ .l = 1, .primitives = &o_p8 },
    .{ .l = 2, .primitives = &o_d1 },
    .{ .l = 2, .primitives = &o_d2 },
    .{ .l = 2, .primitives = &o_d3 },
    .{ .l = 2, .primitives = &o_d4 },
    .{ .l = 3, .primitives = &o_f1 },
    .{ .l = 3, .primitives = &o_f2 },
    .{ .l = 4, .primitives = &o_g1 },
};

// ============================================================================
// F (Z=9): 25 shells
// ============================================================================

const f_s1 = [_]PrimitiveGaussian{
    .{ .alpha = 858.40986550, .coeff = 0.15724526030 },
    .{ .alpha = 329.08541150, .coeff = 0.22410411410 },
    .{ .alpha = 134.26832490, .coeff = 0.59862046130 },
    .{ .alpha = 58.197052700, .coeff = 0.64592399510 },
    .{ .alpha = 26.732570500, .coeff = 0.38663550020 },
};
const f_s2 = [_]PrimitiveGaussian{.{ .alpha = 12.973304700, .coeff = 1.0 }};
const f_s3 = [_]PrimitiveGaussian{.{ .alpha = 6.6262885000, .coeff = 1.0 }};
const f_s4 = [_]PrimitiveGaussian{.{ .alpha = 3.5457034000, .coeff = 1.0 }};
const f_s5 = [_]PrimitiveGaussian{.{ .alpha = 1.9769882000, .coeff = 1.0 }};
const f_s6 = [_]PrimitiveGaussian{.{ .alpha = 1.1415471000, .coeff = 1.0 }};
const f_s7 = [_]PrimitiveGaussian{.{ .alpha = 0.67791470000, .coeff = 1.0 }};
const f_s8 = [_]PrimitiveGaussian{.{ .alpha = 0.41094040000, .coeff = 1.0 }};
const f_s9 = [_]PrimitiveGaussian{.{ .alpha = 0.25224670000, .coeff = 1.0 }};
const f_s10 = [_]PrimitiveGaussian{.{ .alpha = 0.15548980000, .coeff = 1.0 }};

const f_p1 = [_]PrimitiveGaussian{
    .{ .alpha = 146.43308040, .coeff = 0.23153859350 },
    .{ .alpha = 49.907475200, .coeff = 0.45919101790 },
    .{ .alpha = 24.555015800, .coeff = 0.70668711020 },
};
const f_p2 = [_]PrimitiveGaussian{.{ .alpha = 10.579579200, .coeff = 0.48592881900 }};
const f_p3 = [_]PrimitiveGaussian{.{ .alpha = 5.4280075000, .coeff = 1.0 }};
const f_p4 = [_]PrimitiveGaussian{.{ .alpha = 2.9079399000, .coeff = 1.0 }};
const f_p5 = [_]PrimitiveGaussian{.{ .alpha = 1.4550833000, .coeff = 1.0 }};
const f_p6 = [_]PrimitiveGaussian{.{ .alpha = 0.73152390000, .coeff = 1.0 }};
const f_p7 = [_]PrimitiveGaussian{.{ .alpha = 0.36880500000, .coeff = 1.0 }};
const f_p8 = [_]PrimitiveGaussian{.{ .alpha = 0.18611230000, .coeff = 1.0 }};

const f_d1 = [_]PrimitiveGaussian{
    .{ .alpha = 42.931821000, .coeff = 0.094532677800 },
    .{ .alpha = 16.655135500, .coeff = 0.36468605240 },
    .{ .alpha = 6.4612572000, .coeff = 0.92631941360 },
};
const f_d2 = [_]PrimitiveGaussian{.{ .alpha = 2.5066049000, .coeff = 1.0 }};
const f_d3 = [_]PrimitiveGaussian{.{ .alpha = 0.97242190000, .coeff = 1.0 }};
const f_d4 = [_]PrimitiveGaussian{.{ .alpha = 0.37724510000, .coeff = 1.0 }};

const f_f1 = [_]PrimitiveGaussian{.{ .alpha = 3.4749889000, .coeff = 0.70247694460 }};
const f_f2 = [_]PrimitiveGaussian{.{ .alpha = 1.2194812000, .coeff = 0.71170650010 }};

const f_g1 = [_]PrimitiveGaussian{.{ .alpha = 2.0459091000, .coeff = 1.0 }};

const f_data = [_]ShellData{
    .{ .l = 0, .primitives = &f_s1 },
    .{ .l = 0, .primitives = &f_s2 },
    .{ .l = 0, .primitives = &f_s3 },
    .{ .l = 0, .primitives = &f_s4 },
    .{ .l = 0, .primitives = &f_s5 },
    .{ .l = 0, .primitives = &f_s6 },
    .{ .l = 0, .primitives = &f_s7 },
    .{ .l = 0, .primitives = &f_s8 },
    .{ .l = 0, .primitives = &f_s9 },
    .{ .l = 0, .primitives = &f_s10 },
    .{ .l = 1, .primitives = &f_p1 },
    .{ .l = 1, .primitives = &f_p2 },
    .{ .l = 1, .primitives = &f_p3 },
    .{ .l = 1, .primitives = &f_p4 },
    .{ .l = 1, .primitives = &f_p5 },
    .{ .l = 1, .primitives = &f_p6 },
    .{ .l = 1, .primitives = &f_p7 },
    .{ .l = 1, .primitives = &f_p8 },
    .{ .l = 2, .primitives = &f_d1 },
    .{ .l = 2, .primitives = &f_d2 },
    .{ .l = 2, .primitives = &f_d3 },
    .{ .l = 2, .primitives = &f_d4 },
    .{ .l = 3, .primitives = &f_f1 },
    .{ .l = 3, .primitives = &f_f2 },
    .{ .l = 4, .primitives = &f_g1 },
};

// ============================================================================
// Tests
// ============================================================================

test "even-tempered aux basis" {
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const result = buildEvenTemperedAux(center, 2, 3, 0.1, 3.0);

    // l=0: 3 shells, l=1: 3 shells, l=2: 3 shells = 9 total
    try std.testing.expectEqual(@as(usize, 9), result.count);

    // Check exponents for l=0
    try std.testing.expectApproxEqAbs(0.1, result.shells[0].primitives[0].alpha, 1e-10);
    try std.testing.expectApproxEqAbs(0.3, result.shells[1].primitives[0].alpha, 1e-10);
    try std.testing.expectApproxEqAbs(0.9, result.shells[2].primitives[0].alpha, 1e-10);

    // Check angular momentum
    try std.testing.expectEqual(@as(u32, 0), result.shells[0].l);
    try std.testing.expectEqual(@as(u32, 1), result.shells[3].l);
    try std.testing.expectEqual(@as(u32, 2), result.shells[6].l);
}

test "def2-universal-jkfit H" {
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const result = buildDef2UniversalJkfit(1, center).?;
    try std.testing.expectEqual(@as(usize, 6), result.count);

    // Count by angular momentum
    var l_count = [_]usize{ 0, 0, 0, 0, 0 };
    for (result.shells[0..result.count]) |s| {
        l_count[s.l] += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), l_count[0]); // s
    try std.testing.expectEqual(@as(usize, 2), l_count[1]); // p
    try std.testing.expectEqual(@as(usize, 2), l_count[2]); // d
}

test "def2-universal-jkfit O" {
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const result = buildDef2UniversalJkfit(8, center).?;
    try std.testing.expectEqual(@as(usize, 25), result.count);

    var l_count = [_]usize{ 0, 0, 0, 0, 0 };
    for (result.shells[0..result.count]) |s| {
        l_count[s.l] += 1;
    }
    try std.testing.expectEqual(@as(usize, 10), l_count[0]); // s
    try std.testing.expectEqual(@as(usize, 8), l_count[1]); // p
    try std.testing.expectEqual(@as(usize, 4), l_count[2]); // d
    try std.testing.expectEqual(@as(usize, 2), l_count[3]); // f
    try std.testing.expectEqual(@as(usize, 1), l_count[4]); // g
}

test "def2-universal-jkfit total basis functions" {
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };

    // H: 2×1(s) + 2×3(p) + 2×6(d) = 2 + 6 + 12 = 20 Cartesian functions
    const h_result = buildDef2UniversalJkfit(1, center).?;
    var h_nbf: usize = 0;
    for (h_result.shells[0..h_result.count]) |s| {
        h_nbf += gaussian.numCartesian(s.l);
    }
    try std.testing.expectEqual(@as(usize, 20), h_nbf);

    // O: 10×1(s) + 8×3(p) + 4×6(d) + 2×10(f) + 1×15(g) = 10+24+24+20+15 = 93
    const o_result = buildDef2UniversalJkfit(8, center).?;
    var o_nbf: usize = 0;
    for (o_result.shells[0..o_result.count]) |s| {
        o_nbf += gaussian.numCartesian(s.l);
    }
    try std.testing.expectEqual(@as(usize, 93), o_nbf);
}

test "def2-universal-jkfit unsupported element" {
    const center = math.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    try std.testing.expect(buildDef2UniversalJkfit(26, center) == null); // Fe not supported
}
