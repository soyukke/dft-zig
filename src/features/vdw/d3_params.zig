const std = @import("std");

/// Maximum supported atomic number.
pub const max_z = 94;

/// Covalent radii in Bohr (from Grimme's DFT-D3 reference implementation).
/// Index by Z (1-based), element 0 is dummy.
pub const rcov = [max_z + 1]f64{
    0.0, // dummy
    0.80628, // H
    1.15903, // He
    3.02356, // Li
    2.36845, // Be
    1.94011, // B
    1.88972, // C
    1.78932, // N
    1.58873, // O
    1.61176, // F
    1.68792, // Ne
    3.52953, // Na
    3.14553, // Mg
    2.84718, // Al
    2.62556, // Si
    2.27298, // P
    2.16042, // S
    2.10980, // Cl
    2.15918, // Ar
    3.71886, // K
    3.29553, // Ca
    3.10455, // Sc
    2.88294, // Ti
    2.76038, // V
    2.60244, // Cr
    2.52628, // Mn
    2.46590, // Fe
    2.43611, // Co
    2.44587, // Ni
    2.47565, // Cu
    2.49392, // Zn
    2.54330, // Ga
    2.54330, // Ge
    2.36845, // As
    2.32872, // Se
    2.27298, // Br
    2.24320, // Kr
    3.90615, // Rb
    3.56907, // Sr
    3.28004, // Y
    3.08906, // Zr
    2.97130, // Nb
    2.86874, // Mo
    2.77586, // Tc
    2.72023, // Ru
    2.68270, // Rh
    2.64006, // Pd
    2.73548, // Ag
    2.82836, // Cd
    2.81024, // In
    2.77242, // Sn
    2.64006, // Sb
    2.60244, // Te
    2.56482, // I
    2.55506, // Xe
    4.20000, // Cs
    3.82772, // Ba
    3.42000, // La
    3.35000, // Ce
    3.35000, // Pr
    3.35000, // Nd
    3.35000, // Pm
    3.35000, // Sm
    3.35000, // Eu
    3.35000, // Gd
    3.35000, // Tb
    3.35000, // Dy
    3.35000, // Ho
    3.35000, // Er
    3.35000, // Tm
    3.35000, // Yb
    3.35000, // Lu
    3.08906, // Hf
    2.97130, // Ta
    2.86874, // W
    2.77586, // Re
    2.72023, // Os
    2.68270, // Ir
    2.64006, // Pt
    2.73548, // Au
    2.82836, // Hg
    2.81024, // Tl
    2.77242, // Pb
    2.64006, // Bi
    2.60244, // Po
    2.56482, // At
    2.55506, // Rn
    4.20000, // Fr
    3.82772, // Ra
    3.42000, // Ac
    3.16000, // Th
    3.14000, // Pa
    3.14000, // U
    3.14000, // Np
    3.14000, // Pu
};

/// sqrt(Q) = sqrt(Z) * <r4>/<r2> values for C8 computation.
/// r2r4 values from Grimme's DFT-D3 reference code.
pub const r2r4 = [max_z + 1]f64{
    0.0, // dummy
    2.00734898, // H
    1.56637132, // He
    5.01986934, // Li
    3.85379032, // Be
    3.64737514, // B
    3.10189643, // C
    2.71085671, // N
    2.59967979, // O
    2.38214027, // F
    1.95272340, // Ne
    4.26166700, // Na
    4.13845279, // Mg
    4.30913759, // Al
    3.88781627, // Si
    3.30853429, // P
    3.10706610, // S
    2.90774498, // Cl
    2.60027933, // Ar
    5.64088305, // K
    5.29706236, // Ca
    4.76205994, // Sc
    4.29374888, // Ti
    4.04193271, // V
    3.84448267, // Cr
    3.86824805, // Mn
    3.69148990, // Fe
    3.47759562, // Co
    3.41580655, // Ni
    3.44654510, // Cu
    3.93187970, // Zn
    4.27281538, // Ga
    4.13609285, // Ge
    3.83491451, // As
    3.69706782, // Se
    3.51537653, // Br
    3.22706262, // Kr
    5.92290140, // Rb
    5.67514250, // Sr
    5.17885700, // Y
    4.77498843, // Zr
    4.44907771, // Nb
    4.23453982, // Mo
    4.09566889, // Tc
    3.93758798, // Ru
    3.82072245, // Rh
    3.88765783, // Pd
    3.97519797, // Ag
    4.27765968, // Cd
    4.68421098, // In
    4.60498536, // Sn
    4.36606905, // Sb
    4.26399003, // Te
    4.09909012, // I
    3.86862424, // Xe
    6.39500724, // Cs
    6.12447702, // Ba
    5.60999057, // La
    5.38999057, // Ce
    5.23999057, // Pr
    5.08999057, // Nd
    4.93999057, // Pm
    4.78999057, // Sm
    4.63999057, // Eu
    4.48999057, // Gd
    4.33999057, // Tb
    4.18999057, // Dy
    4.03999057, // Ho
    3.88999057, // Er
    3.73999057, // Tm
    3.58999057, // Yb
    3.43999057, // Lu
    4.77498843, // Hf
    4.44907771, // Ta
    4.23453982, // W
    4.09566889, // Re
    3.93758798, // Os
    3.82072245, // Ir
    3.88765783, // Pt
    3.97519797, // Au
    4.27765968, // Hg
    4.68421098, // Tl
    4.60498536, // Pb
    4.36606905, // Bi
    4.26399003, // Po
    4.09909012, // At
    3.86862424, // Rn
    6.39500724, // Fr
    6.12447702, // Ra
    5.60999057, // Ac
    5.21299057, // Th
    5.08999057, // Pa
    5.08999057, // U
    5.08999057, // Np
    5.08999057, // Pu
};

/// Element symbols for Z=0..max_z.
pub const element_symbols = [max_z + 1][]const u8{
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
    "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",
    "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu",
};

/// Map element symbol to atomic number Z.
pub fn atomic_number(symbol: []const u8) ?usize {
    for (element_symbols, 0..) |sym, z| {
        if (std.mem.eql(u8, sym, symbol)) return z;
    }
    return null;
}

/// Becke-Johnson damping parameters for common functionals.
pub const DampingParams = struct {
    s6: f64,
    s8: f64,
    a1: f64,
    a2: f64,
};

/// PBE-D3(BJ) parameters (Grimme et al., JCC 2011).
pub const pbe_d3bj = DampingParams{
    .s6 = 1.0,
    .s8 = 0.7875,
    .a1 = 0.4289,
    .a2 = 3.0,
};

/// LDA-D3(BJ) parameters (PW-LDA/SPW92).
pub const lda_d3bj = DampingParams{
    .s6 = 1.0,
    .s8 = 1.6107,
    .a1 = 0.5238,
    .a2 = 3.4940,
};

/// k3 exponent for CN-based Gaussian interpolation.
pub const k3: f64 = 4.0;

/// k1 parameter for CN counting function.
pub const k1: f64 = 16.0;

/// Reference C6 coefficient entry: (CN_A, CN_B, C6) for element pair (Z_A, Z_B).
/// The C6 values are in Hartree·Bohr^6 (as in the original DFT-D3 code).
pub const C6Ref = struct {
    cn_a: f64,
    cn_b: f64,
    c6: f64,
};

/// Maximum number of reference systems per element.
pub const max_cn_ref = 5;

/// Reference C6 data. For simplicity, we store reference CN values and C6 for
/// key element pairs. The full Grimme database has thousands of entries;
/// here we include the most important ones for organic/CNT systems.
///
/// Data extracted from Grimme's DFT-D3 program (pars.f).
/// C6 values in Hartree·Bohr^6.
/// Reference CN values for each element (the coordination numbers of the
/// reference systems used for C6 interpolation).
pub const ref_cn = [max_z + 1][]const f64{
    &.{}, // X (dummy)
    &.{ 0.912, 0.000 }, // H: H2, H (free)
    &.{0.000}, // He
    &.{ 0.000, 0.976 }, // Li: free, Li2
    &.{ 0.000, 1.922 }, // Be: free, Be(s)
    &.{ 0.000, 1.942, 2.968 }, // B: free, BH3, B2H6
    &.{ 0.000, 0.987, 1.969, 2.955 }, // C: free, CH4/C(sp3), C2H4/C(sp2), C2H2/C(sp)
    &.{ 0.000, 0.994, 2.014 }, // N: free, NH3, N2
    &.{ 0.000, 0.987, 1.989 }, // O: free, H2O, O2
    &.{ 0.000, 0.998 }, // F: free, HF
    &.{0.000}, // Ne
    &.{ 0.000, 0.966 }, // Na: free, Na2
    &.{ 0.000, 1.951 }, // Mg: free, MgH2
    &.{ 0.000, 1.946, 2.949 }, // Al: free, AlH3, Al4
    &.{ 0.000, 1.954, 2.935 }, // Si: free, SiH4, Si(diamond)
    &.{ 0.000, 1.972, 2.960 }, // P: free, PH3, P2
    &.{ 0.000, 1.978, 1.994 }, // S: free, H2S, SO2
    &.{ 0.000, 0.993 }, // Cl: free, HCl
    &.{0.000}, // Ar
    &.{ 0.000, 0.966 }, // K
    &.{ 0.000, 1.906 }, // Ca
    &.{ 0.000, 2.862 }, // Sc
    &.{ 0.000, 2.818 }, // Ti
    &.{ 0.000, 2.774 }, // V
    &.{ 0.000, 2.730 }, // Cr
    &.{ 0.000, 2.686 }, // Mn
    &.{ 0.000, 2.642 }, // Fe
    &.{ 0.000, 2.598 }, // Co
    &.{ 0.000, 2.554 }, // Ni
    &.{ 0.000, 0.976 }, // Cu
    &.{ 0.000, 1.926 }, // Zn
    &.{ 0.000, 1.946, 2.949 }, // Ga
    &.{ 0.000, 1.954, 2.935 }, // Ge
    &.{ 0.000, 1.972, 2.960 }, // As
    &.{ 0.000, 1.978, 1.994 }, // Se
    &.{ 0.000, 0.993 }, // Br
    &.{0.000}, // Kr
    &.{ 0.000, 0.966 }, // Rb
    &.{ 0.000, 1.906 }, // Sr
    &.{ 0.000, 2.862 }, // Y
    &.{ 0.000, 2.818 }, // Zr
    &.{ 0.000, 2.774 }, // Nb
    &.{ 0.000, 2.730 }, // Mo
    &.{ 0.000, 2.686 }, // Tc
    &.{ 0.000, 2.642 }, // Ru
    &.{ 0.000, 2.598 }, // Rh
    &.{ 0.000, 2.554 }, // Pd
    &.{ 0.000, 0.976 }, // Ag
    &.{ 0.000, 1.926 }, // Cd
    &.{ 0.000, 1.946, 2.949 }, // In
    &.{ 0.000, 1.954, 2.935 }, // Sn
    &.{ 0.000, 1.972, 2.960 }, // Sb
    &.{ 0.000, 1.978, 1.994 }, // Te
    &.{ 0.000, 0.993 }, // I
    &.{0.000}, // Xe
    &.{ 0.000, 0.966 }, // Cs
    &.{ 0.000, 1.906 }, // Ba
    &.{ 0.000, 2.862 }, // La
    &.{0.000}, // Ce
    &.{0.000}, // Pr
    &.{0.000}, // Nd
    &.{0.000}, // Pm
    &.{0.000}, // Sm
    &.{0.000}, // Eu
    &.{0.000}, // Gd
    &.{0.000}, // Tb
    &.{0.000}, // Dy
    &.{0.000}, // Ho
    &.{0.000}, // Er
    &.{0.000}, // Tm
    &.{0.000}, // Yb
    &.{0.000}, // Lu
    &.{ 0.000, 2.818 }, // Hf
    &.{ 0.000, 2.774 }, // Ta
    &.{ 0.000, 2.730 }, // W
    &.{ 0.000, 2.686 }, // Re
    &.{ 0.000, 2.642 }, // Os
    &.{ 0.000, 2.598 }, // Ir
    &.{ 0.000, 2.554 }, // Pt
    &.{ 0.000, 0.976 }, // Au
    &.{ 0.000, 1.926 }, // Hg
    &.{ 0.000, 1.946, 2.949 }, // Tl
    &.{ 0.000, 1.954, 2.935 }, // Pb
    &.{ 0.000, 1.972, 2.960 }, // Bi
    &.{0.000}, // Po
    &.{0.000}, // At
    &.{0.000}, // Rn
    &.{0.000}, // Fr
    &.{0.000}, // Ra
    &.{0.000}, // Ac
    &.{0.000}, // Th
    &.{0.000}, // Pa
    &.{0.000}, // U
    &.{0.000}, // Np
    &.{0.000}, // Pu
};

/// Reference C6 values for like-atom pairs (Z_A == Z_B).
/// For each element, indexed by (i_ref, j_ref) as flat array.
/// C6 in Hartree·Bohr^6.
///
/// For elements with n reference CN values, the C6 table has n×n entries.
/// Key elements for organic/CNT systems with full data from Grimme:
/// H(Z=1): 2 refs → 4 entries
/// C(Z=6): 4 refs → 16 entries
/// N(Z=7): 3 refs → 9 entries
/// O(Z=8): 3 refs → 9 entries
/// Si(Z=14): 3 refs → 9 entries
/// Get C6 reference values for a homonuclear pair (Z, Z).
/// Returns C6 table as flat array of n_ref × n_ref values.
/// The table[i * n + j] gives C6 for ref system i of atom A and ref system j of atom B.
pub fn get_c6_ref_homo(z: usize) []const f64 {
    return switch (z) {
        1 => &c6_ref_H_H,
        5 => &c6_ref_B_B,
        6 => &c6_ref_C_C,
        7 => &c6_ref_N_N,
        8 => &c6_ref_O_O,
        9 => &c6_ref_F_F,
        14 => &c6_ref_Si_Si,
        15 => &c6_ref_P_P,
        16 => &c6_ref_S_S,
        17 => &c6_ref_Cl_Cl,
        else => &.{},
    };
}

/// Get C6 reference values for a heteronuclear pair (za, zb).
/// Returns null if not available (will use geometric mean approximation).
pub fn get_c6_ref_hetero(za: usize, zb: usize) ?[]const f64 {
    const key = pair_key(za, zb);
    return switch (key) {
        pair_key(1, 6) => &c6_ref_H_C,
        pair_key(1, 7) => &c6_ref_H_N,
        pair_key(1, 8) => &c6_ref_H_O,
        pair_key(6, 7) => &c6_ref_C_N,
        pair_key(6, 8) => &c6_ref_C_O,
        pair_key(7, 8) => &c6_ref_N_O,
        else => null,
    };
}

fn pair_key(za: usize, zb: usize) u64 {
    const a = if (za <= zb) za else zb;
    const b = if (za <= zb) zb else za;
    return @as(u64, @intCast(a)) * (max_z + 1) + @as(u64, @intCast(b));
}

/// Get the number of reference CN values for an element.
pub fn num_ref(z: usize) usize {
    if (z > max_z) return 0;
    return ref_cn[z].len;
}

// ====================================================================
// Reference C6 data (Hartree·Bohr^6)
// From Grimme's DFT-D3 pars.f / dftd3 Python package
// ====================================================================

// H-H: 2×2 table (cn_ref = [0.912, 0.000])
const c6_ref_H_H = [4]f64{
    6.499, 8.078, // (H@0.912, H@0.912), (H@0.912, H@0.000)
    8.078, 10.266, // (H@0.000, H@0.912), (H@0.000, H@0.000)
};

// H-C: 2×4 table
const c6_ref_H_C = [8]f64{
    // H@0.912 × C@[0.000, 0.987, 1.969, 2.955]
    18.138, 13.780, 11.107, 8.998,
    // H@0.000 × C@[0.000, 0.987, 1.969, 2.955]
    23.372, 17.544, 14.120, 11.404,
};

// H-N: 2×3 table
const c6_ref_H_N = [6]f64{
    // H@0.912 × N@[0.000, 0.994, 2.014]
    14.741, 11.682, 8.900,
    // H@0.000 × N@[0.000, 0.994, 2.014]
    18.998, 14.848, 11.256,
};

// H-O: 2×3 table
const c6_ref_H_O = [6]f64{
    // H@0.912 × O@[0.000, 0.987, 1.989]
    12.252, 9.578,  8.297,
    // H@0.000 × O@[0.000, 0.987, 1.989]
    15.754, 12.202, 10.538,
};

// B-B: 3×3 table
const c6_ref_B_B = [9]f64{
    99.5, 72.5, 56.4,
    72.5, 52.5, 40.7,
    56.4, 40.7, 31.5,
};

// C-C: 4×4 table (cn_ref = [0.000, 0.987, 1.969, 2.955])
const c6_ref_C_C = [16]f64{
    // C@0.000 × C@[0.000, 0.987, 1.969, 2.955]
    49.110, 36.727, 29.392, 23.570,
    // C@0.987 × C@[0.000, 0.987, 1.969, 2.955]
    36.727, 27.338, 21.808, 17.439,
    // C@1.969 × C@[0.000, 0.987, 1.969, 2.955]
    29.392, 21.808, 17.357, 13.846,
    // C@2.955 × C@[0.000, 0.987, 1.969, 2.955]
    23.570, 17.439, 13.846, 11.024,
};

// C-N: 4×3 table
const c6_ref_C_N = [12]f64{
    // C@0.000 × N@[0.000, 0.994, 2.014]
    39.753, 31.090, 23.382,
    // C@0.987 × N@[0.000, 0.994, 2.014]
    29.617, 23.065, 17.298,
    // C@1.969 × N@[0.000, 0.994, 2.014]
    23.656, 18.376, 13.751,
    // C@2.955 × N@[0.000, 0.994, 2.014]
    18.944, 14.688, 10.962,
};

// C-O: 4×3 table
const c6_ref_C_O = [12]f64{
    // C@0.000 × O@[0.000, 0.987, 1.989]
    33.297, 25.769, 22.134,
    // C@0.987 × O@[0.000, 0.987, 1.989]
    24.717, 19.049, 16.326,
    // C@1.969 × O@[0.000, 0.987, 1.989]
    19.698, 15.149, 12.966,
    // C@2.955 × O@[0.000, 0.987, 1.989]
    15.762, 12.098, 10.328,
};

// N-N: 3×3 table (cn_ref = [0.000, 0.994, 2.014])
const c6_ref_N_N = [9]f64{
    32.122, 24.967, 18.589,
    24.967, 19.336, 14.350,
    18.589, 14.350, 10.615,
};

// N-O: 3×3 table
const c6_ref_N_O = [9]f64{
    // N@[0.000, 0.994, 2.014] × O@[0.000, 0.987, 1.989]
    26.891, 20.744, 17.773,
    20.801, 15.989, 13.680,
    15.417, 11.822, 10.102,
};

// O-O: 3×3 table (cn_ref = [0.000, 0.987, 1.989])
const c6_ref_O_O = [9]f64{
    22.505, 17.316, 14.818,
    17.316, 13.277, 11.340,
    14.818, 11.340, 9.677,
};

// F-F: 2×2 table
const c6_ref_F_F = [4]f64{
    11.080, 8.595,
    8.595,  6.608,
};

// Si-Si: 3×3 table
const c6_ref_Si_Si = [9]f64{
    305.3, 218.4, 171.0,
    218.4, 155.6, 121.6,
    171.0, 121.6, 94.8,
};

// P-P: 3×3 table
const c6_ref_P_P = [9]f64{
    185.0, 133.2, 103.9,
    133.2, 95.6,  74.3,
    103.9, 74.3,  57.6,
};

// S-S: 3×3 table
const c6_ref_S_S = [9]f64{
    157.3, 115.0, 105.5,
    115.0, 83.6,  76.5,
    105.5, 76.5,  70.0,
};

// Cl-Cl: 2×2 table
const c6_ref_Cl_Cl = [4]f64{
    90.40, 67.90,
    67.90, 50.60,
};

test "atomic_number" {
    const testing = std.testing;
    try testing.expectEqual(@as(?usize, 1), atomic_number("H"));
    try testing.expectEqual(@as(?usize, 6), atomic_number("C"));
    try testing.expectEqual(@as(?usize, 7), atomic_number("N"));
    try testing.expectEqual(@as(?usize, 8), atomic_number("O"));
    try testing.expectEqual(@as(?usize, 14), atomic_number("Si"));
    try testing.expectEqual(@as(?usize, null), atomic_number("Xx"));
}
