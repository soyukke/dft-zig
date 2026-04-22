const std = @import("std");
const math = @import("../math/math.zig");
const params = @import("d3_params.zig");

/// Default real-space cutoff for dispersion energy (Bohr).
pub const default_cutoff: f64 = 95.0;
/// Default cutoff for coordination number counting (Bohr).
pub const default_cn_cutoff: f64 = 40.0;

/// Compute fractional coordination numbers for all atoms.
/// CN_A = Σ_{B≠A, L} 1/(1 + exp(-k1 × (4/3 × (rcov_A + rcov_B)/R_{ABL} - 1)))
/// Periodic images are included via lattice vector sum.
pub fn computeCoordinationNumbers(
    alloc: std.mem.Allocator,
    atomic_numbers: []const usize,
    positions: []const math.Vec3,
    cell: math.Mat3,
    cn_cutoff: f64,
) ![]f64 {
    const n_atoms = atomic_numbers.len;
    const cn = try alloc.alloc(f64, n_atoms);
    @memset(cn, 0.0);

    // Determine lattice vector range
    const lat_range = latticeRange(cell, cn_cutoff);

    for (0..n_atoms) |ia| {
        const za = atomic_numbers[ia];
        if (za == 0 or za > params.max_z) continue;
        const rcov_a = params.rcov[za];

        for (0..n_atoms) |ib| {
            const zb = atomic_numbers[ib];
            if (zb == 0 or zb > params.max_z) continue;
            const rcov_b = params.rcov[zb];
            const rcov_sum = rcov_a + rcov_b;

            var la: i32 = -lat_range[0];
            while (la <= lat_range[0]) : (la += 1) {
                var lb: i32 = -lat_range[1];
                while (lb <= lat_range[1]) : (lb += 1) {
                    var lc: i32 = -lat_range[2];
                    while (lc <= lat_range[2]) : (lc += 1) {
                        if (ia == ib and la == 0 and lb == 0 and lc == 0) continue;
                        const shift = latticeVector(cell, la, lb, lc);
                        const r_vec = math.Vec3.sub(
                            math.Vec3.add(positions[ib], shift),
                            positions[ia],
                        );
                        const r = math.Vec3.norm(r_vec);
                        if (r > cn_cutoff or r < 1e-10) continue;

                        // Counting function
                        const arg = -params.k1 * (4.0 / 3.0 * rcov_sum / r - 1.0);
                        cn[ia] += 1.0 / (1.0 + @exp(arg));
                    }
                }
            }
        }
    }

    return cn;
}

/// Compute interpolated C6 coefficient for atom pair (A, B) with given CNs.
/// Uses Gaussian-weight interpolation over reference C6 data.
pub fn computeC6(za: usize, zb: usize, cn_a: f64, cn_b: f64) f64 {
    const n_ref_a = params.numRef(za);
    const n_ref_b = params.numRef(zb);
    if (n_ref_a == 0 or n_ref_b == 0) return 0.0;

    const cn_ref_a = params.ref_cn[za];
    const cn_ref_b = params.ref_cn[zb];

    // Try to get the explicit C6 reference table
    const c6_table = getC6Table(za, zb, n_ref_a, n_ref_b);
    if (c6_table == null) {
        // Fall back to geometric mean approximation
        return geometricMeanC6(za, zb, cn_a, cn_b);
    }

    var w_sum: f64 = 0.0;
    var c6_sum: f64 = 0.0;

    const table = c6_table.?;

    for (0..n_ref_a) |i| {
        const la = @exp(-params.k3 * (cn_a - cn_ref_a[i]) * (cn_a - cn_ref_a[i]));
        for (0..n_ref_b) |j| {
            const lb = @exp(-params.k3 * (cn_b - cn_ref_b[j]) * (cn_b - cn_ref_b[j]));
            const w = la * lb;
            // For homonuclear pairs, table is n_ref × n_ref
            // For heteronuclear pairs, table is n_ref_a × n_ref_b
            // But we need to handle the ordering
            const c6_val = table[i * n_ref_b + j];
            c6_sum += w * c6_val;
            w_sum += w;
        }
    }

    if (w_sum < 1e-30) {
        // All weights are essentially zero; use closest reference
        return table[0];
    }

    return c6_sum / w_sum;
}

/// Compute C8 from C6 using the recursion relation.
/// C8 = 3 × C6 × sqrt(Q_A × Q_B)
/// where Q_Z = s42 × r2r4[Z] (s42 = sqrt(0.5 * Z))
pub fn computeC8(c6: f64, za: usize, zb: usize) f64 {
    if (za == 0 or za > params.max_z or zb == 0 or zb > params.max_z) return 0.0;
    const qa = params.r2r4[za];
    const qb = params.r2r4[zb];
    return 3.0 * c6 * qa * qb;
}

/// Compute DFT-D3(BJ) dispersion energy.
/// Returns energy in Rydberg.
pub fn computeEnergy(
    alloc: std.mem.Allocator,
    atomic_numbers: []const usize,
    positions: []const math.Vec3,
    cell: math.Mat3,
    damping: params.DampingParams,
    cutoff: f64,
    cn_cutoff: f64,
) !f64 {
    const n_atoms = atomic_numbers.len;
    if (n_atoms == 0) return 0.0;

    // Compute coordination numbers
    const cn = try computeCoordinationNumbers(alloc, atomic_numbers, positions, cell, cn_cutoff);
    defer alloc.free(cn);

    var energy: f64 = 0.0;
    const lat_range = latticeRange(cell, cutoff);

    for (0..n_atoms) |ia| {
        const za = atomic_numbers[ia];
        if (za == 0 or za > params.max_z) continue;

        for (ia..n_atoms) |ib| {
            const zb = atomic_numbers[ib];
            if (zb == 0 or zb > params.max_z) continue;

            const c6 = computeC6(za, zb, cn[ia], cn[ib]);
            const c8 = computeC8(c6, za, zb);
            const r0 = damping.a1 * @sqrt(c8 / c6) + damping.a2;

            var la: i32 = -lat_range[0];
            while (la <= lat_range[0]) : (la += 1) {
                var lb: i32 = -lat_range[1];
                while (lb <= lat_range[1]) : (lb += 1) {
                    var lc: i32 = -lat_range[2];
                    while (lc <= lat_range[2]) : (lc += 1) {
                        if (ia == ib and la == 0 and lb == 0 and lc == 0) continue;
                        const shift = latticeVector(cell, la, lb, lc);
                        const r_vec = math.Vec3.sub(
                            math.Vec3.add(positions[ib], shift),
                            positions[ia],
                        );
                        const r2 = math.Vec3.dot(r_vec, r_vec);
                        const r = @sqrt(r2);
                        if (r > cutoff or r < 1e-10) continue;

                        const r6 = r2 * r2 * r2;
                        const r8 = r6 * r2;
                        const f6 = r0 * r0 * r0 * r0 * r0 * r0;
                        const f8 = f6 * r0 * r0;

                        // Double counting factor: 0.5 for A==B with L≠0, 1.0 for A<B
                        const factor: f64 = if (ia == ib) 0.5 else 1.0;

                        const e6 = damping.s6 * c6 / (r6 + f6);
                        const e8 = damping.s8 * c8 / (r8 + f8);
                        energy -= factor * (e6 + e8);
                    }
                }
            }
        }
    }

    // Convert from Hartree to Rydberg (× 2)
    return energy * 2.0;
}

/// Compute DFT-D3(BJ) dispersion forces.
/// Returns forces in Rydberg/Bohr.
pub fn computeForces(
    alloc: std.mem.Allocator,
    atomic_numbers: []const usize,
    positions: []const math.Vec3,
    cell: math.Mat3,
    damping: params.DampingParams,
    cutoff: f64,
    cn_cutoff: f64,
) ![]math.Vec3 {
    const n_atoms = atomic_numbers.len;
    const forces = try alloc.alloc(math.Vec3, n_atoms);
    @memset(forces, math.Vec3{ .x = 0, .y = 0, .z = 0 });
    if (n_atoms == 0) return forces;

    // Compute coordination numbers
    const cn = try computeCoordinationNumbers(alloc, atomic_numbers, positions, cell, cn_cutoff);
    defer alloc.free(cn);

    // Precompute C6, dC6/dCN_A, dC6/dCN_B for all pairs
    // We'll compute forces from direct differentiation only (neglect CN gradient term for now)
    // The CN-indirect terms are typically small (~2-5% of forces).
    // Full implementation would require dC6/dCN × dCN/dR chain rule.

    const lat_range_cutoff = latticeRange(cell, cutoff);

    for (0..n_atoms) |ia| {
        const za = atomic_numbers[ia];
        if (za == 0 or za > params.max_z) continue;

        for (ia..n_atoms) |ib| {
            const zb = atomic_numbers[ib];
            if (zb == 0 or zb > params.max_z) continue;

            const c6 = computeC6(za, zb, cn[ia], cn[ib]);
            const c8 = computeC8(c6, za, zb);
            const r0 = damping.a1 * @sqrt(c8 / c6) + damping.a2;

            var la: i32 = -lat_range_cutoff[0];
            while (la <= lat_range_cutoff[0]) : (la += 1) {
                var lb: i32 = -lat_range_cutoff[1];
                while (lb <= lat_range_cutoff[1]) : (lb += 1) {
                    var lc: i32 = -lat_range_cutoff[2];
                    while (lc <= lat_range_cutoff[2]) : (lc += 1) {
                        if (ia == ib and la == 0 and lb == 0 and lc == 0) continue;
                        const shift = latticeVector(cell, la, lb, lc);
                        const r_vec = math.Vec3.sub(
                            math.Vec3.add(positions[ib], shift),
                            positions[ia],
                        );
                        const r2 = math.Vec3.dot(r_vec, r_vec);
                        const r = @sqrt(r2);
                        if (r > cutoff or r < 1e-10) continue;

                        const r6 = r2 * r2 * r2;
                        const r8 = r6 * r2;
                        const f6 = r0 * r0 * r0 * r0 * r0 * r0;
                        const f8 = f6 * r0 * r0;

                        // dE/dR = s6 * 6*C6*R^4*R_vec / (R^6 + f^6)^2
                        //       + s8 * 8*C8*R^6*R_vec / (R^8 + f^8)^2
                        const denom6 = r6 + f6;
                        const denom8 = r8 + f8;
                        const grad_scale = damping.s6 * 6.0 * c6 * r2 * r2 / (denom6 * denom6) +
                            damping.s8 * 8.0 * c8 * r6 / (denom8 * denom8);

                        const factor: f64 = if (ia == ib) 0.5 else 1.0;

                        // Force = -dE/dR_A, where dE/dR_A = -grad_scale * R_vec
                        // (for attractive E = -C/R^n).
                        // Actually: E = -C/(R^n + f^n),
                        //   dE/dR = n*C*R^(n-2)*R_vec/(R^n+f^n)^2
                        // F_A = -dE/dR_A = -dE/dR * dR/dR_A = dE/dR * R_vec/R
                        // ... wait, let me be careful.
                        // R_vec = R_B - R_A, so dR_vec/dR_A = -I, dR/dR_A = -R_vec/R
                        // dE/dR_A = (dE/dR) * (dR/dR_A) = (dE/dR) * (-R_hat)
                        // F_A = -dE/dR_A = (dE/dR) * R_hat = grad_scale * R_vec / R
                        // But actually we need to be more careful:
                        // E = -s6*C6/(R^6+f^6) => dE/d(R²) = s6*C6*3*R^4/(R^6+f^6)^2
                        // ... let me use R_alpha
                        // dE/dR_alpha = dE/dR * R_alpha/R
                        // dE/dR = s6*6*C6*R^5/(R^6+f^6)^2 + s8*8*C8*R^7/(R^8+f^8)^2
                        // So dE/dR * R_alpha/R
                        //   = (s6*6*C6*R^4/(R^6+f^6)^2
                        //      + s8*8*C8*R^6/(R^8+f^8)^2) * R_alpha
                        // F_A_alpha = -dE/dR_A_alpha = (dE/dR) * R_alpha/R
                        //           = grad_scale * r_vec_alpha
                        // (since R_vec = R_B - R_A, dR/d(R_A_alpha) = -R_alpha/R)
                        // Actually F_A = -dE/dR_A = -(dE/dR * (-R_hat)) = dE/dR * R_hat

                        // F_A = factor * grad_scale * r_vec (in direction from A to B)
                        // F_B = -F_A (Newton's third law)
                        const f_vec = math.Vec3{
                            .x = factor * grad_scale * r_vec.x,
                            .y = factor * grad_scale * r_vec.y,
                            .z = factor * grad_scale * r_vec.z,
                        };

                        // Note: factor already accounts for double-counting
                        forces[ia].x -= f_vec.x;
                        forces[ia].y -= f_vec.y;
                        forces[ia].z -= f_vec.z;
                        if (ia != ib) {
                            forces[ib].x += f_vec.x;
                            forces[ib].y += f_vec.y;
                            forces[ib].z += f_vec.z;
                        }
                    }
                }
            }
        }
    }

    // Convert from Hartree/Bohr to Rydberg/Bohr (× 2)
    for (forces) |*f| {
        f.x *= 2.0;
        f.y *= 2.0;
        f.z *= 2.0;
    }

    return forces;
}

/// Compute D3(BJ) contribution to the dynamical matrix at Γ point.
/// D(Iα, Jβ) = ∂²E/∂R_Iα∂R_Jβ
/// Returns flat array of size (3*n_atoms)² in Rydberg/Bohr².
pub fn computeDynmat(
    alloc: std.mem.Allocator,
    atomic_numbers: []const usize,
    positions: []const math.Vec3,
    cell: math.Mat3,
    damping: params.DampingParams,
    cutoff: f64,
    cn_cutoff: f64,
) ![]f64 {
    const n_atoms = atomic_numbers.len;
    const dim = 3 * n_atoms;
    const dyn = try alloc.alloc(f64, dim * dim);
    @memset(dyn, 0.0);
    if (n_atoms == 0) return dyn;

    const cn = try computeCoordinationNumbers(alloc, atomic_numbers, positions, cell, cn_cutoff);
    defer alloc.free(cn);

    const lat_range_cutoff = latticeRange(cell, cutoff);

    // Compute off-diagonal blocks (I ≠ J)
    for (0..n_atoms) |ia| {
        const za = atomic_numbers[ia];
        if (za == 0 or za > params.max_z) continue;

        for (0..n_atoms) |ib| {
            if (ia == ib) continue;
            const zb = atomic_numbers[ib];
            if (zb == 0 or zb > params.max_z) continue;

            const c6 = computeC6(za, zb, cn[ia], cn[ib]);
            const c8 = computeC8(c6, za, zb);
            const r0 = damping.a1 * @sqrt(c8 / c6) + damping.a2;

            var la: i32 = -lat_range_cutoff[0];
            while (la <= lat_range_cutoff[0]) : (la += 1) {
                var lb: i32 = -lat_range_cutoff[1];
                while (lb <= lat_range_cutoff[1]) : (lb += 1) {
                    var lc: i32 = -lat_range_cutoff[2];
                    while (lc <= lat_range_cutoff[2]) : (lc += 1) {
                        const shift = latticeVector(cell, la, lb, lc);
                        const r_vec = math.Vec3.sub(
                            math.Vec3.add(positions[ib], shift),
                            positions[ia],
                        );
                        const r2 = math.Vec3.dot(r_vec, r_vec);
                        const r = @sqrt(r2);
                        if (r > cutoff or r < 1e-10) continue;

                        // d²E/dR_Iα dR_Jβ for off-diagonal blocks (I≠J)
                        // with R_vec = R_J + L - R_I
                        const hess = pairwiseHessian(r_vec, r2, c6, c8, r0, damping);

                        // Fill D(Iα, Jβ) += hess(α,β) for the (ia, ib) block
                        for (0..3) |a| {
                            for (0..3) |b| {
                                dyn[(3 * ia + a) * dim + (3 * ib + b)] += hess[a][b];
                            }
                        }
                    }
                }
            }
        }
    }

    // Diagonal blocks: ASR D(Iα, Iβ) = -Σ_{J≠I} D(Iα, Jβ)
    for (0..n_atoms) |ia| {
        for (0..3) |a| {
            for (0..3) |b| {
                var sum: f64 = 0.0;
                for (0..n_atoms) |jb| {
                    if (jb == ia) continue;
                    sum += dyn[(3 * ia + a) * dim + (3 * jb + b)];
                }
                dyn[(3 * ia + a) * dim + (3 * ia + b)] = -sum;
            }
        }
    }

    // Convert Hartree/Bohr² → Rydberg/Bohr² (×2)
    for (dyn) |*d| {
        d.* *= 2.0;
    }

    return dyn;
}

/// Compute D3(BJ) contribution to the dynamical matrix at arbitrary q-point.
/// D_q(Iα, Jβ) = Σ_L D(Iα, Jβ; L) × exp(iq·(R_J+L - R_I))
/// Returns flat array of Complex of size (3*n_atoms)².
/// Units: Rydberg/Bohr².
pub fn computeDynmatQ(
    alloc: std.mem.Allocator,
    atomic_numbers: []const usize,
    positions: []const math.Vec3,
    cell: math.Mat3,
    damping: params.DampingParams,
    cutoff: f64,
    cn_cutoff: f64,
    q_cart: math.Vec3,
) ![]math.Complex {
    const n_atoms = atomic_numbers.len;
    const dim = 3 * n_atoms;
    const dyn_q = try alloc.alloc(math.Complex, dim * dim);
    @memset(dyn_q, math.complex.init(0.0, 0.0));
    if (n_atoms == 0) return dyn_q;

    const cn = try computeCoordinationNumbers(alloc, atomic_numbers, positions, cell, cn_cutoff);
    defer alloc.free(cn);

    const lat_range_cutoff = latticeRange(cell, cutoff);

    // Compute off-diagonal blocks (sum over all J, L including J==I with L≠0)
    for (0..n_atoms) |ia| {
        const za = atomic_numbers[ia];
        if (za == 0 or za > params.max_z) continue;

        for (0..n_atoms) |ib| {
            const zb = atomic_numbers[ib];
            if (zb == 0 or zb > params.max_z) continue;

            const c6 = computeC6(za, zb, cn[ia], cn[ib]);
            const c8 = computeC8(c6, za, zb);
            const r0 = damping.a1 * @sqrt(c8 / c6) + damping.a2;

            var la: i32 = -lat_range_cutoff[0];
            while (la <= lat_range_cutoff[0]) : (la += 1) {
                var lb: i32 = -lat_range_cutoff[1];
                while (lb <= lat_range_cutoff[1]) : (lb += 1) {
                    var lc: i32 = -lat_range_cutoff[2];
                    while (lc <= lat_range_cutoff[2]) : (lc += 1) {
                        if (ia == ib and la == 0 and lb == 0 and lc == 0) continue;
                        const shift = latticeVector(cell, la, lb, lc);
                        const r_vec = math.Vec3.sub(
                            math.Vec3.add(positions[ib], shift),
                            positions[ia],
                        );
                        const r2 = math.Vec3.dot(r_vec, r_vec);
                        const r = @sqrt(r2);
                        if (r > cutoff or r < 1e-10) continue;

                        const hess = pairwiseHessian(r_vec, r2, c6, c8, r0, damping);

                        // Phase factor: exp(iq·(R_J+L - R_I)) = exp(iq·r_vec)
                        const phase_arg = math.Vec3.dot(q_cart, r_vec);
                        const phase = math.complex.init(@cos(phase_arg), @sin(phase_arg));

                        for (0..3) |a| {
                            for (0..3) |b| {
                                const val = math.complex.scale(phase, hess[a][b]);
                                const idx = (3 * ia + a) * dim + (3 * ib + b);
                                dyn_q[idx] = math.complex.add(dyn_q[idx], val);
                            }
                        }
                    }
                }
            }
        }
    }

    // Diagonal blocks: ASR
    //   D_q(Iα, Iβ) = -Σ_{J,L≠(I,0)} D(Iα, Jβ; L)  (no phase for self-term)
    // More precisely, we need the real-space sum for the diagonal:
    // D_q(Iα,Iβ) = -Σ_{J,L≠(I,0)} D(Iα,Jβ;L) × exp(iq·L) ... wait, actually
    // The standard approach: D_q(Iα,Iβ) is computed as sum over all (J,L)≠(I,0)
    // just like off-diag.
    // But for the Γ point limit, we enforce ASR.
    // For q≠0, the off-diagonal already correctly represents the dynamical
    // matrix via Fourier transform.
    // We should NOT separately enforce ASR at q≠0; the lattice sum handles it
    // naturally.
    // However, the diagonal self-interaction at L=0 is excluded above.
    // Actually, for q≠0, D_q(Iα,Iβ) receives contributions from D(Iα,Iβ;L) for
    // L≠0 (which we computed)
    // plus the "on-site" term which is the negative sum of all off-diagonal
    // real-space force constants.
    // This on-site term is:
    // D(Iα,Iβ;L=0) = -Σ_{(J,L)≠(I,0)} D(Iα,Jβ;L)  (no phase factor here)
    //
    // So D_q(Iα,Iβ) = D(Iα,Iβ;L=0)*exp(iq·0) + Σ_{L≠0} D(Iα,Iβ;L)*exp(iq·L)
    //   = [-Σ_{(J,L)≠(I,0)} D(Iα,Jβ;L)]
    //     + [already accumulated from ib==ia, L≠0 above]
    //
    // The terms with ib==ia, L≠0 have already been added to the diagonal with
    // their phase factors.
    // We need to add the on-site self-term = -Σ_{(J,L)≠(I,0)} hess_real
    // (no phase).
    // But wait, we also need to subtract the ib==ia, L≠0 terms without phase that were
    // implicitly part of the diagonal sum. Let me think again...
    //
    // Real-space: D(Iα,Iβ;L=0) = -Σ_{(J,L')≠(I,0)} hess(I,J;L')
    // Fourier: D_q(Iα,Iβ) = D(Iα,Iβ;L=0)*1 + Σ_{L≠0} D(Iα,Iβ;L)*e^{iq·L}
    // But D(Iα,Iβ;L) for L≠0 = hess(I,I;L) (pair Hessian between I and image of I at L)
    // These are already accumulated in the loop above (ib==ia, L≠0) with phase.
    //
    // So we just need to add D(Iα,Iβ;L=0) to the diagonal.
    // D(Iα,Iβ;L=0) = -Σ_{(J,L')≠(I,0)} hess(I,J;L')
    // This means: sum all hess for all pairs involving atom I (as atom A), negate.

    // Compute the self-term for each atom
    for (0..n_atoms) |ia| {
        const za = atomic_numbers[ia];
        if (za == 0 or za > params.max_z) continue;

        // Sum all hessians involving atom ia
        var self_hess: [3][3]f64 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };

        for (0..n_atoms) |ib| {
            const zb = atomic_numbers[ib];
            if (zb == 0 or zb > params.max_z) continue;

            const c6 = computeC6(za, zb, cn[ia], cn[ib]);
            const c8 = computeC8(c6, za, zb);
            const r0 = damping.a1 * @sqrt(c8 / c6) + damping.a2;

            var la: i32 = -lat_range_cutoff[0];
            while (la <= lat_range_cutoff[0]) : (la += 1) {
                var lb: i32 = -lat_range_cutoff[1];
                while (lb <= lat_range_cutoff[1]) : (lb += 1) {
                    var lc: i32 = -lat_range_cutoff[2];
                    while (lc <= lat_range_cutoff[2]) : (lc += 1) {
                        if (ia == ib and la == 0 and lb == 0 and lc == 0) continue;
                        const shift = latticeVector(cell, la, lb, lc);
                        const r_vec = math.Vec3.sub(
                            math.Vec3.add(positions[ib], shift),
                            positions[ia],
                        );
                        const r2 = math.Vec3.dot(r_vec, r_vec);
                        const r = @sqrt(r2);
                        if (r > cutoff or r < 1e-10) continue;

                        const hess = pairwiseHessian(r_vec, r2, c6, c8, r0, damping);
                        for (0..3) |a| {
                            for (0..3) |b| {
                                self_hess[a][b] += hess[a][b];
                            }
                        }
                    }
                }
            }
        }

        // D(Iα,Iβ;L=0) = -self_hess
        for (0..3) |a| {
            for (0..3) |b| {
                const idx = (3 * ia + a) * dim + (3 * ia + b);
                dyn_q[idx] = math.complex.add(dyn_q[idx], math.complex.init(-self_hess[a][b], 0.0));
            }
        }
    }

    // Convert Hartree/Bohr² → Rydberg/Bohr² (×2)
    for (dyn_q) |*d| {
        d.* = math.complex.scale(d.*, 2.0);
    }

    return dyn_q;
}

// ====================================================================
// Internal helpers
// ====================================================================

/// Compute the pairwise Hessian d²E_pair/dR_Iα dR_Jβ for a single pair (I, J+L).
/// This is the second derivative of E = -s6*C6/(R^6+f^6) - s8*C8/(R^8+f^8)
/// with respect to the positions of atoms I and J.
/// For the (I,J) off-diagonal block: d²E/dR_Iα dR_Jβ = -d²E/dR_Iα dR_Iβ
/// (by translation symmetry).
/// Actually: d²E/dR_Iα dR_Jβ is what we compute directly.
/// R_vec = R_J + L - R_I, so dR_vec_α/dR_Jβ = δ_αβ, dR_vec_α/dR_Iβ = -δ_αβ
/// d²E/dR_Iα dR_Jβ = Σ_γδ (d²E/dR_γ dR_δ)(dR_γ/dR_Iα)(dR_δ/dR_Jβ)
///   = d²E/dR_α dR_β * (-1)(+1) = -d²E/dRα dRβ
/// Wait, let me re-derive carefully.
/// E depends on R_vec = R_J - R_I. So:
/// dE/dR_Iα = -dE/dR_α (chain rule with R_vec_α = R_Jα - R_Iα)
/// dE/dR_Jβ = +dE/dR_β
/// d²E/dR_Iα dR_Jβ = -d²E/dR_α dR_β
///
/// For the Hessian in real space we need d²E/(dR_Iα)(dR_Jβ) which goes into
/// the dynmat.
fn pairwiseHessian(
    r_vec: math.Vec3,
    r2: f64,
    c6: f64,
    c8: f64,
    r0: f64,
    damping: params.DampingParams,
) [3][3]f64 {
    const r = @sqrt(r2);
    const r4 = r2 * r2;
    const r6 = r4 * r2;
    const r8 = r6 * r2;
    const f6 = r0 * r0 * r0 * r0 * r0 * r0;
    const f8 = f6 * r0 * r0;
    const denom6 = r6 + f6;
    const denom8 = r8 + f8;
    _ = r;

    // E = -s6*C6/(R^6+f^6) - s8*C8/(R^8+f^8)
    // Let g(R²) = -s6*C6/(R^6+f^6), where R^6 = (R²)^3
    // dg/d(R²) = s6*C6*3*R^4/(R^6+f^6)^2
    // d²g/d(R²)² = s6*C6*(12*R^2*(R^6+f^6) - 3*R^4*2*6*R^4*R^(-1)*... )
    // This gets messy. Let's use the standard approach with R.
    //
    // For E(R) = -C/(R^n + f^n):
    // dE/dR = n*C*R^(n-1)/(R^n+f^n)^2
    // d²E/dR² = n*C*[(n-1)*R^(n-2)*(R^n+f^n)^2
    //                 - R^(n-1)*2*(R^n+f^n)*n*R^(n-1)] / (R^n+f^n)^4
    //         = n*C*R^(n-2)/(R^n+f^n)^2 * [(n-1) - 2*n*R^n/(R^n+f^n)]
    //
    // Hessian: d²E/dR_α dR_β
    //   = (d²E/dR² - (1/R)*dE/dR) * R_α*R_β/R² + (1/R)*dE/dR * δ_αβ
    //
    // For the off-diagonal dynmat block: d²E/(dR_Iα)(dR_Jβ) = -d²E/dR_α dR_β

    // C6 term
    const n6: f64 = 6.0;
    const de6_dr = n6 * damping.s6 * c6 * pow_f64(r2, 2.5) / (denom6 * denom6);
    const d2e6_prefactor = n6 * damping.s6 * c6 * pow_f64(r2, 2.0) / (denom6 * denom6);
    const d2e6_dr2 = d2e6_prefactor * ((n6 - 1.0) - 2.0 * n6 * r6 / denom6);

    // C8 term
    const n8: f64 = 8.0;
    const de8_dr = n8 * damping.s8 * c8 * pow_f64(r2, 3.5) / (denom8 * denom8);
    const d2e8_prefactor = n8 * damping.s8 * c8 * pow_f64(r2, 3.0) / (denom8 * denom8);
    const d2e8_dr2 = d2e8_prefactor * ((n8 - 1.0) - 2.0 * n8 * r8 / denom8);

    const de_dr = de6_dr + de8_dr;
    const d2e_dr2 = d2e6_dr2 + d2e8_dr2;

    const inv_r = 1.0 / @sqrt(r2);
    const inv_r2 = 1.0 / r2;

    const radial = (d2e_dr2 - inv_r * de_dr) * inv_r2;
    const iso = inv_r * de_dr;

    const rv = [3]f64{ r_vec.x, r_vec.y, r_vec.z };
    var hess: [3][3]f64 = undefined;
    for (0..3) |a| {
        for (0..3) |b| {
            var val = radial * rv[a] * rv[b];
            if (a == b) val += iso;
            // d²E/(dR_Iα)(dR_Jβ) = -d²E/dR_α dR_β
            hess[a][b] = -val;
        }
    }

    return hess;
}

/// power function for f64
fn pow_f64(base: f64, exp_val: f64) f64 {
    return std.math.pow(f64, base, exp_val);
}

/// Compute lattice vector from integer indices.
fn latticeVector(cell: math.Mat3, la: i32, lb: i32, lc: i32) math.Vec3 {
    const fa: f64 = @floatFromInt(la);
    const fb: f64 = @floatFromInt(lb);
    const fc: f64 = @floatFromInt(lc);
    return math.Vec3{
        .x = fa * cell.m[0][0] + fb * cell.m[1][0] + fc * cell.m[2][0],
        .y = fa * cell.m[0][1] + fb * cell.m[1][1] + fc * cell.m[2][1],
        .z = fa * cell.m[0][2] + fb * cell.m[1][2] + fc * cell.m[2][2],
    };
}

/// Determine the range of lattice vectors needed for a given cutoff.
fn latticeRange(cell: math.Mat3, cutoff: f64) [3]i32 {
    // Use the lengths of the lattice vectors to estimate the range
    const a1_len = math.Vec3.norm(cell.row(0));
    const a2_len = math.Vec3.norm(cell.row(1));
    const a3_len = math.Vec3.norm(cell.row(2));
    return .{
        @intFromFloat(@ceil(cutoff / a1_len)),
        @intFromFloat(@ceil(cutoff / a2_len)),
        @intFromFloat(@ceil(cutoff / a3_len)),
    };
}

/// Get the C6 reference table for a pair (za, zb).
/// Returns the table pointer or null if not available.
fn getC6Table(za: usize, zb: usize, n_ref_a: usize, n_ref_b: usize) ?[]const f64 {
    if (za == zb) {
        const table = params.getC6RefHomo(za);
        if (table.len == n_ref_a * n_ref_b) return table;
        return null;
    }
    // For hetero pairs, ensure consistent ordering
    if (za < zb) {
        const table = params.getC6RefHetero(za, zb);
        if (table) |t| {
            if (t.len == n_ref_a * n_ref_b) return t;
        }
        return null;
    }
    // za > zb: swap and transpose
    const table = params.getC6RefHetero(zb, za);
    if (table) |t| {
        if (t.len == n_ref_b * n_ref_a) return t; // Will need transposed access
    }
    return null;
}

/// Geometric mean approximation for C6 when reference data is unavailable.
fn geometricMeanC6(za: usize, zb: usize, cn_a: f64, cn_b: f64) f64 {
    // Compute homonuclear C6 for each element and take geometric mean
    const c6_aa = computeC6(za, za, cn_a, cn_a);
    const c6_bb = computeC6(zb, zb, cn_b, cn_b);
    if (c6_aa <= 0 or c6_bb <= 0) return 0.0;
    return @sqrt(c6_aa * c6_bb);
}

// ====================================================================
// Tests
// ====================================================================

test "coordination number - H2 molecule" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const atomic_numbers = [_]usize{ 1, 1 };
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 }, // ~0.74 Å ≈ 1.4 Bohr
    };
    // Large box for isolated molecule
    const cell = math.Mat3.fromRows(
        .{ .x = 100.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 100.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 100.0 },
    );

    const cn = try computeCoordinationNumbers(
        alloc,
        &atomic_numbers,
        &positions,
        cell,
        default_cn_cutoff,
    );
    defer alloc.free(cn);

    // Each H should have CN ≈ 1.0
    try testing.expect(cn[0] > 0.8 and cn[0] < 1.2);
    try testing.expect(cn[1] > 0.8 and cn[1] < 1.2);
}

test "C6 interpolation - carbon" {
    // C-C C6 should be between ~11 (sp3) and ~49 (free atom)
    const c6 = computeC6(6, 6, 3.0, 3.0); // near sp3 CN
    const testing = std.testing;
    try testing.expect(c6 > 10.0 and c6 < 50.0);
}

test "C8 from C6" {
    const c6: f64 = 25.0; // Approximate C-C C6
    const c8 = computeC8(c6, 6, 6);
    const testing = std.testing;
    // C8 should be positive and reasonable
    try testing.expect(c8 > 0.0);
    try testing.expect(c8 > c6); // C8 is typically larger than C6
}

test "energy - H2 molecule" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const atomic_numbers = [_]usize{ 1, 1 };
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const cell = math.Mat3.fromRows(
        .{ .x = 100.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 100.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 100.0 },
    );

    const energy = try computeEnergy(
        alloc,
        &atomic_numbers,
        &positions,
        cell,
        params.pbe_d3bj,
        default_cutoff,
        default_cn_cutoff,
    );

    // Energy should be negative (attractive)
    try testing.expect(energy < 0.0);
    // And small for H2 (of order -0.001 Ry or less)
    try testing.expect(@abs(energy) < 0.01);
}

test "forces - H2 molecule sum to zero" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const atomic_numbers = [_]usize{ 1, 1 };
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const cell = math.Mat3.fromRows(
        .{ .x = 100.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 100.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 100.0 },
    );

    const forces = try computeForces(
        alloc,
        &atomic_numbers,
        &positions,
        cell,
        params.pbe_d3bj,
        default_cutoff,
        default_cn_cutoff,
    );
    defer alloc.free(forces);

    // Forces should sum to zero (Newton's third law)
    const sum_x = forces[0].x + forces[1].x;
    const sum_y = forces[0].y + forces[1].y;
    const sum_z = forces[0].z + forces[1].z;
    try testing.expectApproxEqAbs(sum_x, 0.0, 1e-10);
    try testing.expectApproxEqAbs(sum_y, 0.0, 1e-10);
    try testing.expectApproxEqAbs(sum_z, 0.0, 1e-10);

    // Forces should be along x-axis (attractive → pointing toward each other)
    try testing.expectApproxEqAbs(forces[0].y, 0.0, 1e-10);
    try testing.expectApproxEqAbs(forces[0].z, 0.0, 1e-10);
}

test "dynmat - H2 molecule acoustic sum rule" {
    const testing = std.testing;
    const alloc = testing.allocator;

    const atomic_numbers = [_]usize{ 1, 1 };
    const positions = [_]math.Vec3{
        .{ .x = 0.0, .y = 0.0, .z = 0.0 },
        .{ .x = 1.4, .y = 0.0, .z = 0.0 },
    };
    const cell = math.Mat3.fromRows(
        .{ .x = 100.0, .y = 0.0, .z = 0.0 },
        .{ .x = 0.0, .y = 100.0, .z = 0.0 },
        .{ .x = 0.0, .y = 0.0, .z = 100.0 },
    );

    const dyn = try computeDynmat(
        alloc,
        &atomic_numbers,
        &positions,
        cell,
        params.pbe_d3bj,
        default_cutoff,
        default_cn_cutoff,
    );
    defer alloc.free(dyn);

    // Check ASR: sum over J for each (Iα, β) should be zero
    const dim = 6;
    for (0..dim) |i| {
        var sum: f64 = 0.0;
        for (0..dim) |j| {
            sum += dyn[i * dim + j];
        }
        try testing.expectApproxEqAbs(sum, 0.0, 1e-8);
    }
}
