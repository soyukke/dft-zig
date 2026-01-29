"""Render an extended XYZ trajectory to MP4 using matplotlib + ffmpeg.

Usage:
  python benchmarks/relax/render_trajectory.py \
    --input benchmarks/relax/run_20260204_122403/dft_zig/out/relax_trajectory.xyz \
    --output benchmarks/relax/run_20260204_122403/dft_zig/out/relax.mp4 \
    --fps 10

Options:
  --cell / --no-cell   Show/hide unit cell (default: show if Lattice exists)
  --wrap / --no-wrap   Wrap atoms into the unit cell (default: wrap)
  --replicate N        Replicate periodic images (0 = none, 1 = 3x3x3)
  --bonds              Draw bonds based on cutoff distance
  --bonds-all          Draw bonds for all replicated images
  --bond-cutoff A      Bond cutoff (Angstrom, default: 2.6)
  --bond-center-index  1-based atom index to center bonds
  --zoom Z             Zoom factor (>1 = closer)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def parse_lattice(comment: str):
    key = 'Lattice="'
    start = comment.find(key)
    if start < 0:
        return None
    start += len(key)
    end = comment.find('"', start)
    if end < 0:
        return None
    parts = comment[start:end].split()
    if len(parts) != 9:
        return None
    values = [float(v) for v in parts]
    return np.array(values, dtype=float).reshape((3, 3))


def read_extxyz(path: Path):
    lines = path.read_text().splitlines()
    frames = []
    elements = []
    lattices = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            break
        if i + 1 >= len(lines):
            break
        lattice = parse_lattice(lines[i + 1])
        frame_elems = []
        frame_pos = []
        for j in range(n_atoms):
            idx = i + 2 + j
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            if len(parts) < 4:
                continue
            frame_elems.append(parts[0])
            frame_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if frame_pos:
            frames.append(np.array(frame_pos, dtype=float))
            elements.append(frame_elems)
            lattices.append(lattice)
        i += 2 + n_atoms
    return frames, elements, lattices


def build_colors(elements):
    unique = sorted({el for frame in elements for el in frame})
    cmap = plt.get_cmap("tab10")
    color_map = {el: cmap(i % 10) for i, el in enumerate(unique)}
    colors = [color_map[el] for el in elements[0]] if elements else []
    return colors


def wrap_positions(pos, lattice):
    if lattice is None:
        return pos
    inv = np.linalg.inv(lattice)
    frac = pos @ inv
    frac = frac - np.floor(frac)
    return frac @ lattice


def replicate_positions(pos, lattice, nrep):
    n = len(pos)
    base_indices = np.arange(n, dtype=int)
    if lattice is None or nrep <= 0:
        return pos, np.ones(n, dtype=bool), base_indices
    offsets = []
    for i in range(-nrep, nrep + 1):
        for j in range(-nrep, nrep + 1):
            for k in range(-nrep, nrep + 1):
                offsets.append((i, j, k))
    out = []
    tags = []
    indices = []
    for idx, (i, j, k) in enumerate(offsets):
        shift = i * lattice[0] + j * lattice[1] + k * lattice[2]
        out.append(pos + shift)
        is_center = i == 0 and j == 0 and k == 0
        tags.extend([is_center] * len(pos))
        indices.extend(base_indices)
    return np.vstack(out), np.array(tags, dtype=bool), np.array(indices, dtype=int)


def bond_segments(pos, lattice, cutoff, center_mask=None, min_image=True):
    segments = []
    use_mask = center_mask is not None
    inv = None
    if lattice is not None and min_image:
        inv = np.linalg.inv(lattice)
    n = pos.shape[0]
    for i in range(n):
        if use_mask and not center_mask[i]:
            continue
        for j in range(i + 1, n):
            delta = pos[j] - pos[i]
            if inv is not None:
                frac = delta @ inv
                frac = frac - np.round(frac)
                delta = frac @ lattice
            dist = np.linalg.norm(delta)
            if dist <= cutoff:
                segments.append((pos[i], pos[i] + delta))
    return segments


def replicate_segments(segments, lattice, nrep):
    if lattice is None or nrep <= 0:
        return segments
    offsets = []
    for i in range(-nrep, nrep + 1):
        for j in range(-nrep, nrep + 1):
            for k in range(-nrep, nrep + 1):
                offsets.append(i * lattice[0] + j * lattice[1] + k * lattice[2])
    out = []
    for shift in offsets:
        for p0, p1 in segments:
            out.append((p0 + shift, p1 + shift))
    return out


def lattice_corners(lattice):
    a1 = lattice[0]
    a2 = lattice[1]
    a3 = lattice[2]
    origin = np.zeros(3)
    return np.array(
        [
            origin,
            a1,
            a2,
            a3,
            a1 + a2,
            a1 + a3,
            a2 + a3,
            a1 + a2 + a3,
        ],
        dtype=float,
    )


def cell_edges(corners):
    idx = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    edges = []
    for i, j in idx:
        edges.append((corners[i], corners[j]))
    return edges


def main():
    parser = argparse.ArgumentParser(description="Render relax_trajectory.xyz to MP4")
    parser.add_argument("--input", required=True, help="Path to extended XYZ trajectory")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--size", type=float, default=4.8, help="Figure size (inches)")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--cell", dest="cell", action="store_true", help="Draw unit cell")
    parser.add_argument("--no-cell", dest="cell", action="store_false", help="Hide unit cell")
    parser.set_defaults(cell=True)
    parser.add_argument("--cell-color", default="#666666")
    parser.add_argument("--cell-alpha", type=float, default=0.7)
    parser.add_argument("--cell-width", type=float, default=1.0)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--replicate-alpha", type=float, default=0.25)
    parser.add_argument("--bonds", dest="bonds", action="store_true")
    parser.add_argument("--no-bonds", dest="bonds", action="store_false")
    parser.set_defaults(bonds=False)
    parser.add_argument("--bonds-all", dest="bonds_all", action="store_true")
    parser.add_argument("--bonds-center", dest="bonds_all", action="store_false")
    parser.set_defaults(bonds_all=False)
    parser.add_argument("--bond-cutoff", type=float, default=2.6)
    parser.add_argument("--bond-center-index", type=int, default=None)
    parser.add_argument("--atom-size", type=float, default=90.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--wrap", dest="wrap", action="store_true", help="Wrap atoms into unit cell")
    parser.add_argument("--no-wrap", dest="wrap", action="store_false", help="Do not wrap atoms")
    parser.set_defaults(wrap=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    frames, elements, lattices = read_extxyz(input_path)
    if not frames:
        raise SystemExit("No frames found in trajectory")
    center_index = None
    if args.bond_center_index is not None:
        if args.bond_center_index <= 0:
            raise SystemExit("--bond-center-index must be >= 1")
        center_index = args.bond_center_index - 1
        if center_index >= frames[0].shape[0]:
            raise SystemExit("--bond-center-index exceeds atom count")

    if args.wrap:
        wrapped_frames = []
        for frame, lattice in zip(frames, lattices):
            wrapped_frames.append(wrap_positions(frame, lattice))
        frames = wrapped_frames

    all_pos = np.concatenate(frames, axis=0)
    lattice0 = lattices[0] if lattices else None
    if lattice0 is not None:
        corners0 = lattice_corners(lattice0)
        if args.replicate > 0:
            rep_corners, _, _ = replicate_positions(corners0, lattice0, args.replicate)
            all_pos = np.vstack([all_pos, rep_corners])
        elif args.cell:
            all_pos = np.vstack([all_pos, corners0])
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)
    center = (mins + maxs) * 0.5
    span = (maxs - mins).max()
    span = max(span, 1e-6)
    if args.zoom <= 0:
        raise SystemExit("--zoom must be > 0")
    half = 0.6 * span / args.zoom

    colors = build_colors(elements)

    fig = plt.figure(figsize=(args.size, args.size), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()

    init = frames[0]
    display_init, tags, base_indices = replicate_positions(init, lattice0, args.replicate)
    display_colors = []
    base_count = len(colors)
    for idx, is_center in enumerate(tags):
        c = colors[idx % base_count]
        if is_center:
            display_colors.append(c)
        else:
            rgba = list(c)
            rgba[3] = args.replicate_alpha
            display_colors.append(rgba)
    xs = display_init[:, 0]
    ys = display_init[:, 1]
    zs = display_init[:, 2]
    zs_any = cast(Any, zs)
    scat = ax.scatter(xs, ys, zs=zs_any, s=args.atom_size, c=display_colors, depthshade=True)
    cell_lines = []
    if args.cell and lattice0 is not None:
        corners = lattice_corners(lattice0)
        for p0, p1 in cell_edges(corners):
            line = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=args.cell_color,
                alpha=args.cell_alpha,
                linewidth=args.cell_width,
            )[0]
            cell_lines.append(line)
    title = ax.set_title("")
    bond_lines = []
    if args.bonds and lattice0 is not None:
        bond_mask = None
        if center_index is not None:
            bond_mask = base_indices == center_index
            if not args.bonds_all:
                bond_mask = bond_mask & tags
        elif not args.bonds_all:
            bond_mask = tags
        segments = bond_segments(display_init, lattice0, args.bond_cutoff, center_mask=bond_mask, min_image=False)
        for p0, p1 in segments:
            line = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="#333333",
                linewidth=1.0,
                alpha=0.9,
            )[0]
            bond_lines.append(line)

    def update(frame_idx):
        pos = frames[frame_idx]
        lattice = lattices[frame_idx] if frame_idx < len(lattices) else None
        display_pos, tags, base_indices = replicate_positions(pos, lattice, args.replicate)
        setattr(scat, "_offsets3d", (display_pos[:, 0], display_pos[:, 1], display_pos[:, 2]))
        if cell_lines and frame_idx < len(lattices):
            lattice = lattices[frame_idx]
            if lattice is not None:
                corners = lattice_corners(lattice)
                for line, (p0, p1) in zip(cell_lines, cell_edges(corners)):
                    line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                    line.set_3d_properties([p0[2], p1[2]])
        if args.bonds and lattice is not None:
            bond_mask = None
            if center_index is not None:
                bond_mask = base_indices == center_index
                if not args.bonds_all:
                    bond_mask = bond_mask & tags
            elif not args.bonds_all:
                bond_mask = tags
            segments = bond_segments(display_pos, lattice, args.bond_cutoff, center_mask=bond_mask, min_image=False)
            if len(segments) > len(bond_lines):
                for _ in range(len(segments) - len(bond_lines)):
                    line = ax.plot([], [], [], color="#333333", linewidth=1.0, alpha=0.9)[0]
                    bond_lines.append(line)
            for line, seg in zip(bond_lines, segments):
                p0, p1 = seg
                line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
                line.set_3d_properties([p0[2], p1[2]])
            for line in bond_lines[len(segments):]:
                line.set_data([], [])
                line.set_3d_properties([])
        title.set_text(f"frame {frame_idx + 1}/{len(frames)}")
        return scat, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / args.fps,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
    anim.save(str(output_path), writer=writer, dpi=args.dpi)


if __name__ == "__main__":
    main()
