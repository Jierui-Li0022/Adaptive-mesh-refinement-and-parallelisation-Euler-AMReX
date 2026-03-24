import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys, glob
from plot_utils import find_final_plotfile

# ── Level colours (level 0 → level 3) ───────────────────────────────────────
LEVEL_COLORS = ['#e8f4f8', '#fde8e8', '#e8f5e9', '#f3e5f5']
LEVEL_EDGE    = ['#2196f3', '#e53935', '#43a047', '#8e24aa']

TESTS = {
    1: {"name": "Sod Shock Tube",       "prob_hi": 1.5, "t": 0.25,
        "dirs": {"x":    "test1_sod_x",
                 "y":    "test1_sod_y",
                 "diag": "test1_sod_diag"}},
    2: {"name": "123 Problem",          "prob_hi": 1.0, "t": 0.15,
        "dirs": {"x":    "test2_123_x",
                 "y":    "test2_123_y",
                 "diag": "test2_123_diag"}},
    3: {"name": "Left Blast Wave",      "prob_hi": 1.0, "t": 0.012,
        "dirs": {"x":    "test3_blast_left_x",
                 "y":    "test3_blast_left_y",
                 "diag": "test3_blast_left_diag"}},
    4: {"name": "Right Blast Wave",     "prob_hi": 1.0, "t": 0.035,
        "dirs": {"x":    "test4_blast_right_x",
                 "y":    "test4_blast_right_y",
                 "diag": "test4_blast_right_diag"}},
    5: {"name": "Two-Shock Collision",  "prob_hi": 1.0, "t": 0.035,
        "dirs": {"x":    "test5_collision_x",
                 "y":    "test5_collision_y",
                 "diag": "test5_collision_diag"}},
}

# Output dirs per orientation (AMR runs)
AMR_BASE = {
    "x":    "output_2d/x_direction_amr",
    "y":    "output_2d/y_direction_amr",
    "diag": "output_2d/diagonal_amr",
}

def read_amrex_header(plotfile_dir):
    with open(os.path.join(plotfile_dir, "Header"), 'r') as f:
        raw = f.read()
    lines = [l.strip() for l in raw.split('\n') if l.strip() != '']
    idx = 0; idx += 1
    nvars = int(lines[idx]); idx += 1
    for _ in range(nvars): idx += 1
    idx += 1                                    # ndim
    time      = float(lines[idx]); idx += 1
    max_level = int(lines[idx]);   idx += 1
    prob_lo   = [float(x) for x in lines[idx].split()]; idx += 1
    prob_hi   = [float(x) for x in lines[idx].split()]; idx += 1
    return prob_lo, prob_hi, max_level, time

def read_level_boxes(plotfile_dir, level):
    """Return list of (lo_i, lo_j, hi_i, hi_j) index boxes for a level."""
    level_dir = os.path.join(plotfile_dir, f"Level_{level}")
    if not os.path.exists(level_dir):
        return []
    with open(os.path.join(level_dir, "Cell_H"), 'r') as f:
        lines = f.readlines()
    ci = 0; ci += 1; ci += 1; ci += 1; ci += 1   # skip header lines
    header = lines[ci].strip(); ci += 1
    ngrids = int(header.replace('(', '').replace(')', '').split()[0])
    boxes = []
    for _ in range(ngrids):
        s = lines[ci].strip().replace('(', ' ').replace(')', ' ').replace(',', ' ')
        ci += 1
        nums = [int(x) for x in s.split()]
        boxes.append((nums[0], nums[1], nums[2], nums[3]))
    return boxes

def boxes_to_physical(boxes, prob_lo, prob_hi, nx0, ny0, level):
    """Convert index boxes to physical (x, y) coordinates."""
    ref = 2 ** level
    dx  = (prob_hi[0] - prob_lo[0]) / (nx0 * ref)
    dy  = (prob_hi[1] - prob_lo[1]) / (ny0 * ref)
    phys = []
    for (lo_i, lo_j, hi_i, hi_j) in boxes:
        x0 = prob_lo[0] + lo_i * dx
        y0 = prob_lo[1] + lo_j * dy
        w  = (hi_i - lo_i + 1) * dx
        h  = (hi_j - lo_j + 1) * dy
        phys.append((x0, y0, w, h))
    return phys

def get_level0_ncells(plotfile_dir, prob_lo, prob_hi):
    """Infer nx0, ny0 from Level_0 Cell_H."""
    boxes = read_level_boxes(plotfile_dir, 0)
    if not boxes:
        return None, None
    nx0 = max(b[2] for b in boxes) + 1
    ny0 = max(b[3] for b in boxes) + 1
    return nx0, ny0

def plot_mesh_patches(plotfile_dir, ax):
    """Draw AMR mesh patches on ax. No title. Returns True if successful."""
    try:
        prob_lo, prob_hi, max_level, time = read_amrex_header(plotfile_dir)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                ha='center', va='center', fontsize=7, color='red')
        return False

    nx0, ny0 = get_level0_ncells(plotfile_dir, prob_lo, prob_hi)
    if nx0 is None:
        return False

    legend_handles = []

    for lev in range(max_level + 1):
        boxes = read_level_boxes(plotfile_dir, lev)
        if not boxes:
            continue
        phys = boxes_to_physical(boxes, prob_lo, prob_hi, nx0, ny0, lev)
        color = LEVEL_COLORS[min(lev, len(LEVEL_COLORS) - 1)]
        edge  = LEVEL_EDGE [min(lev, len(LEVEL_EDGE)   - 1)]
        lw    = max(0.4, 1.2 - lev * 0.2)

        for (x0, y0, w, h) in phys:
            rect = mpatches.Rectangle(
                (x0, y0), w, h,
                linewidth=lw, edgecolor=edge,
                facecolor=color, alpha=0.6, zorder=lev + 1
            )
            ax.add_patch(rect)

        patch = mpatches.Patch(facecolor=color, edgecolor=edge,
                               alpha=0.8, label=f"Level {lev}")
        legend_handles.append(patch)

    ax.set_xlim(prob_lo[0], prob_hi[0])
    ax.set_ylim(prob_lo[1], prob_hi[1])
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right')
    ax.grid(False)
    return True

def main():
    os.makedirs("plot/result/mesh_patches", exist_ok=True)

    for ori in ['x', 'y', 'diag']:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        for col, tid in enumerate([1, 2, 3, 4, 5]):
            T   = TESTS[tid]
            ax  = axes[col]
            dir_suffix = "_amr" if ori == "diag" else ""
            dir_name = T['dirs'][ori] + dir_suffix
            rd  = os.path.join(AMR_BASE[ori], dir_name)

            if os.path.exists(rd):
                try:
                    pf = find_final_plotfile(rd)
                    plot_mesh_patches(pf, ax)
                    print(f"  Test {tid} {ori}: {pf}")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                            ha='center', va='center', fontsize=7, color='red')
                    print(f"  Test {tid} {ori}: Error — {e}")
            else:
                ax.text(0.5, 0.5, "Not found", transform=ax.transAxes,
                        ha='center', va='center', fontsize=8)
                print(f"  Test {tid} {ori}: not found — {rd}")

        plt.tight_layout()
        out = f"plot/result/mesh_patches/mesh_{ori}_all5.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print("\nDone!")

if __name__ == '__main__':
    main()