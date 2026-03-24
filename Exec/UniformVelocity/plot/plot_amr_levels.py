import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys, glob
from plot_utils import find_final_plotfile, exact_riemann, GAMMA

def read_amrex_level_cells(plotfile_dir, level):
    header_path = os.path.join(plotfile_dir, "Header")
    with open(header_path, 'r') as f:
        raw = f.read()
    lines = [l.strip() for l in raw.split('\n') if l.strip() != '']
    idx = 0; idx += 1
    nvars = int(lines[idx]); idx += 1
    for _ in range(nvars): idx += 1
    idx += 1; idx += 1
    max_level = int(lines[idx]); idx += 1
    prob_lo = [float(x) for x in lines[idx].split()]; idx += 1
    prob_hi = [float(x) for x in lines[idx].split()]; idx += 1

    level_dir = os.path.join(plotfile_dir, f"Level_{level}")
    if not os.path.exists(level_dir): return None, None, None, None

    with open(os.path.join(level_dir, "Cell_H"), 'r') as f:
        cell_lines = f.readlines()
    ci = 0; ci += 1; ci += 1
    ncomp = int(cell_lines[ci].strip()); ci += 1; ci += 1
    header_line = cell_lines[ci].strip(); ci += 1
    ngrids = int(header_line.replace('(', '').replace(')', '').split()[0])
    if ngrids == 0: return None, None, None, None

    fab_boxes = []
    for g in range(ngrids):
        box_str = cell_lines[ci].strip(); ci += 1
        box_str = box_str.replace('(', ' ').replace(')', ' ').replace(',', ' ')
        nums = [int(x) for x in box_str.split()]
        fab_boxes.append((nums[0], nums[1], nums[2], nums[3]))
    ci += 1; ci += 1

    fab_files = []; fab_offsets = []
    for g in range(ngrids):
        parts = cell_lines[ci].strip().split(); ci += 1
        if parts[0] == 'FabOnDisk:':
            fab_files.append(parts[1]); fab_offsets.append(int(parts[2]))
        else:
            fab_files.append(parts[0]); fab_offsets.append(int(parts[1]))

    level0_dir = os.path.join(plotfile_dir, "Level_0")
    with open(os.path.join(level0_dir, "Cell_H"), 'r') as f:
        c0 = f.readlines()
    c0i = 0; c0i += 1; c0i += 1; c0i += 1; c0i += 1
    h0 = c0[c0i].strip(); c0i += 1
    ng0 = int(h0.replace('(', '').replace(')', '').split()[0])
    boxes0 = []
    for g in range(ng0):
        bs = c0[c0i].strip(); c0i += 1
        bs = bs.replace('(', ' ').replace(')', ' ').replace(',', ' ')
        boxes0.append([int(x) for x in bs.split()])
    nx_base = max(b[2] for b in boxes0) - min(b[0] for b in boxes0) + 1
    ref_ratio = 2
    dx = (prob_hi[0] - prob_lo[0]) / (nx_base * ref_ratio**level)

    cells = []
    for g in range(ngrids):
        lo_i, lo_j, hi_i, hi_j = fab_boxes[g]
        gx = hi_i - lo_i + 1; gy = hi_j - lo_j + 1
        fab_file = os.path.join(level_dir, fab_files[g])
        with open(fab_file, 'rb') as f:
            f.seek(fab_offsets[g])
            while f.read(1) != b'\n': pass
            raw_data = np.fromfile(f, dtype=np.float64, count=ncomp*gx*gy)
            raw_data = raw_data.reshape((ncomp, gy, gx))
        jmid = gy // 2
        for i_local in range(gx):
            i_glob = lo_i + i_local
            x_left = prob_lo[0] + i_glob * dx
            x_right = x_left + dx
            rho = raw_data[0, jmid, i_local]
            mx  = raw_data[1, jmid, i_local]
            my  = raw_data[2, jmid, i_local]
            E   = raw_data[3, jmid, i_local]
            cells.append((x_left, x_right, rho, mx, my, E))

    return cells, dx, max_level, prob_lo

def get_all_level_cells(plotfile_dir):
    header_path = os.path.join(plotfile_dir, "Header")
    with open(header_path, 'r') as f:
        raw = f.read()
    lines = [l.strip() for l in raw.split('\n') if l.strip() != '']
    idx = 0; idx += 1
    nvars = int(lines[idx]); idx += 1
    for _ in range(nvars): idx += 1
    idx += 1; idx += 1
    max_level = int(lines[idx]); idx += 1

    all_cells = []
    for lev in range(max_level + 1):
        result = read_amrex_level_cells(plotfile_dir, lev)
        if result[0] is None: continue
        cells, dx, ml, plo = result
        for (xl, xr, rho, mx, my, E) in cells:
            all_cells.append((lev, xl, xr, rho, mx, my, E))

    final_cells = []
    for lev in range(max_level + 1):
        lev_cells = [c for c in all_cells if c[0] == lev]
        for cell in lev_cells:
            xc = 0.5 * (cell[1] + cell[2])
            covered = False
            for flev in range(lev + 1, max_level + 1):
                for fc in all_cells:
                    if fc[0] == flev and fc[1] <= xc < fc[2]:
                        covered = True; break
                if covered: break
            if not covered:
                final_cells.append(cell)

    return final_cells, max_level

TESTS = {
    1: {"name": "Sod Shock Tube", "rhoL": 1.0, "uL": 0.0, "pL": 1.0,
        "rhoR": 0.125, "uR": 0.0, "pR": 0.1, "x0": 0.5, "t": 0.25,
        "amr_dir": "test1_sod_amr"},
    2: {"name": "123 Problem", "rhoL": 1.0, "uL": -2.0, "pL": 0.4,
        "rhoR": 1.0, "uR": 2.0, "pR": 0.4, "x0": 0.5, "t": 0.15,
        "amr_dir": "test2_123_amr"},
    3: {"name": "Left Blast Wave", "rhoL": 1.0, "uL": 0.0, "pL": 1000.0,
        "rhoR": 1.0, "uR": 0.0, "pR": 0.01, "x0": 0.5, "t": 0.012,
        "amr_dir": "test3_blast_left_amr"},
    4: {"name": "Right Blast Wave", "rhoL": 1.0, "uL": 0.0, "pL": 0.01,
        "rhoR": 1.0, "uR": 0.0, "pR": 100.0, "x0": 0.5, "t": 0.035,
        "amr_dir": "test4_blast_right_amr"},
    5: {"name": "Two-Shock Collision", "rhoL": 5.99924, "uL": 19.5975, "pL": 460.894,
        "rhoR": 5.99242, "uR": -6.19633, "pR": 46.0950, "x0": 0.4, "t": 0.035,
        "amr_dir": "test5_collision_amr"},
}

level_colors = ['#B0D4F1', '#FF6B6B', '#C62828']
level_labels = ['Level 0 (coarse)', 'Level 1 (refined)', 'Level 2 (finest)']

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    os.makedirs("result/result_1d_amr/amr_level", exist_ok=True)
    test_ids = [1,2,3,4,5] if sys.argv[1] == 'all' else [int(sys.argv[1])]
    for test_id in test_ids:
        test = TESTS[test_id]
        print(f"\n=== Test {test_id}: {test['name']} ===")

        amr_dir = os.path.join("..", "output_1d_amr", test['amr_dir'])
        if not os.path.exists(amr_dir):
            print(f"  Not found: {amr_dir}"); continue

        pf = find_final_plotfile(amr_dir)
        print(f"  Reading: {pf}")
        final_cells, max_level = get_all_level_cells(pf)

        # Compute primitives
        cell_data = []
        for c in final_cells:
            lev, xl, xr, rho, mx, my, E = c
            u_vel = mx / rho; v_vel = my / rho
            p = (GAMMA - 1.0) * (E - 0.5 * rho * (u_vel**2 + v_vel**2))
            e_int = p / (rho * (GAMMA - 1.0))
            cell_data.append((lev, xl, xr, rho, u_vel, p, e_int))

        # Exact
        x_exact = np.linspace(0.001, 0.999, 2000)
        rho_ex, u_ex, p_ex = exact_riemann(
        x_exact, test['t'], test['rhoL'], test['uL'], test['pL'],
            test['rhoR'], test['uR'], test['pR'], test['x0'], GAMMA)
        e_ex = p_ex / (rho_ex * (GAMMA - 1.0))
        exact_vars = [rho_ex, u_ex, p_ex, e_ex]
        var_labels = ['Density', 'Velocity', 'Pressure', 'Internal Energy']

        # --- 2x2 plot with bar rectangles ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for vi, ax in enumerate(axes.flat):
            # Exact solution on top
            ax.plot(x_exact, exact_vars[vi], 'k-', linewidth=1.5, zorder=5)

            # Draw each cell as a bar
            for cd in cell_data:
                lev, xl, xr = cd[0], cd[1], cd[2]
                val = cd[3 + vi]
                width = xr - xl
                if val >= 0:
                    ax.bar(xl, val, width=width, align='edge', bottom=0,
                           color=level_colors[lev], edgecolor='grey',
                           linewidth=0.2, alpha=0.6, zorder=2 + lev)
                else:
                    ax.bar(xl, val, width=width, align='edge', bottom=0,
                           color=level_colors[lev], edgecolor='grey',
                           linewidth=0.2, alpha=0.6, zorder=2 + lev)

            ax.set_xlabel('x')
            ax.set_ylabel(var_labels[vi])
            ax.set_title(var_labels[vi])
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.2)

        # Legend
        legend_elements = [plt.Line2D([], [], color='k', linewidth=1.5, label='Exact')]
        for lev in range(max_level + 1):
            legend_elements.append(mpatches.Patch(color=level_colors[lev], alpha=0.6,
                                                  label=level_labels[lev]))
        axes[0, 0].legend(handles=legend_elements, fontsize=8, loc='best')

        plt.tight_layout()
        outfile = f"result/result_1d_amr/amr_level/amr_levels_test{test_id}.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"  Saved: {outfile}")
        plt.close()

        total = len(final_cells)
        for lev in range(max_level + 1):
            count = sum(1 for c in final_cells if c[0] == lev)
            print(f"  Level {lev}: {count} cells ({100.0*count/total:.1f}%)")

    print("\nDone!")

if __name__ == '__main__':
    main()