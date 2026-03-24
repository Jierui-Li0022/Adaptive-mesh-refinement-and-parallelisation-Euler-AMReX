import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from plot_utils import find_final_plotfile, exact_riemann, compute_L1_error, GAMMA

def read_amrex_level(plotfile_dir, level):
    """Read a single level from an AMReX plotfile."""
    header_path = os.path.join(plotfile_dir, "Header")
    with open(header_path, 'r') as f:
        raw = f.read()
    lines = [l.strip() for l in raw.split('\n') if l.strip() != '']

    idx = 0
    version = lines[idx]; idx += 1
    nvars = int(lines[idx]); idx += 1
    var_names = []
    for _ in range(nvars):
        var_names.append(lines[idx]); idx += 1
    ndim = int(lines[idx]); idx += 1
    time = float(lines[idx]); idx += 1
    max_level = int(lines[idx]); idx += 1
    prob_lo = [float(x) for x in lines[idx].split()]; idx += 1
    prob_hi = [float(x) for x in lines[idx].split()]; idx += 1

    # Skip ref_ratio lines
    if max_level > 0:
        idx += 1  # ref_ratio line(s)

    # Skip domain boxes for each level
    idx += 1  # could be multiple domain box lines

    # Skip step numbers per level
    idx += 1

    # Skip dx for each level
    for lev in range(max_level + 1):
        idx += 1

    # Read level directory
    level_dir = os.path.join(plotfile_dir, f"Level_{level}")
    if not os.path.exists(level_dir):
        return None, None, None, None, None

    cell_header_path = os.path.join(level_dir, "Cell_H")
    with open(cell_header_path, 'r') as f:
        cell_lines = f.readlines()

    ci = 0
    ci += 1  # version
    ci += 1  # how
    ncomp = int(cell_lines[ci].strip()); ci += 1
    nghost = int(cell_lines[ci].strip()); ci += 1

    header_line = cell_lines[ci].strip()
    ci += 1
    ngrids_str = header_line.replace('(', '').replace(')', '').split()[0]
    ngrids = int(ngrids_str)

    if ngrids == 0:
        return None, None, None, None, None

    fab_boxes = []
    for g in range(ngrids):
        box_str = cell_lines[ci].strip()
        ci += 1
        box_str = box_str.replace('(', ' ').replace(')', ' ').replace(',', ' ')
        nums = [int(x) for x in box_str.split()]
        fab_boxes.append((nums[0], nums[1], nums[2], nums[3]))
    ci += 1  # closing ")"
    ci += 1  # nfabs

    fab_files = []
    fab_offsets = []
    for g in range(ngrids):
        parts = cell_lines[ci].strip().split()
        ci += 1
        if parts[0] == 'FabOnDisk:':
            fab_files.append(parts[1])
            fab_offsets.append(int(parts[2]))
        else:
            fab_files.append(parts[0])
            fab_offsets.append(int(parts[1]))

    all_lo_i = min(b[0] for b in fab_boxes)
    all_lo_j = min(b[1] for b in fab_boxes)
    all_hi_i = max(b[2] for b in fab_boxes)
    all_hi_j = max(b[3] for b in fab_boxes)

    # Compute dx for this level
    ref_ratio = 2  # assuming ref_ratio = 2
    nx_lev0 = None

    # We need to figure out the grid spacing at this level
    # Read from level 0 to get base dx, then refine
    level0_dir = os.path.join(plotfile_dir, "Level_0")
    cell_header0 = os.path.join(level0_dir, "Cell_H")
    with open(cell_header0, 'r') as f:
        c0_lines = f.readlines()
    c0i = 0; c0i += 1; c0i += 1
    ncomp0 = int(c0_lines[c0i].strip()); c0i += 1
    nghost0 = int(c0_lines[c0i].strip()); c0i += 1
    h0_line = c0_lines[c0i].strip(); c0i += 1
    ng0 = int(h0_line.replace('(', '').replace(')', '').split()[0])
    boxes0 = []
    for g in range(ng0):
        bs = c0_lines[c0i].strip(); c0i += 1
        bs = bs.replace('(', ' ').replace(')', ' ').replace(',', ' ')
        ns = [int(x) for x in bs.split()]
        boxes0.append(ns)
    base_lo_i = min(b[0] for b in boxes0)
    base_hi_i = max(b[2] for b in boxes0)
    nx_base = base_hi_i - base_lo_i + 1

    dx_base = (prob_hi[0] - prob_lo[0]) / nx_base
    dx_lev = dx_base / (ref_ratio ** level)

    # Read data for this level
    data_patches = []
    for g in range(ngrids):
        lo_i, lo_j, hi_i, hi_j = fab_boxes[g]
        gx = hi_i - lo_i + 1
        gy = hi_j - lo_j + 1

        fab_file = os.path.join(level_dir, fab_files[g])
        with open(fab_file, 'rb') as f:
            f.seek(fab_offsets[g])
            while f.read(1) != b'\n':
                pass
            nvals = ncomp * gx * gy
            raw_data = np.fromfile(f, dtype=np.float64, count=nvals)
            raw_data = raw_data.reshape((ncomp, gy, gx))

        data_patches.append({
        'lo_i': lo_i, 'lo_j': lo_j,
            'hi_i': hi_i, 'hi_j': hi_j,
            'data': raw_data
        })

    return data_patches, dx_lev, prob_lo, prob_hi, var_names

def read_amrex_composite(plotfile_dir):
    """Read AMReX plotfile and composite all levels (finest data wins)."""
    # First, read the header to find max_level
    header_path = os.path.join(plotfile_dir, "Header")
    with open(header_path, 'r') as f:
        raw = f.read()
    lines = [l.strip() for l in raw.split('\n') if l.strip() != '']

    idx = 0; idx += 1  # version
    nvars = int(lines[idx]); idx += 1
    for _ in range(nvars): idx += 1
    idx += 1  # ndim
    time = float(lines[idx]); idx += 1
    max_level = int(lines[idx]); idx += 1

    # Read level 0 to establish base grid
    patches0, dx0, prob_lo, prob_hi, var_names = read_amrex_level(plotfile_dir, 0)
    if patches0 is None:
        raise RuntimeError("Cannot read level 0")

    # Determine base grid size
    all_lo = min(p['lo_i'] for p in patches0)
    all_hi = max(p['hi_i'] for p in patches0)
    nx_base = all_hi - all_lo + 1

    # Use finest effective resolution for output
    ref_ratio = 2
    nx_fine = nx_base * (ref_ratio ** max_level)
    dx_fine = (prob_hi[0] - prob_lo[0]) / nx_fine
    ncomp = patches0[0]['data'].shape[0]

    # Fill from coarse to fine, but skip fine-level data in near-vacuum cells.
    # Fine-grid interpolation errors in e_int are amplified by 1/rho when
    # rho->0, so the coarse volume-averaged value is more reliable there.
    # Always use finest available data (clamp is applied in simulation)
    composite = np.full((nx_fine, ncomp), np.nan)

    for lev in range(max_level + 1):
        patches, dx_lev, _, _, _ = read_amrex_level(plotfile_dir, lev)
        if patches is None:
            continue

        ratio = ref_ratio ** (max_level - lev)

        for patch in patches:
            lo_i = patch['lo_i']
            hi_i = patch['hi_i']
            gx = hi_i - lo_i + 1
            data = patch['data']
            jmid = data.shape[1] // 2

            for i_local in range(gx):
                i_lev = lo_i + i_local
                i_fine_start = i_lev * ratio
                i_fine_end = i_fine_start + ratio
                for n in range(ncomp):
                    val = data[n, jmid, i_local]
                    composite[i_fine_start:i_fine_end, n] = val

    # Build x coordinates
    x = prob_lo[0] + (np.arange(nx_fine) + 0.5) * dx_fine

    return x, composite, var_names, time, max_level

TESTS = {
    1: {"name": "Sod Shock Tube", "rhoL": 1.0, "uL": 0.0, "pL": 1.0,
        "rhoR": 0.125, "uR": 0.0, "pR": 0.1, "x0": 0.5, "t": 0.25,
        "dir": "test1_sod", "amr_dir": "test1_sod_amr"},
    2: {"name": "123 Problem", "rhoL": 1.0, "uL": -2.0, "pL": 0.4,
        "rhoR": 1.0, "uR": 2.0, "pR": 0.4, "x0": 0.5, "t": 0.15,
        "dir": "test2_123", "amr_dir": "test2_123_amr"},
    3: {"name": "Left Blast Wave", "rhoL": 1.0, "uL": 0.0, "pL": 1000.0,
        "rhoR": 1.0, "uR": 0.0, "pR": 0.01, "x0": 0.5, "t": 0.012,
        "dir": "test3_blast_left", "amr_dir": "test3_blast_left_amr"},
    4: {"name": "Right Blast Wave", "rhoL": 1.0, "uL": 0.0, "pL": 0.01,
        "rhoR": 1.0, "uR": 0.0, "pR": 100.0, "x0": 0.5, "t": 0.035,
        "dir": "test4_blast_right", "amr_dir": "test4_blast_right_amr"},
    5: {"name": "Two-Shock Collision", "rhoL": 5.99924, "uL": 19.5975, "pL": 460.894,
        "rhoR": 5.99242, "uR": -6.19633, "pR": 46.0950, "x0": 0.4, "t": 0.035,
        "dir": "test5_collision", "amr_dir": "test5_collision_amr"},
}

UNIFORM_RESOLUTIONS = [100, 200, 400]

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    os.makedirs("result/result_1d_amr", exist_ok=True)

    if sys.argv[1] == 'all':
        test_ids = [1, 2, 3, 4, 5]
    else:
        test_ids = [int(sys.argv[1])]
    for test_id in test_ids:
        test = TESTS[test_id]
        print(f"\n=== Toro Test {test_id}: {test['name']} (AMR comparison) ===")

        x_exact = np.linspace(0.001, 0.999, 2000)
        rho_exact, u_exact, p_exact = exact_riemann(
        x_exact, test['t'], test['rhoL'], test['uL'], test['pL'],
            test['rhoR'], test['uR'], test['pR'], test['x0'], GAMMA)
        e_exact = p_exact / (rho_exact * (GAMMA - 1.0))

        # --- Read AMR data ---
        amr_dir = os.path.join("..", "output_1d_amr", test['amr_dir'])
        amr_data = None
        if os.path.exists(amr_dir):
            try:
                pf_amr = find_final_plotfile(amr_dir)
                print(f"  Reading AMR: {pf_amr}")
                x_amr, comp_amr, vn_amr, t_amr, ml = read_amrex_composite(pf_amr)
                rho_amr = comp_amr[:, 0]
                mx_amr = comp_amr[:, 1]
                my_amr = comp_amr[:, 2]
                E_amr = comp_amr[:, 3]
                u_amr = mx_amr / rho_amr
                p_amr = (GAMMA - 1.0) * (E_amr - 0.5 * rho_amr * (u_amr**2 + (my_amr/rho_amr)**2))
                e_int_amr = p_amr / (rho_amr * (GAMMA - 1.0))
                amr_data = (x_amr, rho_amr, u_amr, p_amr, e_int_amr)
                print(f"  AMR: {len(x_amr)} effective cells, max_level={ml}")
            except Exception as e:
                print(f"  AMR read failed: {e}")
        else:
            print(f"  AMR directory not found: {amr_dir}")

        # --- Read uniform grid data (N=100 for comparison) ---
        uniform_data = {}
        res_dir = os.path.join("..", "output_1d", test['dir'], "100")
        if os.path.exists(res_dir):
            try:
                pf = find_final_plotfile(res_dir)
                print(f"  Reading uniform N=100: {pf}")
                x, y, data, vn, t_uni = read_amrex_plotfile(pf)
                jmid = len(y) // 2
                rho = data[:, jmid, 0]; mx = data[:, jmid, 1]
                my = data[:, jmid, 2]; E = data[:, jmid, 3]
                u_vel = mx / rho; p = (GAMMA - 1.0) * (E - 0.5 * rho * (u_vel**2 + (my/rho)**2))
                e_int = p / (rho * (GAMMA - 1.0))
                uniform_data[100] = (x, rho, u_vel, p, e_int)
            except Exception as e:
                print(f"  Uniform N=100 read failed: {e}")

        # --- Plot: AMR vs Uniform vs Exact ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        var_labels = ['Density', 'Velocity', 'Pressure', 'Internal Energy']

        # Plot exact
        exact_vars = [rho_exact, u_exact, p_exact, e_exact]
        for vi, ax in enumerate(axes.flat):
            ax.plot(x_exact, exact_vars[vi], 'k-', linewidth=1.5, label='Exact')

        # Plot uniform N=100
        if 100 in uniform_data:
            x_u, rho_u, u_u, p_u, e_u = uniform_data[100]
            uni_vars = [rho_u, u_u, p_u, e_u]
            for vi, ax in enumerate(axes.flat):
                ax.plot(x_u, uni_vars[vi], 'bo', markersize=2.5, alpha=0.5, label='Uniform N=100')

        # Plot AMR
        if amr_data is not None:
            x_a, rho_a, u_a, p_a, e_a = amr_data
            amr_vars = [rho_a, u_a, p_a, e_a]
            for vi, ax in enumerate(axes.flat):
                ax.plot(x_a, amr_vars[vi], 'r.', markersize=1.5, alpha=0.6, label='AMR (base=100)')

        for vi, ax in enumerate(axes.flat):
            ax.set_xlabel('x')
            ax.set_ylabel(var_labels[vi])
            ax.set_title(var_labels[vi])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = f"result/result_1d_amr/1d_{test['dir']}_amr.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"  Saved: {outfile}")
        plt.close()

        # --- Print L1 errors comparison ---
        print(f"\n  L1 Error Comparison — Test {test_id}: {test['name']}")
        print(f"  {'Method':<25} {'Density':<14} {'Velocity':<14} {'Pressure':<14} {'Int Energy':<14}")
        print(f"  {'─'*70}")
        if 100 in uniform_data:
            x_u, rho_u, u_u, p_u, e_u = uniform_data[100]
            e1 = compute_L1_error(x_u, rho_u, x_exact, rho_exact)
            e2 = compute_L1_error(x_u, u_u, x_exact, u_exact)
            e3 = compute_L1_error(x_u, p_u, x_exact, p_exact)
            e4 = compute_L1_error(x_u, e_u, x_exact, e_exact)
            print(f"  {'Uniform N=100':<25} {e1:<14.6e} {e2:<14.6e} {e3:<14.6e} {e4:<14.6e}")

        if amr_data is not None:
            x_a, rho_a, u_a, p_a, e_a = amr_data
            e1 = compute_L1_error(x_a, rho_a, x_exact, rho_exact)
            e2 = compute_L1_error(x_a, u_a, x_exact, u_exact)
            e3 = compute_L1_error(x_a, p_a, x_exact, p_exact)
            e4 = compute_L1_error(x_a, e_a, x_exact, e_exact)
            print(f"  {'AMR (base=100, L=2)':<25} {e1:<14.6e} {e2:<14.6e} {e3:<14.6e} {e4:<14.6e}")

    # --- Save all L1 errors to CSV ---
    import csv
    csv_file = "result/result_1d_amr/L1_error_amr_comparison.csv"
    os.makedirs("result/result_1d_amr", exist_ok=True)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test', 'Method', 'Density', 'Velocity', 'Pressure', 'Internal Energy'])

    for test_id in ([1,2,3,4,5] if sys.argv[1] == 'all' else [int(sys.argv[1])]):
        test = TESTS[test_id]
        x_exact = np.linspace(0.001, 0.999, 2000)
        rho_exact, u_exact, p_exact = exact_riemann(
        x_exact, test['t'], test['rhoL'], test['uL'], test['pL'],
            test['rhoR'], test['uR'], test['pR'], test['x0'], GAMMA)
        e_exact = p_exact / (rho_exact * (GAMMA - 1.0))

        rows = []

        # Uniform N=100
        res_dir = os.path.join("..", "output_1d", test['dir'], "100")
        if os.path.exists(res_dir):
            try:
                pf = find_final_plotfile(res_dir)
                x, y, data, vn, t_uni = read_amrex_plotfile(pf)
                jmid = len(y) // 2
                rho = data[:, jmid, 0]; mx = data[:, jmid, 1]
                my = data[:, jmid, 2]; E = data[:, jmid, 3]
                u_vel = mx / rho; p = (GAMMA - 1.0) * (E - 0.5 * rho * (u_vel**2 + (my/rho)**2))
                e_int = p / (rho * (GAMMA - 1.0))
                rows.append([f"Test {test_id}", "Uniform N=100",
                    f"{compute_L1_error(x, rho, x_exact, rho_exact):.6e}",
                    f"{compute_L1_error(x, u_vel, x_exact, u_exact):.6e}",
                    f"{compute_L1_error(x, p, x_exact, p_exact):.6e}",
                    f"{compute_L1_error(x, e_int, x_exact, e_exact):.6e}"])
            except: pass

            amr_dir = os.path.join("..", "output_1d_amr", test['amr_dir'])
        if os.path.exists(amr_dir):
            try:
                pf_amr = find_final_plotfile(amr_dir)
                x_amr, comp_amr, vn_amr, t_amr, ml = read_amrex_composite(pf_amr)
                rho_amr = comp_amr[:, 0]; mx_amr = comp_amr[:, 1]
                my_amr = comp_amr[:, 2]; E_amr = comp_amr[:, 3]
                u_amr = mx_amr / rho_amr
                p_amr = (GAMMA - 1.0) * (E_amr - 0.5 * rho_amr * (u_amr**2 + (my_amr/rho_amr)**2))
                e_amr = p_amr / (rho_amr * (GAMMA - 1.0))
                rows.append([f"Test {test_id}", "AMR (base=100)",
                    f"{compute_L1_error(x_amr, rho_amr, x_exact, rho_exact):.6e}",
                    f"{compute_L1_error(x_amr, u_amr, x_exact, u_exact):.6e}",
                    f"{compute_L1_error(x_amr, p_amr, x_exact, p_exact):.6e}",
                    f"{compute_L1_error(x_amr, e_amr, x_exact, e_exact):.6e}"])
            except: pass

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    print(f"\nL1 errors saved to: {csv_file}")

    # --- L1 Error Comparison Bar Chart ---
    print("\nGenerating L1 error comparison chart...")
    all_test_ids = [1,2,3,4,5] if sys.argv[1] == 'all' else [int(sys.argv[1])]
    var_names_plot = ['Density', 'Velocity', 'Pressure', 'Internal Energy']

    # Collect errors
    uniform_errors = {tid: [] for tid in all_test_ids}
    amr_errors = {tid: [] for tid in all_test_ids}

    for test_id in all_test_ids:
        test = TESTS[test_id]
        x_exact = np.linspace(0.001, 0.999, 2000)
        rho_exact, u_exact, p_exact = exact_riemann(
        x_exact, test['t'], test['rhoL'], test['uL'], test['pL'],
            test['rhoR'], test['uR'], test['pR'], test['x0'], GAMMA)
        e_exact = p_exact / (rho_exact * (GAMMA - 1.0))

        # Uniform N=100
        res_dir = os.path.join("..", "output_1d", test['dir'], "100")
        if os.path.exists(res_dir):
            try:
                pf = find_final_plotfile(res_dir)
                x, y, data, vn, t_u = read_amrex_plotfile(pf)
                jmid = len(y) // 2
                rho = data[:, jmid, 0]; mx = data[:, jmid, 1]
                myy = data[:, jmid, 2]; E = data[:, jmid, 3]
                u_vel = mx / rho; p = (GAMMA-1.0)*(E - 0.5*rho*(u_vel**2+(myy/rho)**2))
                e_int = p / (rho*(GAMMA-1.0))
                uniform_errors[test_id] = [
                    compute_L1_error(x, rho, x_exact, rho_exact),
                    compute_L1_error(x, u_vel, x_exact, u_exact),
                    compute_L1_error(x, p, x_exact, p_exact),
                    compute_L1_error(x, e_int, x_exact, e_exact)]
            except: pass

            amr_dir = os.path.join("..", "output_1d_amr", test['amr_dir'])
        if os.path.exists(amr_dir):
            try:
                pf_amr = find_final_plotfile(amr_dir)
                x_a, comp_a, vn_a, t_a, ml = read_amrex_composite(pf_amr)
                rho_a = comp_a[:,0]; mx_a = comp_a[:,1]
                my_a = comp_a[:,2]; E_a = comp_a[:,3]
                u_a = mx_a/rho_a; p_a = (GAMMA-1.0)*(E_a - 0.5*rho_a*(u_a**2+(my_a/rho_a)**2))
                e_a = p_a / (rho_a*(GAMMA-1.0))
                amr_errors[test_id] = [
                    compute_L1_error(x_a, rho_a, x_exact, rho_exact),
                    compute_L1_error(x_a, u_a, x_exact, u_exact),
                    compute_L1_error(x_a, p_a, x_exact, p_exact),
                    compute_L1_error(x_a, e_a, x_exact, e_exact)]
            except: pass

    # Plot: grouped bar chart, one subplot per variable
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    test_labels = [f"Test {tid}" for tid in all_test_ids]
    x_pos = np.arange(len(all_test_ids))
    bar_w = 0.35

    for vi, ax in enumerate(axes.flat):
        uni_vals = []
        amr_vals = []
        for tid in all_test_ids:
            uni_vals.append(uniform_errors[tid][vi] if len(uniform_errors[tid]) > vi else 0)
            amr_vals.append(amr_errors[tid][vi] if len(amr_errors[tid]) > vi else 0)

        bars1 = ax.bar(x_pos - bar_w/2, uni_vals, bar_w, label='Uniform N=100',
                       color='#4A90D9', alpha=0.8)
        bars2 = ax.bar(x_pos + bar_w/2, amr_vals, bar_w, label='AMR (base=100)',
                       color='#E05252', alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(test_labels, fontsize=9)
        ax.set_ylabel('L1 Error')
        ax.set_title(var_names_plot[vi])
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2e}',
                        ha='center', va='bottom', fontsize=6, rotation=45)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2e}',
                        ha='center', va='bottom', fontsize=6, rotation=45)

    plt.tight_layout()
    chart_file = "result/result_1d_amr/L1_error_comparison.png"
    plt.savefig(chart_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {chart_file}")
    plt.close()

    print("\nDone!")

if __name__ == '__main__':
    main()