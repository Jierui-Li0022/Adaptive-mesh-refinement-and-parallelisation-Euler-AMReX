import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from plot_utils import find_final_plotfile, read_amrex_plotfile, exact_riemann, compute_L1_error, GAMMA

TESTS = {
    1: {"name": "Sod Shock Tube",
        "rhoL": 1.0, "uL": 0.0, "pL": 1.0,
        "rhoR": 0.125, "uR": 0.0, "pR": 0.1,
        "x0": 0.5, "t": 0.25,
        "dir": "test1_sod"},
    2: {"name": "123 Problem",
        "rhoL": 1.0, "uL": -2.0, "pL": 0.4,
        "rhoR": 1.0, "uR": 2.0, "pR": 0.4,
        "x0": 0.5, "t": 0.15,
        "dir": "test2_123"},
    3: {"name": "Left Blast Wave",
        "rhoL": 1.0, "uL": 0.0, "pL": 1000.0,
        "rhoR": 1.0, "uR": 0.0, "pR": 0.01,
        "x0": 0.5, "t": 0.012,
        "dir": "test3_blast_left"},
    4: {"name": "Right Blast Wave",
        "rhoL": 1.0, "uL": 0.0, "pL": 0.01,
        "rhoR": 1.0, "uR": 0.0, "pR": 100.0,
        "x0": 0.5, "t": 0.035,
        "dir": "test4_blast_right"},
    5: {"name": "Two-Shock Collision",
        "rhoL": 5.99924, "uL": 19.5975, "pL": 460.894,
        "rhoR": 5.99242, "uR": -6.19633, "pR": 46.0950,
        "x0": 0.4, "t": 0.035,
        "dir": "test5_collision"},
}

RESOLUTIONS = [100, 200, 400]
COLORS = {100: 'ro', 200: 'bs', 400: 'g^'}
LABELS = {100: 'N=100', 200: 'N=200', 400: 'N=400'}

def convergence_order(e1, e2, n1, n2):
    """Compute convergence order from two error-resolution pairs."""
    if e1 <= 0 or e2 <= 0:
        return float('nan')
    return np.log(e1 / e2) / np.log(n2 / n1)

def print_convergence_table(test_id, test_name, errors, resolutions):
    """Print a formatted convergence table for one test."""
    var_names = ['Density', 'Velocity', 'Pressure', 'Internal Energy']
    
    print(f"\n  {'─'*72}")
    print(f"  Convergence Table — Test {test_id}: {test_name}")
    print(f"  {'─'*72}")
    
    header = f"  {'Variable':<18}"
    for res in resolutions:
        header += f"{'L1(N='+str(res)+')':<14}"
    header += f"{'Order':<10}"
    print(header)
    print(f"  {'─'*72}")
    
    for vi, vname in enumerate(var_names):
        row = f"  {vname:<18}"
        for res in resolutions:
            if res in errors and vi < len(errors[res]):
                row += f"{errors[res][vi]:<14.6e}"
            else:
                row += f"{'N/A':<14}"
        
        # Single overall convergence order: N=100 to N=400
        res_list = [r for r in resolutions if r in errors]
        if len(res_list) >= 2:
            r1, r2 = res_list[0], res_list[-1]
            if r1 in errors and r2 in errors:
                order = convergence_order(errors[r1][vi], errors[r2][vi], r1, r2)
                row += f"{order:<10.2f}"
            else:
                row += f"{'N/A':<10}"
        print(row)
    
    print(f"  {'─'*72}")

def plot_convergence(all_errors, test_ids, resolutions):
    """Plot convergence curves for all tests on one figure."""
    var_names = ['Density', 'Velocity', 'Pressure', 'Internal Energy']
    test_markers = {1: 'o-', 2: 's-', 3: '^-', 4: 'D-', 5: 'v-'}
    test_colors = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    for vi, vname in enumerate(var_names):
        ax = axes.flat[vi]
        
        for tid in test_ids:
            if tid not in all_errors:
                continue
            res_list = sorted(all_errors[tid].keys())
            errs = [all_errors[tid][r][vi] for r in res_list]
            if all(e > 0 for e in errs):
                ax.loglog(res_list, errs, test_markers.get(tid, 'o-'),
                         color=test_colors[tid], label=f'Test {tid}', 
                         markersize=6, linewidth=1.5)
        
            n_ref = np.array([resolutions[0], resolutions[-1]], dtype=float)
        first_tid = [t for t in test_ids if t in all_errors]
        if first_tid:
            e_anchor = all_errors[first_tid[0]][resolutions[0]][vi]
            if e_anchor > 0:
                ax.loglog(n_ref, e_anchor * (n_ref[0]/n_ref)**1.0, 
                         'k--', alpha=0.3, linewidth=1, label='1st order')
                ax.loglog(n_ref, e_anchor * (n_ref[0]/n_ref)**2.0,
                         'k:', alpha=0.3, linewidth=1, label='2nd order')
        
        ax.set_xlabel('N (cells)')
        ax.set_ylabel('L1 Error')
        ax.set_title(vname)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    outfile = "result/result_1d/convergence_analysis.png"
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"\n  Saved convergence plot: {outfile}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("  e.g. python3 plot_toro.py 1")
        print("  or   python3 plot_toro.py all")
        sys.exit(1)

    os.makedirs("result/result_1d", exist_ok=True)

    if sys.argv[1] == 'all':
        test_ids = [1, 2, 3, 4, 5]
    else:
        test_ids = [int(sys.argv[1])]

    all_errors = {}  # {test_id: {res: [e_rho, e_u, e_p, e_eint]}}

    for test_id in test_ids:
        test = TESTS[test_id]
        print(f"\n=== Toro Test {test_id}: {test['name']} ===")
        x_exact = np.linspace(0.001, 0.999, 2000)
        rho_exact, u_exact, p_exact = exact_riemann(
        x_exact, test['t'],
            test['rhoL'], test['uL'], test['pL'],
            test['rhoR'], test['uR'], test['pR'],
            test['x0'], GAMMA)
        e_exact = p_exact / (rho_exact * (GAMMA - 1.0))

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        errors = {}  # {res: [e_rho, e_u, e_p, e_eint]}

        for res in RESOLUTIONS:
            res_dir = os.path.join("..", "output_1d", test['dir'], str(res))
            if not os.path.exists(res_dir):
                print(f"  Skipping {res} cells - directory not found: {res_dir}")
                continue

            try:
                pf = find_final_plotfile(res_dir)
            except FileNotFoundError:
                print(f"  Skipping {res} cells - no plotfile found")
                continue

            print(f"  Reading {res} cells: {pf}")
            x, y, data, var_names, time = read_amrex_plotfile(pf)

            jmid = 0
            rho = data[:, jmid, 0]
            mx  = data[:, jmid, 1]
            my  = data[:, jmid, 2]
            E   = data[:, jmid, 3]

            u_vel = mx / rho
            v_vel = my / rho
            p = (GAMMA - 1.0) * (E - 0.5 * rho * (u_vel**2 + v_vel**2))
            e_int = p / (rho * (GAMMA - 1.0))

            e_rho  = compute_L1_error(x, rho,   x_exact, rho_exact)
            e_u    = compute_L1_error(x, u_vel, x_exact, u_exact)
            e_p    = compute_L1_error(x, p,     x_exact, p_exact)
            e_eint = compute_L1_error(x, e_int, x_exact, e_exact)
            errors[res] = [e_rho, e_u, e_p, e_eint]

            ms = max(1.0, 2.0 - res / 400.0)
            axes[0, 0].plot(x, rho,   COLORS[res], markersize=ms, label=LABELS[res])
            axes[0, 1].plot(x, u_vel, COLORS[res], markersize=ms, label=LABELS[res])
            axes[1, 0].plot(x, p,     COLORS[res], markersize=ms, label=LABELS[res])
            axes[1, 1].plot(x, e_int, COLORS[res], markersize=ms, label=LABELS[res])

            axes[0, 0].plot(x_exact, rho_exact, 'k-', linewidth=1.5, label='Exact')
        axes[0, 1].plot(x_exact, u_exact,   'k-', linewidth=1.5, label='Exact')
        axes[1, 0].plot(x_exact, p_exact,   'k-', linewidth=1.5, label='Exact')
        axes[1, 1].plot(x_exact, e_exact,   'k-', linewidth=1.5, label='Exact')

        labels = ['Density', 'Velocity', 'Pressure', 'Internal Energy']
        for idx, ax in enumerate(axes.flat):
            ax.set_xlabel('x')
            ax.set_ylabel(labels[idx])
            ax.set_title(labels[idx])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = f"result/result_1d/1d_{test['dir']}.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"  Saved: {outfile}")
        plt.close()

        if errors:
            print_convergence_table(test_id, test['name'], errors, RESOLUTIONS)
            all_errors[test_id] = errors

    if len(all_errors) > 0:
        plot_convergence(all_errors, test_ids, RESOLUTIONS)

    print("\nDone!")

if __name__ == '__main__':
    main()