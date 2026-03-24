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
        "dir": "test1_sod_x"},
    2: {"name": "123 Problem",
        "rhoL": 1.0, "uL": -2.0, "pL": 0.4,
        "rhoR": 1.0, "uR": 2.0, "pR": 0.4,
        "x0": 0.5, "t": 0.15,
        "dir": "test2_123_x"},
    3: {"name": "Left Blast Wave",
        "rhoL": 1.0, "uL": 0.0, "pL": 1000.0,
        "rhoR": 1.0, "uR": 0.0, "pR": 0.01,
        "x0": 0.5, "t": 0.012,
        "dir": "test3_blast_left_x"},
    4: {"name": "Right Blast Wave",
        "rhoL": 1.0, "uL": 0.0, "pL": 0.01,
        "rhoR": 1.0, "uR": 0.0, "pR": 100.0,
        "x0": 0.5, "t": 0.035,
        "dir": "test4_blast_right_x"},
    5: {"name": "Two-Shock Collision",
        "rhoL": 5.99924, "uL": 19.5975, "pL": 460.894,
        "rhoR": 5.99242, "uR": -6.19633, "pR": 46.0950,
        "x0": 0.4, "t": 0.035,
        "dir": "test5_collision_x"},
}

def main():
    if len(sys.argv) < 2:
        print("  e.g. python3 plot_toro_2d_x.py 1")
        print("  or   python3 plot_toro_2d_x.py all")
        sys.exit(1)

    os.makedirs("result/result_2d", exist_ok=True)

    if sys.argv[1] == 'all':
        test_ids = [1, 2, 3, 4, 5]
    else:
        test_ids = [int(sys.argv[1])]
    for test_id in test_ids:
        test = TESTS[test_id]
        print(f"\n=== 2D x-direction: Toro Test {test_id}: {test['name']} ===")

        x_exact = np.linspace(0.001, 0.999, 2000)
        rho_exact, u_exact, p_exact = exact_riemann(
        x_exact, test['t'],
            test['rhoL'], test['uL'], test['pL'],
            test['rhoR'], test['uR'], test['pR'],
            test['x0'], GAMMA)
        e_exact = p_exact / (rho_exact * (GAMMA - 1.0))

        res_dir = os.path.join("output_2d", "x_direction", test['dir'])
        if not os.path.exists(res_dir):
            print(f"  Directory not found: {res_dir}")
            continue

        try:
            pf = find_final_plotfile(res_dir)
        except FileNotFoundError:
            print(f"  No plotfile found in {res_dir}")
            continue

        print(f"  Reading: {pf}")
        x, y, data, var_names, time = read_amrex_plotfile(pf)

        # Take slice at y = 0.5 (middle row)
        jmid = len(y) // 2
        rho  = data[:, jmid, 0]
        mx   = data[:, jmid, 1]
        my   = data[:, jmid, 2]
        E    = data[:, jmid, 3]
        rhoS = data[:, jmid, 4]

        u_vel = mx / rho
        v_vel = my / rho
        ke = 0.5 * rho * (u_vel**2 + v_vel**2)
        p = (GAMMA - 1.0) * (E - ke)
        # Dual-energy switching: use entropy variable in near-vacuum region
        S = rhoS / rho
        p_from_S = S * rho**(GAMMA - 1.0)
        e_int = np.where(ke > 0.99 * E, p_from_S / (rho * (GAMMA - 1.0)), p / (rho * (GAMMA - 1.0)))

        # Check y-uniformity: compare two different y-slices
        j_quarter = len(y) // 4
        rho_check = data[:, j_quarter, 0]
        max_y_diff = np.max(np.abs(rho - rho_check))
        print(f"  Max y-variation in density: {max_y_diff:.2e} (should be ~0 for x-split)")

        e_rho  = compute_L1_error(x, rho,   x_exact, rho_exact)
        e_u    = compute_L1_error(x, u_vel, x_exact, u_exact)
        e_p    = compute_L1_error(x, p,     x_exact, p_exact)
        e_eint = compute_L1_error(x, e_int, x_exact, e_exact)

        print(f"  L1 errors: rho={e_rho:.6e}, u={e_u:.6e}, p={e_p:.6e}, e={e_eint:.6e}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].plot(x, rho,   'bo', markersize=2, label='2D (y=0.5 slice)')
        axes[0, 1].plot(x, u_vel, 'bo', markersize=2, label='2D (y=0.5 slice)')
        axes[1, 0].plot(x, p,     'bo', markersize=2, label='2D (y=0.5 slice)')
        axes[1, 1].plot(x, e_int, 'bo', markersize=2, label='2D (y=0.5 slice)')

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
        outfile = f"plot/result/result_2d/x/2d_x_{test['dir']}.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"  Saved: {outfile}")
        plt.close()

    print("\nDone!")

if __name__ == '__main__':
    main()