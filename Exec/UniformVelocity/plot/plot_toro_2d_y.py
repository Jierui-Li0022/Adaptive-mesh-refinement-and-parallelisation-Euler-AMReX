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
        "dir": "test1_sod_y"},
    2: {"name": "123 Problem",
        "rhoL": 1.0, "uL": -2.0, "pL": 0.4,
        "rhoR": 1.0, "uR": 2.0, "pR": 0.4,
        "x0": 0.5, "t": 0.15,
        "dir": "test2_123_y"},
    3: {"name": "Left Blast Wave",
        "rhoL": 1.0, "uL": 0.0, "pL": 1000.0,
        "rhoR": 1.0, "uR": 0.0, "pR": 0.01,
        "x0": 0.5, "t": 0.012,
        "dir": "test3_blast_left_y"},
    4: {"name": "Right Blast Wave",
        "rhoL": 1.0, "uL": 0.0, "pL": 0.01,
        "rhoR": 1.0, "uR": 0.0, "pR": 100.0,
        "x0": 0.5, "t": 0.035,
        "dir": "test4_blast_right_y"},
    5: {"name": "Two-Shock Collision",
        "rhoL": 5.99924, "uL": 19.5975, "pL": 460.894,
        "rhoR": 5.99242, "uR": -6.19633, "pR": 46.0950,
        "x0": 0.4, "t": 0.035,
        "dir": "test5_collision_y"},
}

def main():
    if len(sys.argv) < 2:
        print("  e.g. python3 plot_toro_2d_y.py 1")
        print("  or   python3 plot_toro_2d_y.py all")
        sys.exit(1)

    os.makedirs("plot/result/result_2d/y", exist_ok=True)

    if sys.argv[1] == 'all':
        test_ids = [1, 2, 3, 4, 5]
    else:
        test_ids = [int(sys.argv[1])]
    for test_id in test_ids:
        test = TESTS[test_id]
        print(f"\n=== 2D y-direction: Toro Test {test_id}: {test['name']} ===")

        y_exact = np.linspace(0.001, 0.999, 2000)
        rho_exact, u_exact, p_exact = exact_riemann(
        y_exact, test['t'],
            test['rhoL'], test['uL'], test['pL'],
            test['rhoR'], test['uR'], test['pR'],
            test['x0'], GAMMA)
        e_exact = p_exact / (rho_exact * (GAMMA - 1.0))

        res_dir = os.path.join("output_2d", "y_direction", test['dir'])
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

        # Take slice at x = 0.5 (middle column)
        imid = len(x) // 2
        rho  = data[imid, :, 0]
        mx   = data[imid, :, 1]
        my   = data[imid, :, 2]
        E    = data[imid, :, 3]
        rhoS = data[imid, :, 4]

        u_vel = mx / rho
        v_vel = my / rho
        ke = 0.5 * rho * (u_vel**2 + v_vel**2)
        p = (GAMMA - 1.0) * (E - ke)
        # Dual-energy switching: use entropy variable in near-vacuum region
        e_int = np.where(ke > 0.99 * E, rhoS / rho, p / (rho * (GAMMA - 1.0)))

        # For y-split, the "normal velocity" is v (my/rho), not u (mx/rho)
        normal_vel = v_vel

        # Check x-uniformity
        i_quarter = len(x) // 4
        rho_check = data[i_quarter, :, 0]
        max_x_diff = np.max(np.abs(rho - rho_check))
        print(f"  Max x-variation in density: {max_x_diff:.2e} (should be ~0 for y-split)")

        e_rho  = compute_L1_error(y, rho,        y_exact, rho_exact)
        e_u    = compute_L1_error(y, normal_vel,  y_exact, u_exact)
        e_p    = compute_L1_error(y, p,           y_exact, p_exact)
        e_eint = compute_L1_error(y, e_int,       y_exact, e_exact)

        print(f"  L1 errors: rho={e_rho:.6e}, v={e_u:.6e}, p={e_p:.6e}, e={e_eint:.6e}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].plot(y, rho,        'ro', markersize=2, label='2D (x=0.5 slice)')
        axes[0, 1].plot(y, normal_vel, 'ro', markersize=2, label='2D (x=0.5 slice)')
        axes[1, 0].plot(y, p,          'ro', markersize=2, label='2D (x=0.5 slice)')
        axes[1, 1].plot(y, e_int,      'ro', markersize=2, label='2D (x=0.5 slice)')

        axes[0, 0].plot(y_exact, rho_exact, 'k-', linewidth=1.5, label='Exact')
        axes[0, 1].plot(y_exact, u_exact,   'k-', linewidth=1.5, label='Exact')
        axes[1, 0].plot(y_exact, p_exact,   'k-', linewidth=1.5, label='Exact')
        axes[1, 1].plot(y_exact, e_exact,   'k-', linewidth=1.5, label='Exact')

        titles  = ['Density', 'Velocity (v)', 'Pressure', 'Internal Energy']
        ylabels = ['Density', 'Velocity', 'Pressure', 'Internal Energy']
        for idx, ax in enumerate(axes.flat):
            ax.set_xlabel('y')
            ax.set_ylabel(ylabels[idx])
            ax.set_title(titles[idx])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = f"plot/result/result_2d/y/2d_y_{test['dir']}.png"
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"  Saved: {outfile}")
        plt.close()

    print("\nDone!")

if __name__ == '__main__':
    main()