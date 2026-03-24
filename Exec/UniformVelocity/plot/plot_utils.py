import numpy as np
import os
import glob

GAMMA = 1.4

def find_final_plotfile(res_dir):
    """Return the plotfile with the highest step number in res_dir."""
    dirs = glob.glob(os.path.join(res_dir, "plt*"))
    dirs = [d for d in dirs if '.old' not in os.path.basename(d)]
    if not dirs:
        raise FileNotFoundError(f"No plotfiles found in {res_dir}")
    dirs.sort(key=lambda d: int(os.path.basename(d).replace("plt", "")))
    return dirs[-1]

def read_amrex_header(plotfile_dir):
    """Parse the top-level Header file. Returns prob_lo, prob_hi, max_level, nvars, time."""
    with open(os.path.join(plotfile_dir, "Header"), 'r') as f:
        raw = f.read()
    lines = [l.strip() for l in raw.split('\n') if l.strip() != '']
    idx = 0
    idx += 1                                              # version string
    nvars = int(lines[idx]); idx += 1
    var_names = [lines[idx + i] for i in range(nvars)]
    idx += nvars
    idx += 1                                              # ndim
    time      = float(lines[idx]); idx += 1
    max_level = int(lines[idx]);   idx += 1
    prob_lo   = [float(x) for x in lines[idx].split()]; idx += 1
    prob_hi   = [float(x) for x in lines[idx].split()]; idx += 1
    return prob_lo, prob_hi, max_level, nvars, time, var_names

def _parse_cell_h(plotfile_dir, level):
    """
    Parse Level_N/Cell_H. Returns (fab_boxes, fab_files, fab_offsets, ncomp)
    or None if the level directory does not exist.
    """
    level_dir = os.path.join(plotfile_dir, f"Level_{level}")
    if not os.path.exists(level_dir):
        return None

    with open(os.path.join(level_dir, "Cell_H"), 'r') as f:
        lines = f.readlines()

    ci = 0
    ci += 1                                        # version
    ci += 1                                        # how
    ncomp  = int(lines[ci].strip()); ci += 1
    ci += 1                                        # nghost
    ngrids = int(lines[ci].strip().replace('(', '').replace(')', '').split()[0]); ci += 1

    fab_boxes = []
    for _ in range(ngrids):
        s = lines[ci].strip().replace('(', ' ').replace(')', ' ').replace(',', ' ')
        ci += 1
        nums = [int(x) for x in s.split()]
        fab_boxes.append((nums[0], nums[1], nums[2], nums[3]))
    ci += 1   # closing ")"
    ci += 1   # nfabs line

    fab_files, fab_offsets = [], []
    for _ in range(ngrids):
        parts = lines[ci].strip().split(); ci += 1
        if parts[0] == 'FabOnDisk:':
            fab_files.append(parts[1]); fab_offsets.append(int(parts[2]))
        else:
            fab_files.append(parts[0]); fab_offsets.append(int(parts[1]))

    return fab_boxes, fab_files, fab_offsets, ncomp, level_dir

def read_amrex_plotfile(plotfile_dir):
    """
    Read a 2D AMReX plotfile at level 0.
    Returns (x, y, data, var_names, time) where data has shape (nx, ny, ncomp).
    """
    prob_lo, prob_hi, max_level, nvars, time, var_names = read_amrex_header(plotfile_dir)

    result = _parse_cell_h(plotfile_dir, 0)
    if result is None:
        raise RuntimeError(f"Level_0 not found in {plotfile_dir}")
    fab_boxes, fab_files, fab_offsets, ncomp, level_dir = result

    all_lo_i = min(b[0] for b in fab_boxes)
    all_lo_j = min(b[1] for b in fab_boxes)
    all_hi_i = max(b[2] for b in fab_boxes)
    all_hi_j = max(b[3] for b in fab_boxes)
    nx = all_hi_i - all_lo_i + 1
    ny = all_hi_j - all_lo_j + 1
    dx = [(prob_hi[0] - prob_lo[0]) / nx, (prob_hi[1] - prob_lo[1]) / ny]

    data = np.zeros((nx, ny, ncomp))
    for g, (lo_i, lo_j, hi_i, hi_j) in enumerate(fab_boxes):
        gx = hi_i - lo_i + 1
        gy = hi_j - lo_j + 1
        with open(os.path.join(level_dir, fab_files[g]), 'rb') as f:
            f.seek(fab_offsets[g])
            while f.read(1) != b'\n':
                pass
            raw = np.fromfile(f, dtype=np.float64, count=ncomp * gx * gy)
            raw = raw.reshape((ncomp, gy, gx))
            for n in range(ncomp):
                for j in range(gy):
                    for i in range(gx):
                        data[lo_i - all_lo_i + i, lo_j - all_lo_j + j, n] = raw[n, j, i]

    x = prob_lo[0] + (np.arange(nx) + 0.5) * dx[0]
    y = prob_lo[1] + (np.arange(ny) + 0.5) * dx[1]
    return x, y, data, var_names, time

def read_amrex_level_grids(plotfile_dir, level):
    """
    Read raw grid data for a single AMR level.
    Returns a list of dicts with keys: lo_i, lo_j, hi_i, hi_j, gx, gy, data.
    Returns None if the level does not exist.
    """
    result = _parse_cell_h(plotfile_dir, level)
    if result is None:
        return None
    fab_boxes, fab_files, fab_offsets, ncomp, level_dir = result

    grids = []
    for g, (lo_i, lo_j, hi_i, hi_j) in enumerate(fab_boxes):
        gx = hi_i - lo_i + 1
        gy = hi_j - lo_j + 1
        with open(os.path.join(level_dir, fab_files[g]), 'rb') as f:
            f.seek(fab_offsets[g])
            while f.read(1) != b'\n':
                pass
            raw = np.fromfile(f, dtype=np.float64, count=ncomp * gx * gy)
            raw = raw.reshape((ncomp, gy, gx))
        grids.append({'lo_i': lo_i, 'lo_j': lo_j, 'hi_i': hi_i, 'hi_j': hi_j,
                      'gx': gx, 'gy': gy, 'data': raw})
    return grids

def exact_riemann(x, t, rhoL, uL, pL, rhoR, uR, pR, x0=0.5, gamma=GAMMA):
    """
    Exact Riemann solver for the 1D ideal-gas Euler equations (Toro, Ch. 4).
    Solves f(p*) = fL(p*) + fR(p*) + (uR - uL) = 0 by Newton iteration,
    then samples the self-similar solution at each point in x.
    """
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    def fk(p, rho_k, p_k, c_k):
        A_k = 2.0 / ((gamma + 1.0) * rho_k)
        B_k = (gamma - 1.0) / (gamma + 1.0) * p_k
        if p > p_k:   # shock
            return (p - p_k) * np.sqrt(A_k / (p + B_k))
        else:          # rarefaction
            return 2.0 * c_k / (gamma - 1.0) * ((p / p_k)**((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    def dfk(p, rho_k, p_k, c_k):
        A_k = 2.0 / ((gamma + 1.0) * rho_k)
        B_k = (gamma - 1.0) / (gamma + 1.0) * p_k
        if p > p_k:
            return np.sqrt(A_k / (p + B_k)) * (1.0 - (p - p_k) / (2.0 * (p + B_k)))
        else:
            return 1.0 / (rho_k * c_k) * (p / p_k)**(-(gamma + 1.0) / (2.0 * gamma))

    # Newton iteration for p*
    p_star = 0.5 * (pL + pR)
    for _ in range(100):
        dp = -(fk(p_star, rhoL, pL, cL) + fk(p_star, rhoR, pR, cR) + uR - uL) / \
              (dfk(p_star, rhoL, pL, cL) + dfk(p_star, rhoR, pR, cR))
        p_star += dp
        if p_star < 1e-14:
            p_star = 1e-14
        if abs(dp) < 1e-12 * (1.0 + abs(p_star)):
            break

    u_star = 0.5 * (uL + uR) + 0.5 * (fk(p_star, rhoR, pR, cR) - fk(p_star, rhoL, pL, cL))

    rho_out = np.zeros_like(x)
    u_out   = np.zeros_like(x)
    p_out   = np.zeros_like(x)

    for i, xi in enumerate(x):
        S = (xi - x0) / t

        # Left wave
        if p_star <= pL:   # left rarefaction
            c_starL = cL * (p_star / pL)**((gamma - 1.0) / (2.0 * gamma))
            if S <= uL - cL:
                rho_out[i], u_out[i], p_out[i] = rhoL, uL, pL
            elif S <= u_star - c_starL:
                rho_out[i] = rhoL * (2.0/(gamma+1.0) + (gamma-1.0)/((gamma+1.0)*cL)*(uL - S))**(2.0/(gamma-1.0))
                u_out[i]   = 2.0/(gamma+1.0) * (cL + (gamma-1.0)/2.0 * uL + S)
                p_out[i]   = pL  * (2.0/(gamma+1.0) + (gamma-1.0)/((gamma+1.0)*cL)*(uL - S))**(2.0*gamma/(gamma-1.0))
            elif S <= u_star:
                rho_out[i], u_out[i], p_out[i] = rhoL * (p_star/pL)**(1.0/gamma), u_star, p_star
        else:               # left shock
            S_L = uL - cL * np.sqrt((gamma+1.0)/(2.0*gamma)*p_star/pL + (gamma-1.0)/(2.0*gamma))
            if S <= S_L:
                rho_out[i], u_out[i], p_out[i] = rhoL, uL, pL
            elif S <= u_star:
                rho_starL = rhoL * ((p_star/pL + (gamma-1.0)/(gamma+1.0)) /
                                    ((gamma-1.0)/(gamma+1.0) * p_star/pL + 1.0))
                rho_out[i], u_out[i], p_out[i] = rho_starL, u_star, p_star

        # Right wave
        if S > u_star:
            if p_star <= pR:   # right rarefaction
                c_starR = cR * (p_star / pR)**((gamma - 1.0) / (2.0 * gamma))
                if S >= uR + cR:
                    rho_out[i], u_out[i], p_out[i] = rhoR, uR, pR
                elif S >= u_star + c_starR:
                    rho_out[i] = rhoR * (2.0/(gamma+1.0) - (gamma-1.0)/((gamma+1.0)*cR)*(uR - S))**(2.0/(gamma-1.0))
                    u_out[i]   = 2.0/(gamma+1.0) * (-cR + (gamma-1.0)/2.0 * uR + S)
                    p_out[i]   = pR  * (2.0/(gamma+1.0) - (gamma-1.0)/((gamma+1.0)*cR)*(uR - S))**(2.0*gamma/(gamma-1.0))
                else:
                    rho_out[i], u_out[i], p_out[i] = rhoR * (p_star/pR)**(1.0/gamma), u_star, p_star
            else:               # right shock
                S_R = uR + cR * np.sqrt((gamma+1.0)/(2.0*gamma)*p_star/pR + (gamma-1.0)/(2.0*gamma))
                if S >= S_R:
                    rho_out[i], u_out[i], p_out[i] = rhoR, uR, pR
                else:
                    rho_starR = rhoR * ((p_star/pR + (gamma-1.0)/(gamma+1.0)) /
                                        ((gamma-1.0)/(gamma+1.0) * p_star/pR + 1.0))
                    rho_out[i], u_out[i], p_out[i] = rho_starR, u_star, p_star

    return rho_out, u_out, p_out

def compute_L1_error(x_num, q_num, x_exact, q_exact):
    """L1 error: interpolate exact solution onto the numerical grid, integrate."""
    q_interp = np.interp(x_num, x_exact, q_exact)
    dx = x_num[1] - x_num[0]
    return np.sum(np.abs(q_num - q_interp)) * dx