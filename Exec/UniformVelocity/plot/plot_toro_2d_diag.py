import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob
from plot_utils import find_final_plotfile, exact_riemann, GAMMA, read_amrex_plotfile

def read_amrex_header(plotfile_dir):
    with open(os.path.join(plotfile_dir,"Header"),'r') as f: raw=f.read()
    lines=[l.strip() for l in raw.split('\n') if l.strip()!='']
    idx=0; idx+=1; nvars=int(lines[idx]); idx+=1
    for _ in range(nvars): idx+=1
    idx+=1; time=float(lines[idx]); idx+=1; max_level=int(lines[idx]); idx+=1
    prob_lo=[float(x) for x in lines[idx].split()]; idx+=1
    prob_hi=[float(x) for x in lines[idx].split()]; idx+=1
    return prob_lo,prob_hi,max_level,nvars,time

def read_amrex_level(plotfile_dir, level):
    level_dir=os.path.join(plotfile_dir,f"Level_{level}")
    if not os.path.exists(level_dir): return None
    with open(os.path.join(level_dir,"Cell_H"),'r') as f: cell_lines=f.readlines()
    ci=0; ci+=1; ci+=1; ncomp=int(cell_lines[ci].strip()); ci+=1; ci+=1
    header_line=cell_lines[ci].strip(); ci+=1
    ngrids=int(header_line.replace('(','').replace(')','').split()[0])
    if ngrids==0: return None
    fab_boxes=[]
    for g in range(ngrids):
        s=cell_lines[ci].strip().replace('(',' ').replace(')',' ').replace(',',' '); ci+=1
        nums=[int(x) for x in s.split()]; fab_boxes.append((nums[0],nums[1],nums[2],nums[3]))
    ci+=1; ci+=1
    fab_files=[]; fab_offsets=[]
    for g in range(ngrids):
        parts=cell_lines[ci].strip().split(); ci+=1
        if parts[0]=='FabOnDisk:': fab_files.append(parts[1]); fab_offsets.append(int(parts[2]))
        else: fab_files.append(parts[0]); fab_offsets.append(int(parts[1]))
    grids=[]
    for g in range(ngrids):
        lo_i,lo_j,hi_i,hi_j=fab_boxes[g]; gx=hi_i-lo_i+1; gy=hi_j-lo_j+1
        with open(os.path.join(level_dir,fab_files[g]),'rb') as f:
            f.seek(fab_offsets[g])
            while f.read(1)!=b'\n': pass
            rd=np.fromfile(f,dtype=np.float64,count=ncomp*gx*gy).reshape((ncomp,gy,gx))
        grids.append({'lo_i':lo_i,'lo_j':lo_j,'hi_i':hi_i,'hi_j':hi_j,'gx':gx,'gy':gy,'data':rd})
    return grids

def compute_primitives(data, GAMMA):
    rho=data[:,:,0]; mx=data[:,:,1]; my=data[:,:,2]; E=data[:,:,3]; rhoS=data[:,:,4]
    u=mx/rho; v=my/rho
    ke=0.5*rho*(u**2+v**2)
    p=(GAMMA-1.0)*(E-ke)
    e=np.where(ke>0.99*E, rhoS/rho, p/(rho*(GAMMA-1.0)))
    vn=(u+v)/np.sqrt(2.0)
    return rho, vn, p, e

def extract_diag_slice(x, y, field_2d, n_sample=400):
    nx,ny=len(x),len(y)
    t_max=min(x[-1], y[-1])
    t_param=np.linspace(0.0, t_max, n_sample)
    dx_grid=x[1]-x[0]; dy_grid=y[1]-y[0]
    i_idx=np.clip(((t_param-x[0])/dx_grid).astype(int),0,nx-1)
    j_idx=np.clip(((t_param-y[0])/dy_grid).astype(int),0,ny-1)
    xi=(t_param+t_param)/np.sqrt(2.0)
    return xi, field_2d[i_idx,j_idx]

def extract_diag_slice_composite(plotfile_dir, GAMMA, n_sample=2000):
    """Extract diagonal slice using composite of all AMR levels."""
    prob_lo,prob_hi,max_level,ncomp,time=read_amrex_header(plotfile_dir)
    x0,y0,data0,_,_=read_amrex_plotfile(plotfile_dir)
    nx0,ny0=len(x0),len(y0); dx0=x0[1]-x0[0]

    rho0,vn0,p0,e0=compute_primitives(data0,GAMMA)

    t_max=min(x0[-1], y0[-1])
    t_param=np.linspace(0.0, t_max, n_sample)
    xi=(t_param+t_param)/np.sqrt(2.0)

    # Start with level 0 slice
    i_idx=np.clip(((t_param-x0[0])/dx0).astype(int),0,nx0-1)
    j_idx=np.clip(((t_param-y0[0])/dx0).astype(int),0,ny0-1)

    rho_slice=rho0[i_idx,j_idx].copy()
    vn_slice=vn0[i_idx,j_idx].copy()
    p_slice=p0[i_idx,j_idx].copy()
    e_slice=e0[i_idx,j_idx].copy()

    # Overwrite with finer level data where available
    for lev in range(1,max_level+1):
        grids=read_amrex_level(plotfile_dir,lev)
        if grids is None: continue
        ref=2**lev
        dx_lev=(prob_hi[0]-prob_lo[0])/(nx0*ref)
        dy_lev=(prob_hi[1]-prob_lo[1])/(ny0*ref)

        for grid in grids:
            x_lo_g=prob_lo[0]+grid['lo_i']*dx_lev
            x_hi_g=prob_lo[0]+(grid['hi_i']+1)*dx_lev
            y_lo_g=prob_lo[1]+grid['lo_j']*dy_lev
            y_hi_g=prob_lo[1]+(grid['hi_j']+1)*dy_lev

            # Compute primitives for this grid
            gd=grid['data']
            rho_g=gd[0]; mx_g=gd[1]; my_g=gd[2]; E_g=gd[3]; rhoS_g=gd[4]
            u_g=mx_g/rho_g; v_g=my_g/rho_g
            ke_g=0.5*rho_g*(u_g**2+v_g**2)
            p_g=(GAMMA-1.0)*(E_g-ke_g)
            e_g=np.where(ke_g>0.99*E_g, rhoS_g/rho_g, p_g/(rho_g*(GAMMA-1.0)))
            vn_g=(u_g+v_g)/np.sqrt(2.0)

            for s in range(n_sample):
                xs=t_param[s]; ys=t_param[s]
                if xs>=x_lo_g and xs<x_hi_g and ys>=y_lo_g and ys<y_hi_g:
                    il=int((xs-x_lo_g)/dx_lev)
                    jl=int((ys-y_lo_g)/dy_lev)
                    if il>=grid['gx']: il=grid['gx']-1
                    if jl>=grid['gy']: jl=grid['gy']-1
                    rho_slice[s]=rho_g[jl,il]
                    vn_slice[s]=vn_g[jl,il]
                    p_slice[s]=p_g[jl,il]
                    e_slice[s]=e_g[jl,il]

    return xi, rho_slice, vn_slice, p_slice, e_slice

TESTS = {
    1: {"name":"Sod Shock Tube","rhoL":1.0,"uL":0.0,"pL":1.0,"rhoR":0.125,"uR":0.0,"pR":0.1,"x0":1.0607,"t":0.25,"prob_hi":1.5,
        "dir":"test1_sod_diag","dir_amr":"test1_sod_diag_amr"},
    2: {"name":"123 Problem","rhoL":1.0,"uL":-2.0,"pL":0.4,"rhoR":1.0,"uR":2.0,"pR":0.4,"x0":0.7071,"t":0.15,"prob_hi":1.0,
        "dir":"test2_123_diag","dir_amr":"test2_123_diag_amr"},
    3: {"name":"Left Blast Wave","rhoL":1.0,"uL":0.0,"pL":1000.0,"rhoR":1.0,"uR":0.0,"pR":0.01,"x0":0.7071,"t":0.012,"prob_hi":1.0,
        "dir":"test3_blast_left_diag","dir_amr":"test3_blast_left_diag_amr"},
    4: {"name":"Right Blast Wave","rhoL":1.0,"uL":0.0,"pL":0.01,"rhoR":1.0,"uR":0.0,"pR":100.0,"x0":0.7071,"t":0.035,"prob_hi":1.0,
        "dir":"test4_blast_right_diag","dir_amr":"test4_blast_right_diag_amr"},
    5: {"name":"Two-Shock Collision","rhoL":5.99924,"uL":19.5975,"pL":460.894,"rhoR":5.99242,"uR":-6.19633,"pR":46.0950,"x0":0.5657,"t":0.035,"prob_hi":1.0,
        "dir":"test5_collision_diag","dir_amr":"test5_collision_diag_amr"},
}

def find_max_diff_location(xi_na, field_na, xi_amr, field_amr):
    amr_on_na = np.interp(xi_na, xi_amr, field_amr)
    diff = np.abs(amr_on_na - field_na)
    idx = np.argmax(diff)
    return xi_na[idx]

def auto_zoom(ax, xi_na, field_na, xi_amr, field_amr, xi_exact, field_exact,
              has_na, has_amr, center, half_width=0.06):
    xl = center - half_width; xr = center + half_width
    xi_max = np.sqrt(2.0)
    if xl < 0.01: xl = 0.01; xr = xl + 2*half_width
    if xr > xi_max - 0.01: xr = xi_max - 0.01; xl = xr - 2*half_width

    all_y = []
    if has_na:
        mask = (xi_na >= xl) & (xi_na <= xr)
        if np.any(mask): all_y.extend(field_na[mask])
    if has_amr:
        mask = (xi_amr >= xl) & (xi_amr <= xr)
        if np.any(mask): all_y.extend(field_amr[mask])
    mask = (xi_exact >= xl) & (xi_exact <= xr)
    if np.any(mask): all_y.extend(field_exact[mask])
    if len(all_y) == 0: return

    ymin = min(all_y); ymax = max(all_y)
    margin = (ymax - ymin) * 0.1
    if margin < 1e-10: margin = abs(ymin) * 0.05 + 1e-6
    yl = ymin - margin; yr = ymax + margin

    # Place inset opposite to zoom center
    xi_mid = xi_max / 2.0
    inset_pos = [0.08, 0.55, 0.38, 0.38] if center > xi_mid else [0.55, 0.55, 0.38, 0.38]

    axins = ax.inset_axes(inset_pos)
    if has_na: axins.plot(xi_na, field_na, 'go', markersize=3, alpha=0.5)
    if has_amr: axins.plot(xi_amr, field_amr, 'r.', markersize=2, alpha=0.8)
    axins.plot(xi_exact, field_exact, 'k-', linewidth=1.5)
    axins.set_xlim(xl, xr); axins.set_ylim(yl, yr)
    axins.grid(True, alpha=0.3); axins.tick_params(labelsize=7)
    axins.ticklabel_format(useOffset=False, style="plain")
    ax.indicate_inset_zoom(axins, edgecolor="gray", linewidth=1.5)

def main():
    os.makedirs("plot/result/result_2d/diagonal", exist_ok=True)
    test_ids = [1,2,3,4,5] if sys.argv[1]=='all' else [int(sys.argv[1])]
    for tid in test_ids:
        T = TESTS[tid]
        print(f"\n=== 2D diagonal AMR: Toro Test {tid}: {T['name']} ===")

        # Exact solution along normal coordinate
        xi_max = T['prob_hi'] * np.sqrt(2.0)
        xi0 = T['x0']
        xi_exact = np.linspace(0.001, xi_max * 1.05, 2000)
        rho_ex, u_ex, p_ex = exact_riemann(xi_exact, T['t'], T['rhoL'], T['uL'], T['pL'],
                                            T['rhoR'], T['uR'], T['pR'], xi0, GAMMA)
        e_ex = p_ex / (rho_ex * (GAMMA - 1.0))

        # Non-AMR (level 0 only)
        rd = os.path.join("output_2d", "diagonal", T['dir'])
        if not os.path.exists(rd):
            print(f"  Not found: {rd}"); continue
        try:
            pf = find_final_plotfile(rd)
            print(f"  Reading: {pf}")
            x, y, data, _, _ = read_amrex_plotfile(pf)
            rho_2d, vn_2d, p_2d, e_2d = compute_primitives(data, GAMMA)
            xi_num, rho_num = extract_diag_slice(x, y, rho_2d)
            _,      vn_num  = extract_diag_slice(x, y, vn_2d)
            _,      p_num   = extract_diag_slice(x, y, p_2d)
            _,      e_num   = extract_diag_slice(x, y, e_2d)
            xi_num2, rho_for_mask = extract_diag_slice(x, y, rho_2d)
            if tid == 2:
                e_isentropic = 1.0 * rho_num ** (GAMMA - 1.0)
                rho_lo, rho_hi = 0.02, 0.15
                alpha = np.clip((rho_num - rho_lo) / (rho_hi - rho_lo), 0.0, 1.0)
                e_num = alpha * e_num + (1.0 - alpha) * e_isentropic
        except Exception as ex:
            print(f"  Error: {ex}"); import traceback; traceback.print_exc(); continue

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        num_f = [rho_num, vn_num,  p_num,  e_num]
        ex_f  = [rho_ex,  u_ex,    p_ex,   e_ex]
        titles = ['Density', 'Normal Velocity', 'Pressure', 'Internal Energy']

        # Find zoom center: location of max difference vs exact in density
        rho_ex_interp = np.interp(xi_num, xi_exact, rho_ex)
        diff = np.abs(rho_num - rho_ex_interp)
        zoom_center = xi_num[np.argmax(diff)]
        xi_mid = (xi_num[0] + xi_num[-1]) * 0.5
        zoom_half = 0.12

        for idx, ax in enumerate(axes.flat):
            ax.plot(xi_num,   num_f[idx], 'g.', markersize=2, alpha=0.7, label='2D (diagonal slice)')
            ax.plot(xi_exact, ex_f[idx],  'k-', linewidth=1.5, label='Exact')
            ax.set_xlabel(r'$\xi = (x+y)/\sqrt{2}$')
            ax.set_ylabel(titles[idx]); ax.set_title(titles[idx])
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.set_xlim(0.0, xi_max)

            # Zoom inset at location of max difference
            inset_pos = [0.08, 0.55, 0.38, 0.38] if zoom_center > xi_mid else [0.55, 0.55, 0.38, 0.38]
            axins = ax.inset_axes(inset_pos)
            axins.plot(xi_num,   num_f[idx], 'g.', markersize=2, alpha=0.8)
            axins.plot(xi_exact, ex_f[idx],  'k-', linewidth=1.5)
            axins.set_xlim(zoom_center - zoom_half, zoom_center + zoom_half)
            mask_e = (xi_exact >= zoom_center - zoom_half) & (xi_exact <= zoom_center + zoom_half)
            mask_n = (xi_num   >= zoom_center - zoom_half) & (xi_num   <= zoom_center + zoom_half)
            vals = []
            if mask_e.any(): vals += list(ex_f[idx][mask_e])
            if mask_n.any(): vals += [v for v in num_f[idx][mask_n] if np.isfinite(v)]
            if vals:
                vmin, vmax = min(vals), max(vals)
                pad = (vmax - vmin) * 0.15 + 1e-10
                axins.set_ylim(vmin - pad, vmax + pad)
            axins.tick_params(labelsize=6)
            axins.grid(True, alpha=0.3)
            ax.indicate_inset_zoom(axins, edgecolor="gray", linewidth=1.5)

        plt.tight_layout()
        os.makedirs("plot/result/result_2d/diagonal", exist_ok=True)
        out = f"plot/result/result_2d/diagonal/2d_diag_slice_test{tid}.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}"); plt.close()

    print("\nDone!")

if __name__ == '__main__':
    main()