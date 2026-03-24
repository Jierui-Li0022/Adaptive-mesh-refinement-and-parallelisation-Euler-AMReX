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

def extract_y_slice_composite(plotfile_dir, GAMMA):
    prob_lo,prob_hi,max_level,ncomp,time=read_amrex_header(plotfile_dir)
    x0,y0,data0,_,_=read_amrex_plotfile(plotfile_dir)
    nx0,ny0=len(x0),len(y0); dy0=y0[1]-y0[0]; imid0=nx0//2
    all_y=[]; all_data=[]; all_level=[]
    for j in range(ny0):
        all_y.append(y0[j]); all_data.append(data0[imid0,j,:ncomp]); all_level.append(0)
    for lev in range(1,max_level+1):
        grids=read_amrex_level(plotfile_dir,lev)
        if grids is None: continue
        ref=2**lev; dx_lev=(prob_hi[0]-prob_lo[0])/(nx0*ref); dy_lev=(prob_hi[1]-prob_lo[1])/(ny0*ref)
        for grid in grids:
            x_lo=prob_lo[0]+grid['lo_i']*dx_lev; x_hi=prob_lo[0]+(grid['hi_i']+1)*dx_lev
            if 0.5<x_lo or 0.5>=x_hi: continue
            i_local=int((0.5-x_lo)/dx_lev)
            if i_local>=grid['gx']: i_local=grid['gx']-1
            for j_local in range(grid['gy']):
                j_global=grid['lo_j']+j_local
                y_cell=prob_lo[1]+(j_global+0.5)*dy_lev
                all_y.append(y_cell); all_data.append(grid['data'][:,j_local,i_local]); all_level.append(lev)
    all_y=np.array(all_y); all_data=np.array(all_data); all_level=np.array(all_level)
    keep=np.ones(len(all_y),dtype=bool)
    for i in range(len(all_y)):
        if all_level[i]==0:
            for j in range(len(all_y)):
                if all_level[j]>all_level[i] and abs(all_y[j]-all_y[i])<dy0*0.6:
                    keep[i]=False; break
    all_y=all_y[keep]; all_data=all_data[keep]
    si=np.argsort(all_y); all_y=all_y[si]; all_data=all_data[si]
    rho=all_data[:,0]; mx=all_data[:,1]; my=all_data[:,2]; E=all_data[:,3]; rhoS=all_data[:,4]
    u=mx/rho; v=my/rho; ke=0.5*rho*(u**2+v**2); p=(GAMMA-1.0)*(E-ke)
    e=np.where(ke>0.99*E, rhoS/rho, p/(rho*(GAMMA-1.0)))
    return all_y,rho,v,p,e

def extract_y_slice_level0(x,data,GAMMA):
    imid=len(x)//2; rho=data[imid,:,0]; mx=data[imid,:,1]; my=data[imid,:,2]; E=data[imid,:,3]; rhoS=data[imid,:,4]
    u=mx/rho; v=my/rho; ke=0.5*rho*(u**2+v**2); p=(GAMMA-1.0)*(E-ke)
    e=np.where(ke>0.99*E, rhoS/rho, p/(rho*(GAMMA-1.0)))
    return rho,v,p,e

TESTS = {
    1: {"name":"Sod Shock Tube","rhoL":1.0,"uL":0.0,"pL":1.0,"rhoR":0.125,"uR":0.0,"pR":0.1,"x0":0.5,"t":0.25,
        "dir":"test1_sod_y","dir_amr":"test1_sod_y"},
    2: {"name":"123 Problem","rhoL":1.0,"uL":-2.0,"pL":0.4,"rhoR":1.0,"uR":2.0,"pR":0.4,"x0":0.5,"t":0.15,
        "dir":"test2_123_y","dir_amr":"test2_123_y"},
    3: {"name":"Left Blast Wave","rhoL":1.0,"uL":0.0,"pL":1000.0,"rhoR":1.0,"uR":0.0,"pR":0.01,"x0":0.5,"t":0.012,
        "dir":"test3_blast_left_y","dir_amr":"test3_blast_left_y"},
    4: {"name":"Right Blast Wave","rhoL":1.0,"uL":0.0,"pL":0.01,"rhoR":1.0,"uR":0.0,"pR":100.0,"x0":0.5,"t":0.035,
        "dir":"test4_blast_right_y","dir_amr":"test4_blast_right_y"},
    5: {"name":"Two-Shock Collision","rhoL":5.99924,"uL":19.5975,"pL":460.894,"rhoR":5.99242,"uR":-6.19633,"pR":46.0950,"x0":0.4,"t":0.035,
        "dir":"test5_collision_y","dir_amr":"test5_collision_y"},
}

def find_max_diff_location(y_na, field_na, y_amr, field_amr):
    amr_on_na = np.interp(y_na, y_amr, field_amr)
    diff = np.abs(amr_on_na - field_na)
    idx = np.argmax(diff)
    return y_na[idx]

def auto_zoom(ax, coord_na, field_na, coord_amr, field_amr, coord_exact, field_exact,
              has_noamr, has_amr, center, half_width=0.05):
    xl = center - half_width; xr = center + half_width
    if xl < 0.01: xl = 0.01; xr = xl + 2*half_width
    if xr > 0.99: xr = 0.99; xl = xr - 2*half_width
    all_y = []
    if has_noamr:
        mask = (coord_na >= xl) & (coord_na <= xr)
        if np.any(mask): all_y.extend(field_na[mask])
    if has_amr:
        mask = (coord_amr >= xl) & (coord_amr <= xr)
        if np.any(mask): all_y.extend(field_amr[mask])
    mask = (coord_exact >= xl) & (coord_exact <= xr)
    if np.any(mask): all_y.extend(field_exact[mask])
    if len(all_y) == 0: return
    ymin = min(all_y); ymax = max(all_y)
    margin = (ymax - ymin) * 0.1
    if margin < 1e-10: margin = abs(ymin) * 0.05 + 1e-6
    yl = ymin - margin; yr = ymax + margin
    inset_pos = [0.08, 0.55, 0.38, 0.38] if center > 0.5 else [0.55, 0.55, 0.38, 0.38]
    axins = ax.inset_axes(inset_pos)
    if has_noamr: axins.plot(coord_na, field_na, 'bo', markersize=3, alpha=0.5)
    if has_amr: axins.plot(coord_amr, field_amr, 'r.', markersize=2, alpha=0.8)
    axins.plot(coord_exact, field_exact, 'k-', linewidth=1.5)
    axins.set_xlim(xl, xr); axins.set_ylim(yl, yr)
    axins.grid(True, alpha=0.3); axins.tick_params(labelsize=7)
    axins.ticklabel_format(useOffset=False, style="plain")
    ax.indicate_inset_zoom(axins, edgecolor="gray", linewidth=1.5)

def main():
    os.makedirs("plot/result/result_2d_amr/y",exist_ok=True)
    test_ids=[1,2,3,4,5] if sys.argv[1]=='all' else [int(sys.argv[1])]
    for tid in test_ids:
        T=TESTS[tid]
        print(f"\n=== 2D y-direction AMR: Toro Test {tid}: {T['name']} ===")
        y_exact=np.linspace(0.001,0.999,2000)
        rho_ex,u_ex,p_ex=exact_riemann(y_exact,T['t'],T['rhoL'],T['uL'],T['pL'],T['rhoR'],T['uR'],T['pR'],T['x0'],GAMMA)
        e_ex=p_ex/(rho_ex*(GAMMA-1.0))

        rd=os.path.join("output_2d","y_direction",T['dir']); has_na=False
        y_na=rho_na=v_na=p_na=e_na=None
        if os.path.exists(rd):
            try:
                pf=find_final_plotfile(rd); print(f"  Reading no-AMR: {pf}")
                xd,yd,dd,_,_=read_amrex_plotfile(pf)
                rho_na,v_na,p_na,e_na=extract_y_slice_level0(xd,dd,GAMMA); y_na=yd; has_na=True
            except Exception as ex: print(f"  Error: {ex}")

            rd2=os.path.join("output_2d","y_direction_amr",T['dir_amr']); has_amr=False
        y_amr=rho_amr=v_amr=p_amr=e_amr=None
        if os.path.exists(rd2):
            try:
                pf2=find_final_plotfile(rd2); print(f"  Reading AMR: {pf2}")
                y_amr,rho_amr,v_amr,p_amr,e_amr=extract_y_slice_composite(pf2,GAMMA); has_amr=True
                print(f"  AMR composite: {len(y_amr)} points")
            except Exception as ex: print(f"  Error: {ex}"); import traceback; traceback.print_exc()

        if not has_na and not has_amr: print("  No data"); continue

        if has_na and has_amr:
            zoom_center = find_max_diff_location(y_na, rho_na, y_amr, rho_amr)
            print(f"  Max difference at y = {zoom_center:.4f}")
        else:
            zoom_center = T['x0']

        fig,axes=plt.subplots(2,2,figsize=(14,10))

        na_f=[rho_na,v_na,p_na,e_na]; amr_f=[rho_amr,v_amr,p_amr,e_amr]
        ex_f=[rho_ex,u_ex,p_ex,e_ex]; titles=['Density','Velocity (v)','Pressure','Internal Energy']

        for idx,ax in enumerate(axes.flat):
            if has_na: ax.plot(y_na,na_f[idx],'bo',markersize=2,alpha=0.4,label='No AMR (400×400)')
            if has_amr: ax.plot(y_amr,amr_f[idx],'r.',markersize=1.5,alpha=0.7,label='AMR (400×400)')
            ax.plot(y_exact,ex_f[idx],'k-',linewidth=1.5,label='Exact')
            ax.set_xlabel('y'); ax.set_ylabel(titles[idx]); ax.set_title(titles[idx])
            ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

            if has_na and has_amr:
                auto_zoom(ax, y_na, na_f[idx], y_amr, amr_f[idx], y_exact, ex_f[idx],
                         has_na, has_amr, zoom_center, half_width=0.04)

        plt.tight_layout()
        out=f"plot/result/result_2d_amr/y/2d_y_amr_{T['dir']}.png"
        plt.savefig(out,dpi=200,bbox_inches='tight'); print(f"  Saved: {out}"); plt.close()

    print("\nDone!")

if __name__=='__main__': main()