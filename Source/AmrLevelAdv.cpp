#include <AMReX_VisMF.H>
#include <AMReX_TagBox.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Math.H>
#include <cmath>

#include "AmrLevelAdv.H"
#include "tagging_K.H"
#include "EulerEquations.H"
#include "RiemannSolver.H"
#include "Reconstruction.H"
#include "Constants.H"
//#include "Prob.H"
//#include "Kernels.H"

using namespace amrex;

int      AmrLevelAdv::verbose         = 0;
Real     AmrLevelAdv::cfl             = 0.9;  // Default value - can be overwritten in settings file
int      AmrLevelAdv::do_reflux       = 1;

int      AmrLevelAdv::NUM_STATE       = 5;  // Five variables in the state
int      AmrLevelAdv::NUM_GROW        = 3;  // number of ghost cells

// Mechanism for getting code to work on GPU
ProbParm* AmrLevelAdv::h_prob_parm = nullptr;
ProbParm* AmrLevelAdv::d_prob_parm = nullptr;

// Parameters for mesh refinement
int      AmrLevelAdv::max_phierr_lev  = -1;
int      AmrLevelAdv::max_phigrad_lev = -1;
Vector<Real> AmrLevelAdv::phierr;
Vector<Real> AmrLevelAdv::phigrad;

AmrLevelAdv::AmrLevelAdv () = default;

AmrLevelAdv::AmrLevelAdv (Amr&            papa,
                          int             lev,
                          const Geometry& level_geom,
                          const BoxArray& bl,
                          const DistributionMapping& dm,
                          Real            time)
    :
    AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    if (level > 0 && do_reflux) {
        flux_reg = std::make_unique<FluxRegister>(grids,dmap,crse_ratio,level,NUM_STATE);
    }
}


// std::pow wrapper that's callable on both host and device.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pow_real(amrex::Real a, amrex::Real b) noexcept
{
    return ::pow(a, b);
}

AmrLevelAdv::~AmrLevelAdv () = default;

void
AmrLevelAdv::restart (Amr&          papa,
                      std::istream& is,
                      bool          bReadSpecial)
{
    AmrLevel::restart(papa,is,bReadSpecial);

    if (level > 0 && do_reflux) {
        flux_reg = std::make_unique<FluxRegister>(grids,dmap,crse_ratio,level,NUM_STATE);
    }
}

void
AmrLevelAdv::checkPoint (const std::string& dir,
                         std::ostream&      os,
                         VisMF::How         how,
                         bool               dump_old)
{
    AmrLevel::checkPoint(dir, os, how, dump_old);
}

void
AmrLevelAdv::writePlotFile (const std::string& dir,
                             std::ostream&      os,
                            VisMF::How         how)
{
    AmrLevel::writePlotFile (dir,os,how);
}

void
AmrLevelAdv::variableSetUp ()
{
  BL_ASSERT(desc_lst.size() == 0);

  h_prob_parm = new ProbParm{};
  d_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));

  read_params();

  const int storedGhostZones = 0;

  desc_lst.addDescriptor(Phi_Type,IndexType::TheCellType(),
			 StateDescriptor::Point,storedGhostZones,NUM_STATE,
			 &pc_interp);

  int lo_bc[BL_SPACEDIM];
  int hi_bc[BL_SPACEDIM];

  for (int i = 0; i < BL_SPACEDIM; ++i) {
      lo_bc[i] = hi_bc[i] = BCType::foextrap;  // transmissive
  }

  BCRec bc(lo_bc, hi_bc);

  StateDescriptor::BndryFunc bndryfunc(nullfill);
  bndryfunc.setRunOnGPU(true);

  desc_lst.setComponent(Phi_Type, 0, "density",    bc, bndryfunc);
  desc_lst.setComponent(Phi_Type, 1, "momentum_x", bc, bndryfunc);
  desc_lst.setComponent(Phi_Type, 2, "momentum_y", bc, bndryfunc);
  desc_lst.setComponent(Phi_Type, 3, "energy",     bc, bndryfunc);
  desc_lst.setComponent(Phi_Type, 4, "entropy",    bc, bndryfunc);
}

void
AmrLevelAdv::variableCleanUp ()
{
    desc_lst.clear();

    delete h_prob_parm;
    The_Arena()->free(d_prob_parm);
}

void
AmrLevelAdv::initData ()
{
  if (verbose) {
    amrex::Print() << "Initializing the data at level " << level << std::endl;
  }

  const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
  const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

  MultiFab& S_new = get_new_data(Phi_Type);

  for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
  {
    const Box& box = mfi.validbox();
    const Array4<Real>& phi = S_new.array(mfi);

    // Read Riemann initial conditions from the inputs file.
    // Defaults are the Sod shock tube (Test 1 from Toro).
    ParmParse pp("prob");
    Real rhoL = 1.0, uL = 0.0, pL = 1.0;
    Real rhoR = 0.125, uR = 0.0, pR = 0.1;
    Real x0 = 0.5;
    pp.query("rhoL", rhoL);
    pp.query("uL",   uL);
    pp.query("pL",   pL);
    pp.query("rhoR", rhoR);
    pp.query("uR",   uR);
    pp.query("pR",   pR);
    pp.query("x0",   x0);

    // split_dir controls discontinuity orientation:
    //   0 = x-direction
    //   1 = y-direction
    //   2 = diagonal / arbitrary angle (split_angle degrees)
    int  split_dir_flag = 0;
    Real split_angle    = 45.0;
    pp.query("split_dir",   split_dir_flag);
    pp.query("split_angle", split_angle);

    const int split_dir_val = split_dir_flag;

    // Unit normal n = (nx, ny) for the discontinuity interface.
    Real nx = 1.0, ny = 0.0;
    if (split_dir_val == 0) {
        nx = 1.0; ny = 0.0;
    } else if (split_dir_val == 1) {
        nx = 0.0; ny = 1.0;
    } else {
        const Real angle_rad = split_angle * Real(3.14159265358979323846) / Real(180.0);
        nx = std::cos(angle_rad);
        ny = std::sin(angle_rad);
        const Real nrm = std::sqrt(nx*nx + ny*ny);
        if (nrm > 0.0) { nx /= nrm; ny /= nrm; }
    }

    // Use xi = n·x as the 1D coordinate for all orientations so that
    // diagonal and axis-aligned cases share the same x0 threshold.
    int de_flag = 0;
    pp.query("use_dual_energy", de_flag);
    const int use_de = de_flag;

    ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
        Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];

        const Real xi = nx * x + ny * y;
        const bool is_left = (xi < x0);

        Real rho, un, p;
        

        if (is_left) {
            rho = rhoL;  un = uL;  p = pL;
        } else {
            rho = rhoR;  un = uR;  p = pR;
        }

        // Project the 1D normal velocity onto (u, v).
        const Real u = un * nx;
        const Real v = un * ny;

        phi(i,j,k,0) = rho;
        phi(i,j,k,1) = rho * u;
        phi(i,j,k,2) = rho * v;
        phi(i,j,k,3) = p/(GammaGas-1.0) + 0.5*rho*(u*u + v*v);
        phi(i,j,k,4) = 0.0;
    });

    // Dual-energy: store rho*S where S = p / rho^(GammaGas-1) is the
    // specific entropy proxy.
    if (use_de) {
      
      ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          Real rho = phi(i,j,k,0);
          Real mx  = phi(i,j,k,1);
          Real my  = phi(i,j,k,2);
          Real E   = phi(i,j,k,3);

          if (rho < RhoVac) { phi(i,j,k,4) = 0.0; return; }

          Real p = (GammaGas - 1.0) * (E - 0.5*(mx*mx + my*my)/rho);
          if (p < PMin) p = PMin;

          Real S = p / pow_real(rho, GammaGas - 1.0);
          phi(i,j,k,4) = rho * S;
      });
    }
  }

  if (verbose) {
    amrex::Print() << "Done initializing the level " << level
                   << " data " << std::endl;
  }
}

void
AmrLevelAdv::init (AmrLevel &old)
{
  auto* oldlev = (AmrLevelAdv*) &old;

  Real dt_new    = parent->dtLevel(level);
  Real cur_time  = oldlev->state[Phi_Type].curTime();
  Real prev_time = oldlev->state[Phi_Type].prevTime();
  Real dt_old    = cur_time - prev_time;
  setTimeLevel(cur_time,dt_old,dt_new);

  MultiFab& S_new = get_new_data(Phi_Type);

  const int zeroGhosts = 0;
  FillPatch(old, S_new, zeroGhosts, cur_time, Phi_Type, 0, NUM_STATE);
}

void
AmrLevelAdv::init ()
{
  Real dt        = parent->dtLevel(level);
  Real cur_time  = getLevel(level-1).state[Phi_Type].curTime();
  Real prev_time = getLevel(level-1).state[Phi_Type].prevTime();

  Real dt_old = (cur_time - prev_time)/(Real)parent->MaxRefRatio(level-1);

  setTimeLevel(cur_time,dt_old,dt);
  MultiFab& S_new = get_new_data(Phi_Type);

  const int zeroGhosts = 0;
  FillCoarsePatch(S_new, zeroGhosts, cur_time, Phi_Type, 0, NUM_STATE);
}

Real
AmrLevelAdv::advance (Real time,
                      Real dt,
                      int  iteration,
                      int  /*ncycle*/)
{
  MultiFab& S_mm = get_new_data(Phi_Type);

  Real maxval = S_mm.max(0);
  Real minval = S_mm.min(0);
  amrex::Print() << "density max = " << maxval << ", min = " << minval  << std::endl;

  for (int k = 0; k < NUM_STATE_TYPE; k++) {
    state[k].allocOldData();
    state[k].swapTimeLevels(dt);
  }

  MultiFab& S_new = get_new_data(Phi_Type);

  GpuArray<Real,BL_SPACEDIM> dx = geom.CellSizeArray();

  FluxRegister *fine    = nullptr;
  FluxRegister *current = nullptr;

  int finest_level = parent->finestLevel();

  if (do_reflux && level < finest_level) {
    fine = &getFluxReg(level+1);
    fine->setVal(0.0);
  }

  if (do_reflux && level > 0) {
    current = &getFluxReg(level);
  }

  // Cumulative flux integrals over the full timestep, used for refluxing
  // at coarse-fine boundaries.
  MultiFab fluxes[BL_SPACEDIM];
  for (int j = 0; j < BL_SPACEDIM; j++)
  {
    BoxArray ba = S_new.boxArray();
    ba.surroundingNodes(j);
    fluxes[j].define(ba, dmap, NUM_STATE, 0);
    fluxes[j].setVal(0.0);
  }

  MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW);
  FillPatcherFill(Sborder, 0, NUM_STATE, NUM_GROW, time, Phi_Type, 0);

  ParmParse pp_de("prob");
  int use_dual_energy = 0;
  pp_de.query("use_dual_energy", use_dual_energy);

  // CTU (Corner Transport Upwind) unsplit 2D advance.
  // Based on Colella 1990 / LeVeque: the transverse correction is applied
  // before reconstruction to eliminate the Strang commutator error.
  //
  // Pass 1: compute predictor fluxes Fx0, Fy0 with no transverse correction.
  // Pass 2: recompute Fx using Fy0 as transverse input, and vice versa.
  // Update: U^{n+1} = U^n - dt/dx*div(Fx) - dt/dy*div(Fy)
  //
  // 1D and 3D fall back to Strang splitting.

#if (AMREX_SPACEDIM == 1)
  {
    BoxArray ba1 = S_new.boxArray(); ba1.surroundingNodes(0);
    MultiFab flux_tmp(ba1, dmap, NUM_STATE, 0);
    for (MFIter mfi(flux_tmp, true); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      const auto& arr = Sborder.array(mfi);
      const auto& fx  = flux_tmp.array(mfi);
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        Real F_rho,F_mx,F_my,F_E;
        muscl_hancock_flux(arr,i,j,k,dt,dx[0],F_rho,F_mx,F_my,F_E,0,2);
        fx(i,j,k,0)=F_rho; fx(i,j,k,1)=F_mx;
        fx(i,j,k,2)=F_my;  fx(i,j,k,3)=F_E; fx(i,j,k,4)=0.0;
      });
    }
    const int nUp1 = use_dual_energy ? 5 : 4;
    for (MFIter mfi(Sborder); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      const auto& arr = Sborder.array(mfi);
      const auto& fx  = flux_tmp.array(mfi);
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        for (int n=0;n<nUp1;n++) arr(i,j,k,n)-=(dt/dx[0])*(fx(i+1,j,k,n)-fx(i,j,k,n));
      });
    }
    if (do_reflux) {
      flux_tmp.mult(dt*dx[1], 0, NUM_STATE);
      MultiFab::Add(fluxes[0], flux_tmp, 0, 0, NUM_STATE, 0);
    }
  }

#elif (AMREX_SPACEDIM == 2)

  // Pass 1: predictor fluxes without transverse correction.
  // One ghost cell so boundary faces can safely read trans_flux neighbours.
  BoxArray bax = S_new.boxArray(); bax.surroundingNodes(0);
  BoxArray bay = S_new.boxArray(); bay.surroundingNodes(1);
  MultiFab Fx_pred(bax, dmap, NUM_STATE, 1);
  MultiFab Fy_pred(bay, dmap, NUM_STATE, 1);
  Fx_pred.setVal(0.0);
  Fy_pred.setVal(0.0);

  for (MFIter mfi(Fx_pred, true); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    const auto& arr = Sborder.array(mfi);
    const auto& fx  = Fx_pred.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      Real F_rho,F_mx,F_my,F_E;
      muscl_hancock_flux(arr,i,j,k,dt,dx[0],F_rho,F_mx,F_my,F_E,0,2);
      fx(i,j,k,0)=F_rho; fx(i,j,k,1)=F_mx;
      fx(i,j,k,2)=F_my;  fx(i,j,k,3)=F_E;
      // TODO: implement entropy upwinding for component 4
      const Real rhoL0 = arr(i-1,j,k,0); const Real rhoSL0 = arr(i-1,j,k,4);
      const Real rhoR0 = arr(i,  j,k,0); const Real rhoSR0 = arr(i,  j,k,4);
      const Real SL0 = (rhoL0 > RhoVac) ? (rhoSL0/rhoL0) : 0.0;
      const Real SR0 = (rhoR0 > RhoVac) ? (rhoSR0/rhoR0) : 0.0;
      fx(i,j,k,4) = 0.0;
    });
  }
  Fx_pred.FillBoundary(geom.periodicity());

  for (MFIter mfi(Fy_pred, true); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    const auto& arr = Sborder.array(mfi);
    const auto& fy  = Fy_pred.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      Real F_rho,F_mx,F_my,F_E;
      muscl_hancock_flux(arr,i,j,k,dt,dx[1],F_rho,F_mx,F_my,F_E,1,2);
      fy(i,j,k,0)=F_rho; fy(i,j,k,1)=F_mx;
      fy(i,j,k,2)=F_my;  fy(i,j,k,3)=F_E;
      // TODO: implement entropy upwinding for component 4
      const Real rhoL1 = arr(i,j-1,k,0); const Real rhoSL1 = arr(i,j-1,k,4);
      const Real rhoR1 = arr(i,j,  k,0); const Real rhoSR1 = arr(i,j,  k,4);
      const Real SL1 = (rhoL1 > RhoVac) ? (rhoSL1/rhoL1) : 0.0;
      const Real SR1 = (rhoR1 > RhoVac) ? (rhoSR1/rhoR1) : 0.0;
      fy(i,j,k,4) = 0.0;
    });
  }
  Fy_pred.FillBoundary(geom.periodicity());

  // Pass 2: final fluxes with CTU transverse correction.
  // Fx uses Fy_pred as transverse input; Fy uses Fx_pred.
  MultiFab Fx_final(bax, dmap, NUM_STATE, 0);
  MultiFab Fy_final(bay, dmap, NUM_STATE, 0);

  const Real coeff_x = 0.5*dt/dx[1];  // dt/(2*dy) for x-flux transverse correction
  const Real coeff_y = 0.5*dt/dx[0];  // dt/(2*dx) for y-flux transverse correction

  for (MFIter mfi(Fx_final, true); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    const auto& arr = Sborder.array(mfi);
    const auto& fy  = Fy_pred.array(mfi);
    const auto& fx  = Fx_final.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      Real F_rho,F_mx,F_my,F_E;
      muscl_hancock_flux_ctu(arr,fy,i,j,k,dt,dx[0],coeff_x,F_rho,F_mx,F_my,F_E,0,2);
      fx(i,j,k,0)=F_rho; fx(i,j,k,1)=F_mx;
      fx(i,j,k,2)=F_my;  fx(i,j,k,3)=F_E;
      // TODO: implement entropy upwinding for component 4
      const Real rhoLf = arr(i-1,j,k,0); const Real rhoSLf = arr(i-1,j,k,4);
      const Real rhoRf = arr(i,  j,k,0); const Real rhoSRf = arr(i,  j,k,4);
      const Real SLf = (rhoLf > RhoVac) ? (rhoSLf/rhoLf) : 0.0;
      const Real SRf = (rhoRf > RhoVac) ? (rhoSRf/rhoRf) : 0.0;
      fx(i,j,k,4) = 0.0;
    });
  }

  for (MFIter mfi(Fy_final, true); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    const auto& arr = Sborder.array(mfi);
    const auto& fx  = Fx_pred.array(mfi);
    const auto& fy  = Fy_final.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      Real F_rho,F_mx,F_my,F_E;
      muscl_hancock_flux_ctu(arr,fx,i,j,k,dt,dx[1],coeff_y,F_rho,F_mx,F_my,F_E,1,2);
      fy(i,j,k,0)=F_rho; fy(i,j,k,1)=F_mx;
      fy(i,j,k,2)=F_my;  fy(i,j,k,3)=F_E;
      // TODO: implement entropy upwinding for component 4
      const Real rhoLg = arr(i,j-1,k,0); const Real rhoSLg = arr(i,j-1,k,4);
      const Real rhoRg = arr(i,j,  k,0); const Real rhoSRg = arr(i,j,  k,4);
      const Real SLg = (rhoLg > RhoVac) ? (rhoSLg/rhoLg) : 0.0;
      const Real SRg = (rhoRg > RhoVac) ? (rhoSRg/rhoRg) : 0.0;
      fy(i,j,k,4) = 0.0;
    });
  }

  // Conservative update
  const int nUp2 = use_dual_energy ? 5 : 4;
  for (MFIter mfi(Sborder); mfi.isValid(); ++mfi) {
    const Box& vbx = mfi.validbox();
    const auto& arr = Sborder.array(mfi);
    const auto& fx  = Fx_final.array(mfi);
    const auto& fy  = Fy_final.array(mfi);
    const Real cx = dt/dx[0], cy = dt/dx[1];
    ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      for (int n=0; n<nUp2; n++)
        arr(i,j,k,n) -= cx*(fx(i+1,j,k,n)-fx(i,j,k,n))
                      + cy*(fy(i,j+1,k,n)-fy(i,j,k,n));
    });
  }

  if (do_reflux) {
    Fx_final.mult(dt*dx[1], 0, NUM_STATE);
    MultiFab::Add(fluxes[0], Fx_final, 0, 0, NUM_STATE, 0);
    Fy_final.mult(dt*dx[0], 0, NUM_STATE);
    MultiFab::Add(fluxes[1], Fy_final, 0, 0, NUM_STATE, 0);
  }

#else
  // 3D: Strang splitting with the sequence x/2, y/2, z, y/2, x/2.
  {
    Vector<int>  sd3; Vector<Real> sdt3;
    sd3.push_back(0); sdt3.push_back(0.5*dt);
    sd3.push_back(1); sdt3.push_back(0.5*dt);
    sd3.push_back(2); sdt3.push_back(dt);
    sd3.push_back(1); sdt3.push_back(0.5*dt);
    sd3.push_back(0); sdt3.push_back(0.5*dt);
    for (int s=0; s<(int)sd3.size(); ++s) {
      const int d=sd3[s]; const Real dt_s=sdt3[s];
      const int iO=(d==0?1:0), jO=(d==1?1:0), kO=(d==2?1:0);
      BoxArray ba3=S_new.boxArray(); ba3.surroundingNodes(d);
      MultiFab ft3(ba3,dmap,NUM_STATE,0);
      for (MFIter mfi(ft3,true); mfi.isValid(); ++mfi) {
        const Box& bx=mfi.tilebox();
        const auto& arr=Sborder.array(mfi); const auto& fa=ft3.array(mfi);
        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
          Real F_rho,F_mx,F_my,F_E;
          muscl_hancock_flux(arr,i,j,k,dt_s,dx[d],F_rho,F_mx,F_my,F_E,d,2);
          fa(i,j,k,0)=F_rho; fa(i,j,k,1)=F_mx; fa(i,j,k,2)=F_my; fa(i,j,k,3)=F_E; fa(i,j,k,4)=0.0;
        });
      }
      const int nU3=use_dual_energy?5:4;
      for (MFIter mfi(Sborder); mfi.isValid(); ++mfi) {
        const Box& bx=mfi.tilebox();
        const auto& arr=Sborder.array(mfi); const auto& fa=ft3.array(mfi);
        ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
          for(int n=0;n<nU3;n++) arr(i,j,k,n)-=(dt_s/dx[d])*(fa(i+iO,j+jO,k+kO,n)-fa(i,j,k,n));
        });
      }
      Sborder.FillBoundary(geom.periodicity());
      if (do_reflux) {
        Real sf=dt_s; for(int sd=0;sd<amrex::SpaceDim;++sd){if(sd!=d)sf*=dx[sd];}
        ft3.mult(sf,0,NUM_STATE); MultiFab::Add(fluxes[d],ft3,0,0,NUM_STATE,0);
      }
    }
  }
#endif

  // Dual-energy correction: if kinetic energy dominates (ke > 0.99*E),
  // the internal energy e = E - ke is computed as a small difference of
  // large numbers and becomes inaccurate. In that regime we rebuild p
  // from the entropy proxy rhoS instead.
  if (use_dual_energy) {
    
    for (MFIter mfi(Sborder); mfi.isValid(); ++mfi)
    {
      const Box& bx = mfi.tilebox();
      const auto& arr = Sborder.array(mfi);

      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          Real rho  = arr(i,j,k,0);
          Real mx   = arr(i,j,k,1);
          Real my   = arr(i,j,k,2);
          Real E    = arr(i,j,k,3);
          Real rhoS = arr(i,j,k,4);

          if (rho < RhoVac) return;

          Real ke = 0.5 * (mx*mx + my*my) / rho;

          if (ke > KeSwitch * E) {
              Real S = rhoS / rho;
              Real p = S * pow_real(rho, GammaGas - 1.0);
              if (p < PMin) p = PMin;
              arr(i,j,k,3) = ke + p / (GammaGas - 1.0);
          }
      });
    }
  }

  if (use_dual_energy) {
    Real E_max_check = Sborder.max(3);
    Real E_min_check = Sborder.min(3);
    Real S_max = Sborder.max(4);
    Real S_min = Sborder.min(4);
    amrex::Print() << "E range: " << E_min_check << " to " << E_max_check
                   << " S range: " << S_min << " to " << S_max << "\n";
  }


  // Optional isentropic clamp for near-vacuum cells on fine AMR levels.
  // Enabled via prob.e_int_clamp = 1 in the inputs file.
  // Intended for Test 2 (double rarefaction) where AMR coarse-fine interface
  // errors are amplified by 1/rho in near-vacuum cells.
  {
    int e_clamp_flag = 0;
    amrex::ParmParse pp_clamp("prob");
    pp_clamp.query("e_int_clamp", e_clamp_flag);
    if (e_clamp_flag) {
      
      constexpr Real rho_thresh = 0.5;
      constexpr Real S0         = 2.0;
      constexpr Real margin     = 1.0;
      constexpr Real e_floor    = 1e-8;
      for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto& arr = Sborder.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
          Real rho = arr(i,j,k,0);
          if (rho < RhoVac || rho >= RhoClampThresh) return;
          Real mx = arr(i,j,k,1); Real my = arr(i,j,k,2);
          Real E  = arr(i,j,k,3);
          Real ke = 0.5*(mx*mx+my*my)/rho;
          Real e  = (E-ke)/rho;
          Real e_max = margin * S0 * std::pow(rho, GammaGas - 1.0);
          if (e > e_max)        arr(i,j,k,3) = ke + rho * e_max;
          else if (e < e_floor) arr(i,j,k,3) = ke + rho * e_floor;
        });
      }
    }
  }

  MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, 0);

  if (do_reflux) {
    if (current) {
      for (int i = 0; i < AMREX_SPACEDIM ; i++) {
        current->FineAdd(fluxes[i], i, 0, 0, NUM_STATE, 1.);
      }
    }
    if (fine) {
      for (int i = 0; i < AMREX_SPACEDIM ; i++) {
        fine->CrseInit(fluxes[i], i, 0, 0, NUM_STATE, -1.);
      }
    }
  }

  return dt;
}

Real
AmrLevelAdv::estTimeStep (Real)
{
  Real dt_est  = 1.0e+20;

  GpuArray<Real,BL_SPACEDIM> dx = geom.CellSizeArray();
  const MultiFab& S_new = get_new_data(Phi_Type);

  

  for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.validbox();
    const auto& arr = S_new.array(mfi);

    const Dim3 lo = lbound(bx);
    const Dim3 hi = ubound(bx);

    for (int k = lo.z; k <= hi.z; k++) {
    for (int j = lo.y; j <= hi.y; j++) {
    for (int i = lo.x; i <= hi.x; i++) {
        Real rho = arr(i,j,k,0);
        if (rho <= 0.0) rho = RhoMin;
        Real u   = arr(i,j,k,1) / rho;
        Real v   = arr(i,j,k,2) / rho;
        Real E   = arr(i,j,k,3);
        Real p   = (GammaGas - 1.0) * (E - 0.5*rho*(u*u + v*v));
        if (p <= 0.0) p = PMin;
        Real c   = std::sqrt(GammaGas * p / rho);

        Real dt_cell = dx[0] / (std::abs(u) + c);
        dt_est = std::min(dt_est, dt_cell);

        if (AMREX_SPACEDIM >= 2) {
            dt_cell = dx[1] / (std::abs(v) + c);
            dt_est = std::min(dt_est, dt_cell);
        }
    }}}\
  }

  ParallelDescriptor::ReduceRealMin(dt_est);
  dt_est *= cfl;

  if (verbose) {
      amrex::Print() << "AmrLevelAdv::estTimeStep at level " << level
                     << ":  dt_est = " << dt_est << std::endl;
  }

  return dt_est;
}

Real
AmrLevelAdv::initialTimeStep ()
{
  return estTimeStep(0.0);
}

void
AmrLevelAdv::computeInitialDt (int                   finest_level,
                               int                   /*sub_cycle*/,
                               Vector<int>&           n_cycle,
                               const Vector<IntVect>& /*ref_ratio*/,
                               Vector<Real>&          dt_level,
                               Real                  stop_time)
{
  if (level > 0) {
    return;
  }

  Real dt_0 = 1.0e+100;
  int n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    dt_level[i] = getLevel(i).initialTimeStep();
    n_factor   *= n_cycle[i];
    dt_0 = std::min(dt_0,n_factor*dt_level[i]);
  }

  const Real eps = 0.001*dt_0;
  Real cur_time  = state[Phi_Type].curTime();
  if (stop_time >= 0.0) {
    if ((cur_time + dt_0) > (stop_time - eps)) {
      dt_0 = stop_time - cur_time;
    }
  }

  n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0/n_factor;
  }
}

void
AmrLevelAdv::computeNewDt (int                   finest_level,
                           int                   /*sub_cycle*/,
                           Vector<int>&           n_cycle,
                           const Vector<IntVect>& /*ref_ratio*/,
                           Vector<Real>&          dt_min,
                           Vector<Real>&          dt_level,
                           Real                  stop_time,
                           int                   post_regrid_flag)
{
  if (level > 0) {
    return;
  }

  for (int i = 0; i <= finest_level; i++)
  {
    AmrLevelAdv& adv_level = getLevel(i);
    dt_min[i] = adv_level.estTimeStep(dt_level[i]);
  }

  if (post_regrid_flag == 1)
  {
    for (int i = 0; i <= finest_level; i++)
    {
      dt_min[i] = std::min(dt_min[i],dt_level[i]);
    }
  }
  else
  {
    static Real change_max = 1.1;
    for (int i = 0; i <= finest_level; i++)
    {
      dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
    }
  }

  Real dt_0 = 1.0e+100;
  int n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    n_factor *= n_cycle[i];
    dt_0 = std::min(dt_0,n_factor*dt_min[i]);
  }

  const Real eps = 0.001*dt_0;
  Real cur_time  = state[Phi_Type].curTime();
  if (stop_time >= 0.0) {
    if ((cur_time + dt_0) > (stop_time - eps)) {
      dt_0 = stop_time - cur_time;
    }
  }

  n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0/n_factor;
  }
}

void
AmrLevelAdv::post_timestep (int iteration)
{
  int finest_level = parent->finestLevel();

  if (do_reflux && level < finest_level) {
    reflux();
  }

  if (level < finest_level) {
    avgDown();
  }

  if (level < finest_level) {
    getLevel(level+1).resetFillPatcher();
  }

  amrex::ignore_unused(iteration);
}

void
AmrLevelAdv::post_regrid (int lbase, int /*new_finest*/) {
  amrex::ignore_unused(lbase);
}

void
AmrLevelAdv::post_restart()
{
}

void
AmrLevelAdv::post_init (Real /*stop_time*/)
{
  if (level > 0) {
    return;
  }

  int finest_level = parent->finestLevel();
  for (int k = finest_level-1; k>= 0; k--) {
    getLevel(k).avgDown();
  }
}

void
AmrLevelAdv::errorEst (TagBoxArray& tags,
                       int          /*clearval*/,
                       int          /*tagval*/,
                       Real         /*time*/,
                       int          /*n_error_buf*/,
                       int          /*ngrow*/)
{
  MultiFab& S_new = get_new_data(Phi_Type);

  const int oneGhost = 1;
  MultiFab phitmp;
  if (level < max_phigrad_lev)
  {
    const Real cur_time = state[Phi_Type].curTime();
    phitmp.define(S_new.boxArray(), S_new.DistributionMap(), NUM_STATE, 1);
    FillPatch(*this, phitmp, oneGhost, cur_time, Phi_Type, 0, NUM_STATE);
  }
  MultiFab const& phi = (level < max_phigrad_lev) ? phitmp : S_new;

  const char tagval = TagBox::SET;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  {
    for (MFIter mfi(phi,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& tilebx  = mfi.tilebox();
      const auto phiarr  = phi.array(mfi);
      auto       tagarr  = tags.array(mfi);

      if (level < max_phierr_lev) {
        const Real phierr_lev  = phierr[level];
        amrex::ParallelFor(tilebx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                           {
                             state_error(i, j, k, tagarr, phiarr, phierr_lev, tagval);
                           });
      }

      if (level < max_phigrad_lev) {
        const Real phigrad_lev = phigrad[level];
        amrex::ParallelFor(tilebx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                           {
                             grad_error(i, j, k, tagarr, phiarr, phigrad_lev, tagval);
                           });
      }

      // De-tag near-vacuum cells to prevent runaway refinement in Test 2
      // (double rarefaction), where rho->0 causes large internal energy
      // errors at coarse-fine interfaces.
      amrex::ParallelFor(tilebx,
                         [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                         {
                           if (phiarr(i,j,k,0) < RhoTagMin)
                             tagarr(i,j,k) = TagBox::CLEAR;
                         });
    }
  }
}

void
AmrLevelAdv::read_params ()
{
  static bool done = false;
  if (done) { return; }
  done = true;

  ParmParse pp("adv");

  pp.query("v",verbose);
  pp.query("cfl",cfl);
  pp.query("do_reflux",do_reflux);

  Geometry const* gg = AMReX::top()->getDefaultGeometry();

  if (! gg->IsCartesian()) {
    amrex::Abort("Please set geom.coord_sys = 0");
  }

  get_tagging_params();
}

void
AmrLevelAdv::reflux ()
{
  BL_ASSERT(level<parent->finestLevel());

  const auto strt = amrex::second();

  getFluxReg(level+1).Reflux(get_new_data(Phi_Type),1.0,0,0,NUM_STATE,geom);

  if (verbose)
  {
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    auto      end    = amrex::second() - strt;

    ParallelDescriptor::ReduceRealMax(end,IOProc);

    amrex::Print() << "AmrLevelAdv::reflux() at level " << level
                   << " : time = " << end << std::endl;
  }
}

void
AmrLevelAdv::avgDown ()
{
  if (level == parent->finestLevel()) { return; }
  avgDown(Phi_Type);
}

void
AmrLevelAdv::avgDown (int state_indx)
{
  if (level == parent->finestLevel()) { return; }

  AmrLevelAdv& fine_lev = getLevel(level+1);
  MultiFab&  S_fine   = fine_lev.get_new_data(state_indx);
  MultiFab&  S_crse   = get_new_data(state_indx);

  amrex::average_down(S_fine,S_crse,
                      fine_lev.geom,geom,
                      0,S_fine.nComp(),parent->refRatio(level));
}