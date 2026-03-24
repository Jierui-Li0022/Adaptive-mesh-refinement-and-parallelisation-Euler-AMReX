// MUSCL-Hancock reconstruction and flux computation for the 2D Euler equations.
// Adapted from a 1D ADER solver to 2D using dimension splitting (dir=0: x, dir=1: y).
//
// muscl_hancock_flux:     standard second-order flux at a single face
// muscl_hancock_flux_ctu: same, but with Colella 1990 transverse correction applied
//                         to the cell states before reconstruction (used in the CTU
//                         two-pass scheme in AmrLevelAdv::advance)

#ifndef RECONSTRUCTION_H_
#define RECONSTRUCTION_H_

#include "EulerEquations.H"
#include "RiemannSolver.H"
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Array4.H>
#include <cmath>

using amrex::Real;
using amrex::Array4;


// Slope limiters. All operate on the left and right undivided differences.

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real sgn(Real a) { return (a > 0) - (a < 0); }

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real minmod2(Real a, Real b) {
    if (a * b <= 0.0) return 0.0;
    return sgn(a) * amrex::min(std::abs(a), std::abs(b));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real minmod3(Real a, Real b, Real c) {
    return minmod2(a, minmod2(b, c));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real vanleer(Real a, Real b) {
    if (a * b <= 0.0) return 0.0;
    return (2.0 * a * b) / (a + b);
}

// MC limiter — less dissipative than minmod on smooth extrema.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real mc_limiter(Real a, Real b) {
    const Real t1 = 0.5 * (a + b);
    const Real t2 = 2.0 * a;
    const Real t3 = 2.0 * b;
    return minmod3(t1, t2, t3);
}

// limiter_type: 0 = minmod, 1 = van Leer, 2 = MC
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real limit_slope(Real dl, Real dr, int limiter_type = 0) {
    if (limiter_type == 0) return minmod2(dl, dr);
    if (limiter_type == 1) return vanleer(dl, dr);
    return mc_limiter(dl, dr);
}


// Clamp primitive variables after reconstruction.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_physical_prim(Real& rho, Real& u, Real& v, Real& p) {
    if (!std::isfinite(rho) || rho < RhoMin) rho = RhoMin;
    if (!std::isfinite(u)) u = 0.0;
    if (!std::isfinite(v)) v = 0.0;
    if (!std::isfinite(p) || p < PMin) p = PMin;
}


// Zhang-Shu positivity-preserving limiter (JCP 229, 2010, §2.1).
// Scales all slopes by theta in [0,1] to keep reconstructed density
// and pressure above ZsEps. Preserves second-order accuracy in smooth regions.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void zhang_shu_limit(Real rho, Real p,
                     Real& s_rho, Real& s_u, Real& s_v, Real& s_p)
{
    Real rho_L = rho - 0.5 * s_rho;
    Real rho_R = rho + 0.5 * s_rho;
    Real RhoMin = amrex::min(rho_L, rho_R);
    if (RhoMin < ZsEps) {
        Real theta = (rho - ZsEps) / (rho - RhoMin + 1e-30);
        theta = amrex::max(Real(0.0), amrex::min(Real(1.0), theta));
        s_rho *= theta; s_u *= theta; s_v *= theta; s_p *= theta;
    }

    Real p_L = p - 0.5 * s_p;
    Real p_R = p + 0.5 * s_p;
    Real PMin = amrex::min(p_L, p_R);
    if (PMin < ZsEps) {
        Real theta = (p - ZsEps) / (p - PMin + 1e-30);
        theta = amrex::max(Real(0.0), amrex::min(Real(1.0), theta));
        s_rho *= theta; s_u *= theta; s_v *= theta; s_p *= theta;
    }
}


// MUSCL-Hancock flux at face i+1/2 in direction dir.
//
// The face sits between cell L = (i-iOff, j-jOff, k) and cell R = (i, j, k).
// Steps:
//   1. Read a 4-cell stencil (LL, L, R, RR) and convert to primitives.
//   2. Compute limited slopes for L and R; apply Zhang-Shu and shock flattening.
//   3. Reconstruct interface states at the L/R face boundaries (Lp, Lm, Rm, Rp).
//   4. Half-step (Hancock predictor): evolve Lp and Rm by dt/2 using the
//      1D cell-averaged fluxes at the opposite face of each cell.
//   5. Riemann solve on the evolved (Lp, Rm) pair.
//
// Near-vacuum cells (rho or p below RhoNearVac) skip the half-step to avoid
// NaN in the predictor; this degrades to first-order in time locally but keeps
// the scheme stable.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void muscl_hancock_flux(const Array4<Real const>& arr,
                        int i, int j, int k,
                        Real dt, Real dx,
                        Real& F_rho, Real& F_mx, Real& F_my, Real& F_E,
                        int dir, int limiter_type = 0)
{
    const int iOff = (dir == 0) ? 1 : 0;
    const int jOff = (dir == 1) ? 1 : 0;

    const int iL = i - iOff,  jL = j - jOff;
    const int iR = i,          jR = j;
    const int iLL = iL - iOff, jLL = jL - jOff;
    const int iRR = iR + iOff, jRR = jR + jOff;

    Real rho_LL, u_LL, v_LL, p_LL;
    conserved_to_primitive(arr(iLL,jLL,k,URHO), arr(iLL,jLL,k,UMX),
                           arr(iLL,jLL,k,UMY),  arr(iLL,jLL,k,UEDEN),
                           rho_LL, u_LL, v_LL, p_LL);

    Real rho_L, u_L, v_L, p_L;
    conserved_to_primitive(arr(iL,jL,k,URHO), arr(iL,jL,k,UMX),
                           arr(iL,jL,k,UMY),  arr(iL,jL,k,UEDEN),
                           rho_L, u_L, v_L, p_L);

    Real rho_R, u_R, v_R, p_R;
    conserved_to_primitive(arr(iR,jR,k,URHO), arr(iR,jR,k,UMX),
                           arr(iR,jR,k,UMY),  arr(iR,jR,k,UEDEN),
                           rho_R, u_R, v_R, p_R);

    Real rho_RR, u_RR, v_RR, p_RR;
    conserved_to_primitive(arr(iRR,jRR,k,URHO), arr(iRR,jRR,k,UMX),
                           arr(iRR,jRR,k,UMY),  arr(iRR,jRR,k,UEDEN),
                           rho_RR, u_RR, v_RR, p_RR);

    Real slope_rho_L = limit_slope(rho_L - rho_LL, rho_R - rho_L, limiter_type);
    Real slope_u_L   = limit_slope(u_L   - u_LL,   u_R   - u_L,   limiter_type);
    Real slope_v_L   = limit_slope(v_L   - v_LL,   v_R   - v_L,   limiter_type);
    Real slope_p_L   = limit_slope(p_L   - p_LL,   p_R   - p_L,   limiter_type);

    Real slope_rho_R = limit_slope(rho_R - rho_L,  rho_RR - rho_R, limiter_type);
    Real slope_u_R   = limit_slope(u_R   - u_L,    u_RR   - u_R,   limiter_type);
    Real slope_v_R   = limit_slope(v_R   - v_L,    v_RR   - v_R,   limiter_type);
    Real slope_p_R   = limit_slope(p_R   - p_L,    p_RR   - p_R,   limiter_type);

    zhang_shu_limit(rho_L, p_L, slope_rho_L, slope_u_L, slope_v_L, slope_p_L);
    zhang_shu_limit(rho_R, p_R, slope_rho_R, slope_u_R, slope_v_R, slope_p_R);

    // Flatten slopes near strong shocks in compression to suppress post-shock
    // oscillations. Only applied when flow is actually converging.
    Real p_max = amrex::max(p_LL, amrex::max(p_L, amrex::max(p_R, p_RR)));
    Real p_min_val = amrex::min(p_LL, amrex::min(p_L, amrex::min(p_R, p_RR)));
    Real vel_L = (dir == 0) ? u_L : v_L;
    Real vel_R = (dir == 0) ? u_R : v_R;

    if (p_max > ShockFlatRatio * p_min_val + 1e-14 && (vel_R - vel_L) < 0.0) {
        slope_rho_L = 0.0; slope_u_L = 0.0; slope_v_L = 0.0; slope_p_L = 0.0;
        slope_rho_R = 0.0; slope_u_R = 0.0; slope_v_R = 0.0; slope_p_R = 0.0;
    }

    Real rho_Lp = rho_L + 0.5 * slope_rho_L;
    Real u_Lp   = u_L   + 0.5 * slope_u_L;
    Real v_Lp   = v_L   + 0.5 * slope_v_L;
    Real p_Lp   = p_L   + 0.5 * slope_p_L;
    enforce_physical_prim(rho_Lp, u_Lp, v_Lp, p_Lp);

    Real rho_Lm = rho_L - 0.5 * slope_rho_L;
    Real u_Lm   = u_L   - 0.5 * slope_u_L;
    Real v_Lm   = v_L   - 0.5 * slope_v_L;
    Real p_Lm   = p_L   - 0.5 * slope_p_L;
    enforce_physical_prim(rho_Lm, u_Lm, v_Lm, p_Lm);

    Real rho_Rm = rho_R - 0.5 * slope_rho_R;
    Real u_Rm   = u_R   - 0.5 * slope_u_R;
    Real v_Rm   = v_R   - 0.5 * slope_v_R;
    Real p_Rm   = p_R   - 0.5 * slope_p_R;
    enforce_physical_prim(rho_Rm, u_Rm, v_Rm, p_Rm);

    Real rho_Rp = rho_R + 0.5 * slope_rho_R;
    Real u_Rp   = u_R   + 0.5 * slope_u_R;
    Real v_Rp   = v_R   + 0.5 * slope_v_R;
    Real p_Rp   = p_R   + 0.5 * slope_p_R;
    enforce_physical_prim(rho_Rp, u_Rp, v_Rp, p_Rp);

    Real rho_face_min = amrex::min(rho_Lp, amrex::min(rho_Rm,
                        amrex::min(rho_Lm, rho_Rp)));
    Real p_face_min   = amrex::min(p_Lp,   amrex::min(p_Rm,
                        amrex::min(p_Lm,   p_Rp)));
    if (rho_face_min < RhoNearVac || p_face_min < RhoNearVac) {
        Real rho_Lp_c, mx_Lp_c, my_Lp_c, E_Lp_c;
        primitive_to_conserved(rho_Lp, u_Lp, v_Lp, p_Lp,
                               rho_Lp_c, mx_Lp_c, my_Lp_c, E_Lp_c);
        Real rho_Rm_c, mx_Rm_c, my_Rm_c, E_Rm_c;
        primitive_to_conserved(rho_Rm, u_Rm, v_Rm, p_Rm,
                               rho_Rm_c, mx_Rm_c, my_Rm_c, E_Rm_c);
        hllc_flux(rho_Lp_c, mx_Lp_c, my_Lp_c, E_Lp_c,
                  rho_Rm_c, mx_Rm_c, my_Rm_c, E_Rm_c,
                  F_rho, F_mx, F_my, F_E, dir);
        return;
    }

    // Half-step for cell L: evolve the Lp state using the flux divergence
    // across cell L (between its Lm and Lp boundaries).
    Real rho_Lp_c, mx_Lp, my_Lp, E_Lp;
    primitive_to_conserved(rho_Lp, u_Lp, v_Lp, p_Lp, rho_Lp_c, mx_Lp, my_Lp, E_Lp);

    Real rho_Lm_c, mx_Lm, my_Lm, E_Lm;
    primitive_to_conserved(rho_Lm, u_Lm, v_Lm, p_Lm, rho_Lm_c, mx_Lm, my_Lm, E_Lm);

    Real FL_rho, FL_mn, FL_mt, FL_E;
    Real FR_rho, FR_mn, FR_mt, FR_E;

    if (dir == 0) {
        const Real EL_total = p_Lm / (GammaGas - 1.0) + 0.5 * rho_Lm * (u_Lm * u_Lm + v_Lm * v_Lm);
        const Real ER_total = p_Lp / (GammaGas - 1.0) + 0.5 * rho_Lp * (u_Lp * u_Lp + v_Lp * v_Lp);
        FL_rho = rho_Lm * u_Lm;  FL_mn = rho_Lm * u_Lm * u_Lm + p_Lm;
        FL_mt  = rho_Lm * v_Lm * u_Lm;  FL_E = (EL_total + p_Lm) * u_Lm;
        FR_rho = rho_Lp * u_Lp;  FR_mn = rho_Lp * u_Lp * u_Lp + p_Lp;
        FR_mt  = rho_Lp * v_Lp * u_Lp;  FR_E = (ER_total + p_Lp) * u_Lp;
    } else {
        const Real EL_total = p_Lm / (GammaGas - 1.0) + 0.5 * rho_Lm * (u_Lm * u_Lm + v_Lm * v_Lm);
        const Real ER_total = p_Lp / (GammaGas - 1.0) + 0.5 * rho_Lp * (u_Lp * u_Lp + v_Lp * v_Lp);
        FL_rho = rho_Lm * v_Lm;  FL_mn = rho_Lm * v_Lm * v_Lm + p_Lm;
        FL_mt  = rho_Lm * u_Lm * v_Lm;  FL_E = (EL_total + p_Lm) * v_Lm;
        FR_rho = rho_Lp * v_Lp;  FR_mn = rho_Lp * v_Lp * v_Lp + p_Lp;
        FR_mt  = rho_Lp * u_Lp * v_Lp;  FR_E = (ER_total + p_Lp) * v_Lp;
    }

    const Real half_dt_dx = 0.5 * dt / dx;
    Real dU_rho = -half_dt_dx * (FR_rho - FL_rho);
    Real dU_mn  = -half_dt_dx * (FR_mn  - FL_mn);
    Real dU_mt  = -half_dt_dx * (FR_mt  - FL_mt);
    Real dU_E   = -half_dt_dx * (FR_E   - FL_E);

    rho_Lp_c += dU_rho;
    mx_Lp    += (dir == 0) ? dU_mn : dU_mt;
    my_Lp    += (dir == 0) ? dU_mt : dU_mn;
    E_Lp     += dU_E;
    enforce_physical(rho_Lp_c, mx_Lp, my_Lp, E_Lp);

    // Half-step for cell R.
    Real rho_Rm_c, mx_Rm, my_Rm, E_Rm;
    primitive_to_conserved(rho_Rm, u_Rm, v_Rm, p_Rm, rho_Rm_c, mx_Rm, my_Rm, E_Rm);

    Real rho_Rp_c, mx_Rp, my_Rp, E_Rp;
    primitive_to_conserved(rho_Rp, u_Rp, v_Rp, p_Rp, rho_Rp_c, mx_Rp, my_Rp, E_Rp);

    Real FL2_rho, FL2_mn, FL2_mt, FL2_E;
    Real FR2_rho, FR2_mn, FR2_mt, FR2_E;

    if (dir == 0) {
        const Real EL2_total = p_Rm / (GammaGas - 1.0) + 0.5 * rho_Rm * (u_Rm * u_Rm + v_Rm * v_Rm);
        const Real ER2_total = p_Rp / (GammaGas - 1.0) + 0.5 * rho_Rp * (u_Rp * u_Rp + v_Rp * v_Rp);
        FL2_rho = rho_Rm * u_Rm;  FL2_mn = rho_Rm * u_Rm * u_Rm + p_Rm;
        FL2_mt  = rho_Rm * v_Rm * u_Rm;  FL2_E = (EL2_total + p_Rm) * u_Rm;
        FR2_rho = rho_Rp * u_Rp;  FR2_mn = rho_Rp * u_Rp * u_Rp + p_Rp;
        FR2_mt  = rho_Rp * v_Rp * u_Rp;  FR2_E = (ER2_total + p_Rp) * u_Rp;
    } else {
        const Real EL2_total = p_Rm / (GammaGas - 1.0) + 0.5 * rho_Rm * (u_Rm * u_Rm + v_Rm * v_Rm);
        const Real ER2_total = p_Rp / (GammaGas - 1.0) + 0.5 * rho_Rp * (u_Rp * u_Rp + v_Rp * v_Rp);
        FL2_rho = rho_Rm * v_Rm;  FL2_mn = rho_Rm * v_Rm * v_Rm + p_Rm;
        FL2_mt  = rho_Rm * u_Rm * v_Rm;  FL2_E = (EL2_total + p_Rm) * v_Rm;
        FR2_rho = rho_Rp * v_Rp;  FR2_mn = rho_Rp * v_Rp * v_Rp + p_Rp;
        FR2_mt  = rho_Rp * u_Rp * v_Rp;  FR2_E = (ER2_total + p_Rp) * v_Rp;
    }

    Real dU2_rho = -half_dt_dx * (FR2_rho - FL2_rho);
    Real dU2_mn  = -half_dt_dx * (FR2_mn  - FL2_mn);
    Real dU2_mt  = -half_dt_dx * (FR2_mt  - FL2_mt);
    Real dU2_E   = -half_dt_dx * (FR2_E   - FL2_E);

    rho_Rm_c += dU2_rho;
    mx_Rm    += (dir == 0) ? dU2_mn : dU2_mt;
    my_Rm    += (dir == 0) ? dU2_mt : dU2_mn;
    E_Rm     += dU2_E;
    enforce_physical(rho_Rm_c, mx_Rm, my_Rm, E_Rm);

    // If the half-step produced a negative pressure (can happen at very strong
    // shocks), revert to the pre-evolution reconstructed states.
    Real p_Lp_check = (GammaGas - 1.0) * (E_Lp - 0.5*(mx_Lp*mx_Lp + my_Lp*my_Lp)/rho_Lp_c);
    Real p_Rm_check = (GammaGas - 1.0) * (E_Rm - 0.5*(mx_Rm*mx_Rm + my_Rm*my_Rm)/rho_Rm_c);
    if (p_Lp_check < PHalfstepMin || p_Rm_check < PHalfstepMin ||
        rho_Lp_c < PHalfstepMin || rho_Rm_c < PHalfstepMin) {
        primitive_to_conserved(rho_Lp, u_Lp, v_Lp, p_Lp,
                               rho_Lp_c, mx_Lp, my_Lp, E_Lp);
        primitive_to_conserved(rho_Rm, u_Rm, v_Rm, p_Rm,
                               rho_Rm_c, mx_Rm, my_Rm, E_Rm);
    }

    hllc_flux(rho_Lp_c, mx_Lp, my_Lp, E_Lp,
              rho_Rm_c, mx_Rm, my_Rm, E_Rm,
              F_rho, F_mx, F_my, F_E, dir);
}


// First-order Godunov flux — no reconstruction, just HLLC on cell averages.
// Kept for testing.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void godunov_flux(const Array4<Real const>& arr,
                  int i, int j, int k,
                  Real& F_rho, Real& F_mx, Real& F_my, Real& F_E,
                  int dir)
{
    const int iOff = (dir == 0) ? 1 : 0;
    const int jOff = (dir == 1) ? 1 : 0;
    const int iL = i - iOff;
    const int jL = j - jOff;

    hllc_flux(arr(iL,jL,k,URHO), arr(iL,jL,k,UMX), arr(iL,jL,k,UMY), arr(iL,jL,k,UEDEN),
              arr(i, j, k,URHO), arr(i, j, k,UMX), arr(i, j, k,UMY), arr(i, j, k,UEDEN),
              F_rho, F_mx, F_my, F_E, dir);
}


// MUSCL-Hancock with CTU transverse correction (Colella 1990).
//
// Before reconstruction, each cell's conserved state is corrected by
// half a timestep of the transverse flux divergence:
//   U_corrected = U - coeff_trans * (F_trans(upper) - F_trans(lower))
// where coeff_trans = dt/(2*dx_trans).
//
// This eliminates the Strang splitting commutator error [Lx, Ly], which
// is what causes the diagonal-problem accuracy degradation without CTU.
// trans_flux uses AMReX face indexing: trans_flux(i,j,k,n) is the flux
// at the lower face of cell (i,j,k) in the transverse direction.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void muscl_hancock_flux_ctu(const Array4<Real const>& arr,
                             const Array4<Real const>& trans_flux,
                             int i, int j, int k,
                             Real dt, Real dx,
                             Real coeff_trans,
                             Real& F_rho, Real& F_mx, Real& F_my, Real& F_E,
                             int dir, int limiter_type = 0)
{
    const int iOff = (dir == 0) ? 1 : 0;
    const int jOff = (dir == 1) ? 1 : 0;

    const int iL = i - iOff;  const int jL = j - jOff;
    const int iR = i;          const int jR = j;

    // Transverse face offsets (perpendicular to dir).
    // dir=0: transverse is y, upper face is (i, j+1)
    // dir=1: transverse is x, upper face is (i+1, j)
    const int iToff = (dir == 1) ? 1 : 0;
    const int jToff = (dir == 0) ? 1 : 0;

    Real rhoL_c = arr(iL,jL,k,URHO)
                  - coeff_trans * (trans_flux(iL+iToff, jL+jToff, k, URHO)
                                 - trans_flux(iL,       jL,        k, URHO));
    Real mxL_c  = arr(iL,jL,k,UMX)
                  - coeff_trans * (trans_flux(iL+iToff, jL+jToff, k, UMX)
                                 - trans_flux(iL,       jL,        k, UMX));
    Real myL_c  = arr(iL,jL,k,UMY)
                  - coeff_trans * (trans_flux(iL+iToff, jL+jToff, k, UMY)
                                 - trans_flux(iL,       jL,        k, UMY));
    Real EL_c   = arr(iL,jL,k,UEDEN)
                  - coeff_trans * (trans_flux(iL+iToff, jL+jToff, k, UEDEN)
                                 - trans_flux(iL,       jL,        k, UEDEN));

    Real rhoR_c = arr(iR,jR,k,URHO)
                  - coeff_trans * (trans_flux(iR+iToff, jR+jToff, k, URHO)
                                 - trans_flux(iR,       jR,        k, URHO));
    Real mxR_c  = arr(iR,jR,k,UMX)
                  - coeff_trans * (trans_flux(iR+iToff, jR+jToff, k, UMX)
                                 - trans_flux(iR,       jR,        k, UMX));
    Real myR_c  = arr(iR,jR,k,UMY)
                  - coeff_trans * (trans_flux(iR+iToff, jR+jToff, k, UMY)
                                 - trans_flux(iR,       jR,        k, UMY));
    Real ER_c   = arr(iR,jR,k,UEDEN)
                  - coeff_trans * (trans_flux(iR+iToff, jR+jToff, k, UEDEN)
                                 - trans_flux(iR,       jR,        k, UEDEN));

    rhoL_c = amrex::max(rhoL_c, RhoMin);
    rhoR_c = amrex::max(rhoR_c, RhoMin);
    EL_c   = amrex::max(EL_c,   PMin);
    ER_c   = amrex::max(ER_c,   PMin);

    // From here the logic is identical to muscl_hancock_flux, but reading
    // from the transverse-corrected L/R states instead of arr directly.

    const int iLL = iL - iOff;  const int jLL = jL - jOff;
    const int iRR = iR + iOff;  const int jRR = jR + jOff;

    Real rho_L, u_L, v_L, p_L;
    conserved_to_primitive(rhoL_c, mxL_c, myL_c, EL_c, rho_L, u_L, v_L, p_L);

    Real rho_R, u_R, v_R, p_R;
    conserved_to_primitive(rhoR_c, mxR_c, myR_c, ER_c, rho_R, u_R, v_R, p_R);

    Real rho_LL, u_LL, v_LL, p_LL;
    conserved_to_primitive(arr(iLL,jLL,k,URHO), arr(iLL,jLL,k,UMX),
                           arr(iLL,jLL,k,UMY),  arr(iLL,jLL,k,UEDEN),
                           rho_LL, u_LL, v_LL, p_LL);

    Real rho_RR, u_RR, v_RR, p_RR;
    conserved_to_primitive(arr(iRR,jRR,k,URHO), arr(iRR,jRR,k,UMX),
                           arr(iRR,jRR,k,UMY),  arr(iRR,jRR,k,UEDEN),
                           rho_RR, u_RR, v_RR, p_RR);

    Real slope_rho_L = limit_slope(rho_L - rho_LL, rho_R - rho_L, limiter_type);
    Real slope_u_L   = limit_slope(u_L   - u_LL,   u_R   - u_L,   limiter_type);
    Real slope_v_L   = limit_slope(v_L   - v_LL,   v_R   - v_L,   limiter_type);
    Real slope_p_L   = limit_slope(p_L   - p_LL,   p_R   - p_L,   limiter_type);

    Real slope_rho_R = limit_slope(rho_R - rho_L,  rho_RR - rho_R, limiter_type);
    Real slope_u_R   = limit_slope(u_R   - u_L,    u_RR   - u_R,   limiter_type);
    Real slope_v_R   = limit_slope(v_R   - v_L,    v_RR   - v_R,   limiter_type);
    Real slope_p_R   = limit_slope(p_R   - p_L,    p_RR   - p_R,   limiter_type);

    zhang_shu_limit(rho_L, p_L, slope_rho_L, slope_u_L, slope_v_L, slope_p_L);
    zhang_shu_limit(rho_R, p_R, slope_rho_R, slope_u_R, slope_v_R, slope_p_R);

    Real p_max = amrex::max(p_LL, amrex::max(p_L, amrex::max(p_R, p_RR)));
    Real p_min_val = amrex::min(p_LL, amrex::min(p_L, amrex::min(p_R, p_RR)));
    Real vel_L = (dir == 0) ? u_L : v_L;
    Real vel_R = (dir == 0) ? u_R : v_R;
    if (p_max > ShockFlatRatio * p_min_val + 1e-14 && (vel_R - vel_L) < 0.0) {
        slope_rho_L=0; slope_u_L=0; slope_v_L=0; slope_p_L=0;
        slope_rho_R=0; slope_u_R=0; slope_v_R=0; slope_p_R=0;
    }

    Real rho_Lp = rho_L + 0.5*slope_rho_L; Real u_Lp = u_L + 0.5*slope_u_L;
    Real v_Lp   = v_L   + 0.5*slope_v_L;   Real p_Lp = p_L + 0.5*slope_p_L;
    enforce_physical_prim(rho_Lp, u_Lp, v_Lp, p_Lp);

    Real rho_Lm = rho_L - 0.5*slope_rho_L; Real u_Lm = u_L - 0.5*slope_u_L;
    Real v_Lm   = v_L   - 0.5*slope_v_L;   Real p_Lm = p_L - 0.5*slope_p_L;
    enforce_physical_prim(rho_Lm, u_Lm, v_Lm, p_Lm);

    Real rho_Rm = rho_R - 0.5*slope_rho_R; Real u_Rm = u_R - 0.5*slope_u_R;
    Real v_Rm   = v_R   - 0.5*slope_v_R;   Real p_Rm = p_R - 0.5*slope_p_R;
    enforce_physical_prim(rho_Rm, u_Rm, v_Rm, p_Rm);

    Real rho_Rp = rho_R + 0.5*slope_rho_R; Real u_Rp = u_R + 0.5*slope_u_R;
    Real v_Rp   = v_R   + 0.5*slope_v_R;   Real p_Rp = p_R + 0.5*slope_p_R;
    enforce_physical_prim(rho_Rp, u_Rp, v_Rp, p_Rp);

    Real rho_face_min = amrex::min(rho_Lp, amrex::min(rho_Rm,
                        amrex::min(rho_Lm, rho_Rp)));
    Real p_face_min   = amrex::min(p_Lp,   amrex::min(p_Rm,
                        amrex::min(p_Lm,   p_Rp)));
    if (rho_face_min < RhoNearVac || p_face_min < RhoNearVac) {
        Real rho_Lp_c, mx_Lp_c, my_Lp_c, E_Lp_c;
        primitive_to_conserved(rho_Lp, u_Lp, v_Lp, p_Lp,
                               rho_Lp_c, mx_Lp_c, my_Lp_c, E_Lp_c);
        Real rho_Rm_c, mx_Rm_c, my_Rm_c, E_Rm_c;
        primitive_to_conserved(rho_Rm, u_Rm, v_Rm, p_Rm,
                               rho_Rm_c, mx_Rm_c, my_Rm_c, E_Rm_c);
        hllc_flux(rho_Lp_c, mx_Lp_c, my_Lp_c, E_Lp_c,
                  rho_Rm_c, mx_Rm_c, my_Rm_c, E_Rm_c,
                  F_rho, F_mx, F_my, F_E, dir);
        return;
    }

    // Half-step for cell L
    Real rho_Lp_c, mx_Lp, my_Lp, E_Lp;
    primitive_to_conserved(rho_Lp, u_Lp, v_Lp, p_Lp, rho_Lp_c, mx_Lp, my_Lp, E_Lp);
    Real rho_Lm_c, mx_Lm, my_Lm, E_Lm;
    primitive_to_conserved(rho_Lm, u_Lm, v_Lm, p_Lm, rho_Lm_c, mx_Lm, my_Lm, E_Lm);

    Real FL_rho, FL_mn, FL_mt, FL_E, FR_rho, FR_mn, FR_mt, FR_E;
    if (dir == 0) {
        const Real ELt = p_Lm/(GammaGas-1.0) + 0.5*rho_Lm*(u_Lm*u_Lm+v_Lm*v_Lm);
        const Real ERt = p_Lp/(GammaGas-1.0) + 0.5*rho_Lp*(u_Lp*u_Lp+v_Lp*v_Lp);
        FL_rho=rho_Lm*u_Lm; FL_mn=rho_Lm*u_Lm*u_Lm+p_Lm; FL_mt=rho_Lm*v_Lm*u_Lm; FL_E=(ELt+p_Lm)*u_Lm;
        FR_rho=rho_Lp*u_Lp; FR_mn=rho_Lp*u_Lp*u_Lp+p_Lp; FR_mt=rho_Lp*v_Lp*u_Lp; FR_E=(ERt+p_Lp)*u_Lp;
    } else {
        const Real ELt = p_Lm/(GammaGas-1.0) + 0.5*rho_Lm*(u_Lm*u_Lm+v_Lm*v_Lm);
        const Real ERt = p_Lp/(GammaGas-1.0) + 0.5*rho_Lp*(u_Lp*u_Lp+v_Lp*v_Lp);
        FL_rho=rho_Lm*v_Lm; FL_mn=rho_Lm*v_Lm*v_Lm+p_Lm; FL_mt=rho_Lm*u_Lm*v_Lm; FL_E=(ELt+p_Lm)*v_Lm;
        FR_rho=rho_Lp*v_Lp; FR_mn=rho_Lp*v_Lp*v_Lp+p_Lp; FR_mt=rho_Lp*u_Lp*v_Lp; FR_E=(ERt+p_Lp)*v_Lp;
    }
    const Real hdx = 0.5*dt/dx;
    rho_Lp_c += -hdx*(FR_rho-FL_rho);
    mx_Lp    += -hdx*((dir==0)?(FR_mn-FL_mn):(FR_mt-FL_mt));
    my_Lp    += -hdx*((dir==0)?(FR_mt-FL_mt):(FR_mn-FL_mn));
    E_Lp     += -hdx*(FR_E-FL_E);
    enforce_physical(rho_Lp_c, mx_Lp, my_Lp, E_Lp);

    // Half-step for cell R
    Real rho_Rm_c, mx_Rm, my_Rm, E_Rm;
    primitive_to_conserved(rho_Rm, u_Rm, v_Rm, p_Rm, rho_Rm_c, mx_Rm, my_Rm, E_Rm);
    Real rho_Rp_c, mx_Rp, my_Rp, E_Rp;
    primitive_to_conserved(rho_Rp, u_Rp, v_Rp, p_Rp, rho_Rp_c, mx_Rp, my_Rp, E_Rp);

    Real FL2_rho, FL2_mn, FL2_mt, FL2_E, FR2_rho, FR2_mn, FR2_mt, FR2_E;
    if (dir == 0) {
        const Real ELt = p_Rm/(GammaGas-1.0)+0.5*rho_Rm*(u_Rm*u_Rm+v_Rm*v_Rm);
        const Real ERt = p_Rp/(GammaGas-1.0)+0.5*rho_Rp*(u_Rp*u_Rp+v_Rp*v_Rp);
        FL2_rho=rho_Rm*u_Rm; FL2_mn=rho_Rm*u_Rm*u_Rm+p_Rm; FL2_mt=rho_Rm*v_Rm*u_Rm; FL2_E=(ELt+p_Rm)*u_Rm;
        FR2_rho=rho_Rp*u_Rp; FR2_mn=rho_Rp*u_Rp*u_Rp+p_Rp; FR2_mt=rho_Rp*v_Rp*u_Rp; FR2_E=(ERt+p_Rp)*u_Rp;
    } else {
        const Real ELt = p_Rm/(GammaGas-1.0)+0.5*rho_Rm*(u_Rm*u_Rm+v_Rm*v_Rm);
        const Real ERt = p_Rp/(GammaGas-1.0)+0.5*rho_Rp*(u_Rp*u_Rp+v_Rp*v_Rp);
        FL2_rho=rho_Rm*v_Rm; FL2_mn=rho_Rm*v_Rm*v_Rm+p_Rm; FL2_mt=rho_Rm*u_Rm*v_Rm; FL2_E=(ELt+p_Rm)*v_Rm;
        FR2_rho=rho_Rp*v_Rp; FR2_mn=rho_Rp*v_Rp*v_Rp+p_Rp; FR2_mt=rho_Rp*u_Rp*v_Rp; FR2_E=(ERt+p_Rp)*v_Rp;
    }
    rho_Rm_c += -hdx*(FR2_rho-FL2_rho);
    mx_Rm    += -hdx*((dir==0)?(FR2_mn-FL2_mn):(FR2_mt-FL2_mt));
    my_Rm    += -hdx*((dir==0)?(FR2_mt-FL2_mt):(FR2_mn-FL2_mn));
    E_Rm     += -hdx*(FR2_E-FL2_E);
    enforce_physical(rho_Rm_c, mx_Rm, my_Rm, E_Rm);

    Real p_Lp_check = (GammaGas-1.0)*(E_Lp - 0.5*(mx_Lp*mx_Lp+my_Lp*my_Lp)/rho_Lp_c);
    Real p_Rm_check = (GammaGas-1.0)*(E_Rm - 0.5*(mx_Rm*mx_Rm+my_Rm*my_Rm)/rho_Rm_c);
    if (p_Lp_check < PHalfstepMin || p_Rm_check < PHalfstepMin ||
        rho_Lp_c < PHalfstepMin || rho_Rm_c < PHalfstepMin) {
        primitive_to_conserved(rho_Lp, u_Lp, v_Lp, p_Lp, rho_Lp_c, mx_Lp, my_Lp, E_Lp);
        primitive_to_conserved(rho_Rm, u_Rm, v_Rm, p_Rm, rho_Rm_c, mx_Rm, my_Rm, E_Rm);
    }

    hllc_flux(rho_Lp_c, mx_Lp, my_Lp, E_Lp,
              rho_Rm_c, mx_Rm, my_Rm, E_Rm,
              F_rho, F_mx, F_my, F_E, dir);
}

#endif