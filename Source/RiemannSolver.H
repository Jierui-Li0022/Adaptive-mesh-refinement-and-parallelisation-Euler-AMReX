// Approximate and exact Riemann solvers for the 2D Euler equations.
// All solvers use dimension splitting: dir=0 solves in x, dir=1 in y.
// HLLC is the default solver used by the MUSCL-Hancock scheme.
// HLL and the exact solver are kept for testing and comparison.

#ifndef RIEMANN_SOLVER_H_
#define RIEMANN_SOLVER_H_

#include "EulerEquations.H"
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>
#include <algorithm>

using amrex::Real;


// Lightweight pressure and sound speed helpers used in the hot path.
// These skip the full clamping logic in pressure() for speed; the caller
// is expected to have already guarded against negative density.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real fast_pressure(Real rho, Real mx, Real my, Real E) {
    const Real rho_safe = (rho > RhoMin) ? rho : RhoMin;
    Real p = (GammaGas - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho_safe);
    return (p > PMin) ? p : PMin;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real fast_sound_speed(Real rho, Real p) {
    return std::sqrt(GammaGas * p / rho);
}


// Decompose (mx, my) into normal and tangential components for direction dir.
// dir=0: normal = x, tangential = y
// dir=1: normal = y, tangential = x
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void get_normal_tangential(Real mx, Real my, int dir,
                           Real& mn, Real& mt) {
    if (dir == 0) { mn = mx; mt = my; }
    else          { mn = my; mt = mx; }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void set_normal_tangential(Real mn, Real mt, int dir,
                           Real& mx, Real& my) {
    if (dir == 0) { mx = mn; my = mt; }
    else          { mx = mt; my = mn; }
}


// HLLC approximate Riemann solver (Toro et al. 1994).
// Wave speeds estimated from the Roe-averaged eigenvalues.
// The contact wave speed sM is resolved explicitly, giving better
// resolution of contact discontinuities than plain HLL.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hllc_flux(Real rhoL, Real mxL, Real myL, Real EL,
               Real rhoR, Real mxR, Real myR, Real ER,
               Real& F_rho, Real& F_mx, Real& F_my, Real& F_E,
               int dir)
{
    Real mnL, mtL, mnR, mtR;
    get_normal_tangential(mxL, myL, dir, mnL, mtL);
    get_normal_tangential(mxR, myR, dir, mnR, mtR);

    const Real rhoL_s = (rhoL > RhoMin) ? rhoL : RhoMin;
    const Real rhoR_s = (rhoR > RhoMin) ? rhoR : RhoMin;
    const Real unL = mnL / rhoL_s;
    const Real unR = mnR / rhoR_s;
    const Real utL = mtL / rhoL_s;
    const Real utR = mtR / rhoR_s;
    const Real pL  = fast_pressure(rhoL_s, mxL, myL, EL);
    const Real pR  = fast_pressure(rhoR_s, mxR, myR, ER);
    const Real aL  = fast_sound_speed(rhoL_s, pL);
    const Real aR  = fast_sound_speed(rhoR_s, pR);

    const Real sL = amrex::min(unL - aL, unR - aR);
    const Real sR = amrex::max(unL + aL, unR + aR);

    // Compute normal fluxes on each side directly from primitives,
    // avoiding a second pressure evaluation.
    const Real fL_rho = mnL;
    const Real fL_mn  = mnL * unL + pL;
    const Real fL_mt  = mtL * unL;
    const Real fL_E   = (EL + pL) * unL;

    const Real fR_rho = mnR;
    const Real fR_mn  = mnR * unR + pR;
    const Real fR_mt  = mtR * unR;
    const Real fR_E   = (ER + pR) * unR;

    Real F_rho_n, F_mn, F_mt, F_E_n;

    if (sL >= 0.0) {
        F_rho_n = fL_rho; F_mn = fL_mn; F_mt = fL_mt; F_E_n = fL_E;
    }
    else if (sR <= 0.0) {
        F_rho_n = fR_rho; F_mn = fR_mn; F_mt = fR_mt; F_E_n = fR_E;
    }
    else {
        // Contact wave speed from the HLLC pressure condition
        const Real denom = rhoL_s * (sL - unL) - rhoR_s * (sR - unR);
        const Real sM = (std::abs(denom) < 1e-14)
            ? 0.5 * (unL + unR)
            : (pR - pL + rhoL_s * unL * (sL - unL) - rhoR_s * unR * (sR - unR)) / denom;

        if (sM >= 0.0) {
            const Real fac = rhoL_s * (sL - unL) / (sL - sM);
            const Real Us_rho = fac;
            const Real Us_mn  = fac * sM;
            const Real Us_mt  = fac * utL;
            const Real Us_E   = fac * (EL / rhoL_s + (sM - unL) * (sM + pL / (rhoL_s * (sL - unL))));

            F_rho_n = fL_rho + sL * (Us_rho - rhoL);
            F_mn    = fL_mn  + sL * (Us_mn  - mnL);
            F_mt    = fL_mt  + sL * (Us_mt  - mtL);
            F_E_n   = fL_E   + sL * (Us_E   - EL);
        } else {
            const Real fac = rhoR_s * (sR - unR) / (sR - sM);
            const Real Us_rho = fac;
            const Real Us_mn  = fac * sM;
            const Real Us_mt  = fac * utR;
            const Real Us_E   = fac * (ER / rhoR_s + (sM - unR) * (sM + pR / (rhoR_s * (sR - unR))));

            F_rho_n = fR_rho + sR * (Us_rho - rhoR);
            F_mn    = fR_mn  + sR * (Us_mn  - mnR);
            F_mt    = fR_mt  + sR * (Us_mt  - mtR);
            F_E_n   = fR_E   + sR * (Us_E   - ER);
        }
    }

    F_rho = F_rho_n;
    F_E   = F_E_n;
    set_normal_tangential(F_mn, F_mt, dir, F_mx, F_my);
}


// HLL solver — kept for comparison. Smears contacts more than HLLC
// but is slightly cheaper and unconditionally entropy-satisfying.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hll_flux(Real rhoL, Real mxL, Real myL, Real EL,
              Real rhoR, Real mxR, Real myR, Real ER,
              Real& F_rho, Real& F_mx, Real& F_my, Real& F_E,
              int dir)
{
    Real mnL, mtL, mnR, mtR;
    get_normal_tangential(mxL, myL, dir, mnL, mtL);
    get_normal_tangential(mxR, myR, dir, mnR, mtR);

    const Real rhoL_s = (rhoL > RhoMin) ? rhoL : RhoMin;
    const Real rhoR_s = (rhoR > RhoMin) ? rhoR : RhoMin;
    const Real unL = mnL / rhoL_s;
    const Real unR = mnR / rhoR_s;
    const Real pL  = fast_pressure(rhoL_s, mxL, myL, EL);
    const Real pR  = fast_pressure(rhoR_s, mxR, myR, ER);
    const Real aL  = fast_sound_speed(rhoL_s, pL);
    const Real aR  = fast_sound_speed(rhoR_s, pR);

    const Real sL = amrex::min(unL - aL, unR - aR);
    const Real sR = amrex::max(unL + aL, unR + aR);

    const Real fL_rho = mnL;
    const Real fL_mn  = mnL * unL + pL;
    const Real fL_mt  = mtL * unL;
    const Real fL_E   = (EL + pL) * unL;

    const Real fR_rho = mnR;
    const Real fR_mn  = mnR * unR + pR;
    const Real fR_mt  = mtR * unR;
    const Real fR_E   = (ER + pR) * unR;

    Real F_rho_n, F_mn, F_mt, F_E_n;

    if (sL >= 0.0) {
        F_rho_n = fL_rho; F_mn = fL_mn; F_mt = fL_mt; F_E_n = fL_E;
    }
    else if (sR <= 0.0) {
        F_rho_n = fR_rho; F_mn = fR_mn; F_mt = fR_mt; F_E_n = fR_E;
    }
    else {
        const Real inv_ds = 1.0 / (sR - sL);
        F_rho_n = (sR * fL_rho - sL * fR_rho + sL * sR * (rhoR - rhoL))  * inv_ds;
        F_mn    = (sR * fL_mn  - sL * fR_mn  + sL * sR * (mnR  - mnL))   * inv_ds;
        F_mt    = (sR * fL_mt  - sL * fR_mt  + sL * sR * (mtR  - mtL))   * inv_ds;
        F_E_n   = (sR * fL_E   - sL * fR_E   + sL * sR * (ER   - EL))    * inv_ds;
    }

    F_rho = F_rho_n;
    F_E   = F_E_n;
    set_normal_tangential(F_mn, F_mt, dir, F_mx, F_my);
}


// Helper functions for the exact solver.
// fk is the wave function for side k; its sign determines shock vs rarefaction.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real fk(Real p_star, Real v_k, Real rho_k, Real p_k) {
    if (!std::isfinite(p_star)) p_star = PMin;
    if (!std::isfinite(p_k) || p_k < PMin) p_k = PMin;
    if (!std::isfinite(rho_k) || rho_k < RhoMin) rho_k = RhoMin;

    const Real c_k = std::sqrt(GammaGas * p_k / rho_k);
    const Real A_k = 2.0 / ((GammaGas + 1.0) * rho_k);
    const Real B_k = (GammaGas - 1.0) / (GammaGas + 1.0) * p_k;

    if (p_star >= p_k) {
        return (p_star - p_k) * std::sqrt(A_k / (p_star + B_k));
    } else {
        return 2.0 * c_k / (GammaGas - 1.0) *
               (std::pow(p_star / p_k, (GammaGas - 1.0) / (2.0 * GammaGas)) - 1.0);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real dfk_dp(Real p_star, Real v_k, Real rho_k, Real p_k) {
    if (!std::isfinite(p_star)) p_star = PMin;
    if (!std::isfinite(p_k) || p_k < PMin) p_k = PMin;
    if (!std::isfinite(rho_k) || rho_k < RhoMin) rho_k = RhoMin;

    const Real c_k = std::sqrt(GammaGas * p_k / rho_k);
    const Real A_k = 2.0 / ((GammaGas + 1.0) * rho_k);
    const Real B_k = (GammaGas - 1.0) / (GammaGas + 1.0) * p_k;

    if (p_star >= p_k) {
        return std::sqrt(A_k / (p_star + B_k)) *
               (1.0 - 0.5 * (p_star - p_k) / (p_star + B_k));
    } else {
        return 1.0 / (rho_k * c_k) *
               std::pow(p_star / p_k, -(GammaGas + 1.0) / (2.0 * GammaGas));
    }
}

// Newton-Raphson iteration for the star-state pressure.
// Initial guess from the primitive-variable Riemann solver (PVRS).
// Iterations are clamped to [0.1*p_old, 10*p_old] to prevent divergence
// on strong shocks.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real solve_p_star(Real vL, Real vR, Real rhoL, Real pL, Real rhoR, Real pR) {
    if (!std::isfinite(pL) || pL < PMin) pL = PMin;
    if (!std::isfinite(pR) || pR < PMin) pR = PMin;
    if (!std::isfinite(rhoL) || rhoL < RhoMin) rhoL = RhoMin;
    if (!std::isfinite(rhoR) || rhoR < RhoMin) rhoR = RhoMin;

    const Real aL = std::sqrt(GammaGas * pL / rhoL);
    const Real aR = std::sqrt(GammaGas * pR / rhoR);
    Real p0 = 0.5 * (pL + pR) - 0.125 * (vR - vL) * (rhoL + rhoR) * (aL + aR);
    Real p_star = amrex::max(p0, PMin);
    if (p_star > PStarMax) p_star = PStarMax;

    for (int iter = 0; iter < NrMaxIter; ++iter) {
        const Real f  = fk(p_star, vL, rhoL, pL) + fk(p_star, vR, rhoR, pR) + (vR - vL);
        const Real df = dfk_dp(p_star, vL, rhoL, pL) + dfk_dp(p_star, vR, rhoR, pR);

        if (!std::isfinite(f) || !std::isfinite(df) || std::abs(df) < 1e-14)
            return p_star;

        const Real p_old = p_star;
        Real p_new = p_old - f / df;

        if (!std::isfinite(p_new)) return p_old;

        p_new = amrex::max(p_new, PMin);
        p_new = amrex::max(p_new, 0.1 * p_old);
        p_new = amrex::min(p_new, 10.0 * p_old);
        p_new = amrex::min(p_new, PStarMax);

        if (std::abs(p_new - p_old) <= 1e-8 * amrex::max(PMin, p_new))
            return p_new;

        p_star = p_new;
    }

    return p_star;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real calculate_v_star(Real p_star,
                      Real vL, Real rhoL, Real pL,
                      Real vR, Real rhoR, Real pR) {
    return 0.5 * (vL + vR)
         + 0.5 * (fk(p_star, vR, rhoR, pR) - fk(p_star, vL, rhoL, pL));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real rho_star_side(Real p_star, Real rho, Real p) {
    const Real pr = p_star / p;
    if (p_star > p) {
        const Real gm1 = GammaGas - 1.0;
        const Real gp1 = GammaGas + 1.0;
        return rho * ((pr + gm1 / gp1) / ((gm1 / gp1) * pr + 1.0));
    } else {
        return rho * std::pow(pr, 1.0 / GammaGas);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void exact_riemann_flux_1d(Real p_star, Real v_star, Real rho_star,
                           Real& F1, Real& F2, Real& F3) {
    const Real E = p_star / (GammaGas - 1.0) + 0.5 * rho_star * v_star * v_star;
    F1 = rho_star * v_star;
    F2 = rho_star * v_star * v_star + p_star;
    F3 = (E + p_star) * v_star;
}

// Exact Riemann solver. Falls back to HLLC if Newton-Raphson diverges
// or the initial states are unphysical. The tangential velocity is simply
// advected from the upwind side of the contact.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void exact_flux(Real rhoL, Real mxL, Real myL, Real EL,
                Real rhoR, Real mxR, Real myR, Real ER,
                Real& F_rho, Real& F_mx, Real& F_my, Real& F_E,
                int dir)
{
    Real mnL, mtL, mnR, mtR;
    get_normal_tangential(mxL, myL, dir, mnL, mtL);
    get_normal_tangential(mxR, myR, dir, mnR, mtR);

    const Real rhoL_s = clamp_rho(rhoL);
    const Real rhoR_s = clamp_rho(rhoR);
    const Real unL = mnL / rhoL_s;
    const Real unR = mnR / rhoR_s;
    const Real utL = mtL / rhoL_s;
    const Real utR = mtR / rhoR_s;
    const Real pL  = fast_pressure(rhoL_s, mxL, myL, EL);
    const Real pR  = fast_pressure(rhoR_s, mxR, myR, ER);

    auto bad = [](Real rho, Real p) {
        return (!std::isfinite(rho) || !std::isfinite(p) || rho <= RhoMin || p <= PMin);
    };

    if (bad(rhoL_s, pL) || bad(rhoR_s, pR)) {
        hllc_flux(rhoL, mxL, myL, EL, rhoR, mxR, myR, ER, F_rho, F_mx, F_my, F_E, dir);
        return;
    }

    Real p_star = solve_p_star(unL, unR, rhoL_s, pL, rhoR_s, pR);

    if (!std::isfinite(p_star) || p_star <= PMin || p_star > PStarMax) {
        hllc_flux(rhoL, mxL, myL, EL, rhoR, mxR, myR, ER, F_rho, F_mx, F_my, F_E, dir);
        return;
    }

    Real u_star = calculate_v_star(p_star, unL, rhoL_s, pL, unR, rhoR_s, pR);
    if (!std::isfinite(u_star)) {
        hllc_flux(rhoL, mxL, myL, EL, rhoR, mxR, myR, ER, F_rho, F_mx, F_my, F_E, dir);
        return;
    }

    Real rhoL_star = rho_star_side(p_star, rhoL_s, pL);
    Real rhoR_star = rho_star_side(p_star, rhoR_s, pR);

    if (!std::isfinite(rhoL_star) || !std::isfinite(rhoR_star) ||
        rhoL_star <= RhoMin || rhoR_star <= RhoMin) {
        hllc_flux(rhoL, mxL, myL, EL, rhoR, mxR, myR, ER, F_rho, F_mx, F_my, F_E, dir);
        return;
    }

    Real rho_star = (u_star >= 0.0) ? rhoL_star : rhoR_star;
    Real ut_star  = (u_star >= 0.0) ? utL : utR;

    Real F1, F2, F3;
    exact_riemann_flux_1d(p_star, u_star, rho_star, F1, F2, F3);

    Real F_mt = rho_star * ut_star * u_star;

    // Energy flux includes the tangential kinetic energy contribution
    Real E_star = p_star / (GammaGas - 1.0) + 0.5 * rho_star * (u_star * u_star + ut_star * ut_star);
    Real F_E_full = (E_star + p_star) * u_star;

    F_rho = F1;
    F_E   = F_E_full;
    set_normal_tangential(F2, F_mt, dir, F_mx, F_my);
}


#endif