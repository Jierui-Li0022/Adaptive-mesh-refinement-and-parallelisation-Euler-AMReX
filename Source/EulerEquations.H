// Equation of state and flux functions for the 2D Euler equations.
// Conserved variables: rho, mx = rho*u, my = rho*v, E (total energy).
// All functions are GPU-device inlined for use inside AMReX ParallelFor kernels.

#ifndef EULER_EQUATIONS_H_
#define EULER_EQUATIONS_H_

#include "Constants.H"
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

using amrex::Real;

// Conserved variable indices into AMReX Array4
constexpr int URHO  = 0;
constexpr int UMX   = 1;
constexpr int UMY   = 2;
constexpr int UEDEN = 3;
constexpr int NCOMP = 4;


// Density and pressure floors to prevent unphysical states in near-vacuum regions.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real clamp_rho(Real rho) {
    if (!std::isfinite(rho)) return RhoMin;
    return (rho < RhoMin) ? RhoMin : rho;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real clamp_p(Real p) {
    if (!std::isfinite(p)) return PMin;
    return (p < PMin) ? PMin : p;
}


// Pressure from conserved variables.
// Kinetic energy is computed directly from momenta to avoid forming u, v first.
// E is clamped so that p >= PMin before the floor is applied.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho, Real mx, Real my, Real E) {
    rho = clamp_rho(rho);

    Real ke = 0.5 * (mx * mx + my * my) / rho;

    const Real Emin_phys = ke + PMin / (GammaGas - 1.0);
    if (!std::isfinite(E) || E < Emin_phys) E = Emin_phys;

    Real p = (GammaGas - 1.0) * (E - ke);
    if (p < PMin) p = PMin;
    return p;
}

// Sound speed c = sqrt(gamma * p / rho).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real sound_speed(Real rho, Real mx, Real my, Real E) {
    rho = clamp_rho(rho);
    Real p = pressure(rho, mx, my, E);
    return std::sqrt(GammaGas * p / rho);
}


// Conserved -> primitive conversion.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void conserved_to_primitive(Real rho_in, Real mx, Real my, Real E,
                            Real& rho_out, Real& u, Real& v, Real& p) {
    rho_out = clamp_rho(rho_in);
    u = mx / rho_out;
    v = my / rho_out;
    p = pressure(rho_out, mx, my, E);
}

// Primitive -> conserved conversion.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void primitive_to_conserved(Real rho, Real u, Real v, Real p,
                            Real& rho_out, Real& mx, Real& my, Real& E) {
    rho_out = rho;
    mx = rho * u;
    my = rho * v;
    E = p / (GammaGas - 1.0) + 0.5 * rho * (u * u + v * v);
}


// Physical flux F(U) in direction dir (0 = x, 1 = y).
// x: F = (mx,  mx*u + p,  my*u,      (E+p)*u)
// y: F = (my,  mx*v,      my*v + p,  (E+p)*v)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void physical_flux(Real rho, Real mx, Real my, Real E,
                   Real& F_rho, Real& F_mx, Real& F_my, Real& F_E,
                   int dir) {
    rho = clamp_rho(rho);
    Real p = pressure(rho, mx, my, E);

    if (dir == 0) {
        Real u = mx / rho;
        F_rho = mx;
        F_mx  = mx * u + p;
        F_my  = my * u;
        F_E   = (E + p) * u;
    } else {
        Real v = my / rho;
        F_rho = my;
        F_mx  = mx * v;
        F_my  = my * v + p;
        F_E   = (E + p) * v;
    }
}


// Maximum signal speed |u| + c or |v| + c in direction dir.
// Used for the CFL timestep estimate.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real max_signal_speed(Real rho, Real mx, Real my, Real E, int dir) {
    rho = clamp_rho(rho);
    Real p = pressure(rho, mx, my, E);
    Real c = std::sqrt(GammaGas * p / rho);

    if (dir == 0) {
        return std::abs(mx / rho) + c;
    } else {
        return std::abs(my / rho) + c;
    }
}

// Global max signal speed sqrt(u^2+v^2) + c across both directions.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real max_signal_speed_all(Real rho, Real mx, Real my, Real E) {
    rho = clamp_rho(rho);
    Real p = pressure(rho, mx, my, E);
    Real c = std::sqrt(GammaGas * p / rho);
    Real u = mx / rho;
    Real v = my / rho;
    return std::sqrt(u * u + v * v) + c;
}


// Returns false if density or pressure is non-positive or non-finite.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool check_conserved(Real rho, Real mx, Real my, Real E) {
    if (!std::isfinite(rho) || rho <= 0) return false;
    Real p = (GammaGas - 1.0) * (E - 0.5 * (mx * mx + my * my) / rho);
    if (!std::isfinite(p) || p <= 0) return false;
    return true;
}

// Clamp a conserved state to physical validity in-place.
// Used as a last resort after the half-step predictor.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void enforce_physical(Real& rho, Real& mx, Real& my, Real& E) {
    if (!std::isfinite(rho) || rho < RhoMin) rho = RhoMin;

    Real u = mx / rho;
    Real v = my / rho;
    if (!std::isfinite(u)) { mx = 0.0; u = 0.0; }
    if (!std::isfinite(v)) { my = 0.0; v = 0.0; }

    const Real Emin = 0.5 * (mx * mx + my * my) / rho + PMin / (GammaGas - 1.0);
    if (!std::isfinite(E) || E < Emin) E = Emin;
}


#endif