#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <AMReX_REAL.H>

using amrex::Real;

// Ideal gas specific heat ratio
constexpr Real GammaGas = 1.4;

// Floors to prevent unphysical states in near-vacuum regions
constexpr Real RhoMin       = 1e-12;  // density floor
constexpr Real PMin         = 1e-12;  // pressure floor

// Thresholds used in flux computation and positivity checks
constexpr Real RhoVac       = 1e-14;  // treat as vacuum below this density
constexpr Real RhoNearVac  = 1e-6;   // skip Hancock half-step near vacuum
                                        // (lowered from 1e-3 after Toro Test 2,
                                        // where star-region pressure ~8.5e-4)
constexpr Real PHalfstepMin = 1e-10; // revert half-step if post-predictor
                                        // pressure falls below this

// Zhang-Shu positivity-preserving limiter (JCP 229, 2010)
constexpr Real ZsEps        = 1e-13;

// Shock flattening: zero slopes when p_max > ShockFlatRatio * PMin
// and flow is compressing
constexpr Real ShockFlatRatio = 3.0;

// Dual-energy switch: rebuild pressure from entropy proxy when
// kinetic energy dominates total energy by this fraction
constexpr Real KeSwitch     = 0.99;

// AMR tagging: de-tag cells below this density to prevent runaway
// refinement in near-vacuum regions (e.g. Toro Test 2)
constexpr Real RhoTagMin   = 0.3;

// Isentropic clamp for near-vacuum cells on fine AMR levels
// (enabled via prob.e_int_clamp = 1)
constexpr Real RhoClampThresh = 0.5;

// Exact Riemann solver (Newton-Raphson)
constexpr Real PStarMax    = 1e6;   // pressure upper bound
constexpr int  NrMaxIter   = 50;    // maximum iterations

#endif