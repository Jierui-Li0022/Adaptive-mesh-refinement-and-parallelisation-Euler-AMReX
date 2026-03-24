# AMReX Euler Solver — 2D Adaptive Mesh Refinement

## Files written by me

| File | Description |
|------|-------------|
| `Source/AmrLevelAdv.cpp` | Time advance, CTU scheme, dual-energy correction, AMR tagging |
| `Source/EulerEquations.H` | EOS, conserved/primitive conversion, physical flux |
| `Source/RiemannSolver.H` | HLLC, HLL, and exact Riemann solvers |
| `Source/Reconstruction.H` | MUSCL-Hancock reconstruction, Zhang-Shu limiter, CTU transverse correction |
| `Source/Constants.H` | Physical and numerical constants |
| `plot/plot_utils.py` | Shared AMReX plotfile reader and exact Riemann solver |
| `plot/plot_*.py` | Plotting scripts for 1D/2D/AMR results |

All other files in `Source/` and `Exec/` are from the AMReX tutorial
scaffold and are unchanged.

## Dependencies

- AMReX (included in this repository, unchanged from scaffold)
- MPI
- Python 3 with numpy and matplotlib (for plotting, via the `plotenv` conda environment)

## Compiling

```bash
cd Exec/UniformVelocity
make -j4
```

This produces `main2d.gnu.DEBUG.MPI.ex`.

## Running

All input files are in `Exec/UniformVelocity/`. Run from that directory:

```bash
./main2d.gnu.DEBUG.MPI.ex inputs_1d/test1_sod_100
```

| Directory | Contents |
|-----------|----------|
| `inputs_1d/` | 1D Toro tests (Tests 1–5), three resolutions each |
| `input_1d_amr/` | 1D tests with AMR enabled |
| `input_2d/` | 2D tests (x, y, diagonal orientations) |
| `input_lax_liu/no_amr/` | Lax-Liu without AMR, three resolutions |
| `input_lax_liu/amr/` | Lax-Liu with AMR, various parameter combinations |

Key parameters in the inputs file:

```
prob.split_dir   = 0       # 0=x, 1=y, 2=diagonal
prob.rhoL        = 1.0
prob.rhoR        = 0.125
prob.pL          = 1.0
prob.pR          = 0.1
prob.x0          = 0.5     # discontinuity location
prob.use_dual_energy = 0   # enable dual-energy formulation
```

## Plotting

Activate the plotting environment first, then run scripts from the `plot/` directory.
All scripts depend on `plot_utils.py` in the same directory.

```bash
conda activate plotenv
cd plot

# 1D Toro tests (3 resolutions + exact solution + convergence table)
python3 plot_toro.py all

# 1D AMR results
python3 plot_toro_amr.py all

# 2D x-direction slice vs exact
python3 plot_toro_2d_x.py all

# 2D y-direction slice vs exact
python3 plot_toro_2d_y.py all

# 2D diagonal slice vs exact
python3 plot_toro_2d_diag.py all

# AMR mesh patch visualisation
python3 plot_amr_mesh.py

# Lax-Liu Configuration 3
python3 plot_lax_liu.py
```

Results are saved under `plot/result/`.

## Lax-Liu parallelisation study

The five parts of the parallelisation study all use the Lax-Liu Configuration 3 test.
Run all commands from `Exec/UniformVelocity/`.

**Part 1 — three resolutions, no AMR, single core**

```bash
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_200
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_400
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_800
```

Record the `Run time` printed at the end of each run.

**Part 2 — MPI scaling, 1 to 16 cores**

```bash
mpirun -n 1  ./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_400
mpirun -n 2  ./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_400
mpirun -n 4  ./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_400
mpirun -n 8  ./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_400
mpirun -n 16 ./main2d.gnu.DEBUG.MPI.ex input_lax_liu/no_amr/lax_liu_400
```

**Part 3 — AMR vs uniform grid, single core**

```bash
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev1
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2
```

Compare run times against the uniform-grid results from Part 1.

**Part 4 — AMR + MPI, maximum speedup**

```bash
mpirun -n 16 ./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2
```

**Part 5 — AMR parameter study**

```bash
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2_bf2
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2_bf8
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2_errbuf_1
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2_errbuf_2
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2_errbuf_4
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2_phigrad_03
./main2d.gnu.DEBUG.MPI.ex input_lax_liu/amr/lax_liu_200_lev2_phigrad_08
```

Parameters varied: buffer factor (`bf`), error buffer cells (`errbuf`),
gradient tagging threshold (`phigrad`).

**Plotting Lax-Liu results**

```bash
conda activate plotenv
cd plot
python3 plot_lax_liu.py
python3 plot_amr_mesh.py
python3 plot_amr_grid_diag.py
```