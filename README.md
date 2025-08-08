# Parareal with Spectral Coarse Solvers

This repository contains all code required to reproduce the numerical
experiments from

> Gander, Ohlberger, Rave: A Parareal Algorithm with Spectral Coarse Solver, 2025.

To run all experiments, install [uv](https://docs.astral.sh/uv/) on an x86_64
Linux machine and execute the `run_all.sh` script. The code has been tested on
an Ubuntu 22.04.5 server with uv 0.5.1. A working MPI environment is required.
The `large` experiments are executed with 25 MPI ranks, so a sufficiently large
machine or compute cluster is needed.

To generate the figures from the paper, run `make_figures.sh` afterward. The
only exception is the figure showing the solution of the heat sink test problem.
To reproduce this figure, execute `uv run python heatsink.py`, which will
generate XDMF output files that can be visualized with
[ParaView](https://www.paraview.org/).


## Code structure

All experiments are defined as jobs in `experiments.py`. The actual
implementations of the different Parareal variants are contained in `pararb.py`.
In `heatsink.py`, the model for the heat sink test problem is defined.

The data generated from the experiments is stored in the `data/` directory.
`compute_solutions.py` computes and stores the solutions of the first three test
problems in the paper. The other `compute_*.py` files are data post-processing
scripts.


## Heat sink geometry files

### heatsink_data/heatsink.FCStd
FreeCAD geometry created with version
[1.0.0.RC1](https://github.com/FreeCAD/FreeCAD/releases/download/1.0rc1/FreeCAD_1.0.0RC1-conda-Linux-x86_64-py311.AppImage).
Meshing was done with gmsh 4.12.2 installed via PyPI.

### heatsink_data/heatsink.inp
Mesh exported by FreeCAD in Abaqus file format. Direct export to VTU does not
work as mesh groups are not properly exported.

### heatsink_data/heatsink.vtu
Output of `heatsink_data/convert_inp_to_vtu.py`. As loading an INP-file into skfem
(using meshio internally) is very slow, we use skfem to convert the mesh into
the VTU file format. The generated output retains all subdomain information.
