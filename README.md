# Reproduction of numerical experiments

To reproduce the numerical results on x86_64 Linux install
[uv](https://docs.astral.sh/uv/) and execute:

```shell
OPENBLAS_NUM_THREADS=1 OPENMP_NUM_THREADS=1 uv run pararb.py NUM_PROCESSES
```

# Heatsink model files

## heatsink_data/heatsink.FCStd
FreeCAD geometry created with version
[1.0.0.RC1](https://github.com/FreeCAD/FreeCAD/releases/download/1.0rc1/FreeCAD_1.0.0RC1-conda-Linux-x86_64-py311.AppImage)
Meshing with gmsh 4.12.2 installed via PyPI

## heatsink_data/heatsink.inp
Mesh exported by FreeCAD in Abaqus file format.
Direct export to vtu does not work as mesh groups are
not properly exported.

## heatsink_data/heatsink.vtu
Output of heatsink_data/convert_inp_to_vtu.py.
As loading an inp-file into skfem (using meshio internally)
is very slow, we use skfem to convert into vtu format.
The generated output retains all subdomain information.
