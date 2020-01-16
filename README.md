# Elliptic Multiscale Problem with MsFEM in Deal.II (MPI Parallel)

This repository is for teaching purposes only.

## Building the executable

To build the executable and Eclipse project files you must clone the repository:

```
git clone https://github.com/konsim83/MPI-Parallel-Multiscale-Diffusion-FEM.git elliptic_msfem_MPI
```
We want an out-of-source-build with build files in a folder parallel to the code:

```
mkdir elliptic_msfem_MPI_build
cd elliptic_msfem_MPI_build
```
Then create the build files with `cmake`:

```
cmake -DDEAL_II_DIR=/path/to/dealii -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j4 -G"Eclipse CDT4 - Unix Makefiles" ../elliptic_msfem_MPI
```
You can now import an existing project in Eclipse. To generate the executable in debug mode type

```
make debug
make -jN
```
If you want to produce a faster reslease version type

```
make release
make -jN
```
Then test the executable with

```
mpirun -n n ./main
```
where n is the number of MPI processes. You should also be able to run this on clusters using SLURM.

To run all tests type

```
ctest -V -R
```

## Building the Documentation

You will need `doxygen`, `mathjax` and some other packages such as `GraphViz` installed.

To build the documentation with `doxygen` enter the code folder

```
cd elliptic_msfem_MPI/doc
```
and type

```
doxygen Doxyfile
```
This will generate a html documentation of classes in the `elliptic_msfem_MPI/documentation/html` directory.
To open it open the `index.html` in a web browser.
