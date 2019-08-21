# Elliptic Multiscale Problem with MsFEM in Deal.II (MPI Parallel)

This repository is for teaching purposes only.

## Building the executable

To build the executable and Eclipse project files you must clone the repository:

```
git clone https://github.com/konsim83/deal.ii-9.1.1_SS19_demo_elliptic_multiscale_FEM_MPI.git elliptic_msfem_MPI
```
We want an out-of-source-build with build files in a folder parallel to the code:

```
mkdir elliptic_msfem_MPI_build
cd elliptic_msfem_MPI_build
```
Then create the build files with `cmake`:

```
cmake -DDEAL_II_DIR=/path/to/dealii -G"Eclipse CDT4 - Unix Makefiles" ../elliptic_msfem_MPI
```
You can now import an existing project in Eclipse. To generate the executable in debug mode type

```
make debug
make
```
If you want to produce a faster reslease version type

```
make debug
make
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
