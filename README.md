# bfs-sycl-fpga
This work has been submitted to Ninth International Workshop on Heterogeneous High-performance Reconfigurable Computing (H2RC'23) 

The Breadth-First Search algorithm implementation using Intel oneAPI (SYCL2020) on Intel Arria10 and Stratix10 FPGAs

## Producing and Converting Datasets
Tested with Python 3.9.6
```bash
python -m venv venv-graphgen
. venv-graphgen/bin/activate
pip install -U pip
pip install numpy networkit pandas
pip install cython;
cd scripts
python genGraph.py rmat 24 16
```
If you have datasets ready as text file, convert graph text files to the bin files : 
```bash
python generator.py (dataset name) (partition)
python generator.py wiki-Talk 1
```

## Build for Hardware Execution
```bash
cd bfs_*cu
mkdir build
cd build
cmake .. -DFPGA_DEVICE=$AOCL_BOARD_PACKAGE_ROOT:$FPGA_BOARD_NAME 
make fpga
```

## Build for Hardware Emulation
```bash
cd bfs_*cu
mkdir build
cd build
cmake .. -DFPGA_DEVICE=$AOCL_BOARD_PACKAGE_ROOT:$FPGA_BOARD_NAME 
make fpga_emu
```

## AOCL Profiler
```bash
cd bfs_*cu
aocl profile -output-dir /path/to/bfs_*cu/aocl/ ./build/bfs.fpga (GraphName)  (Partition) (RootNode)
```

## Optimisations
[Optimisations Wiki Page](https://github.com/kaanolgu/bfs-sycl-fpga/wiki/Optimisation-Guide)

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
