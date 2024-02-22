# bfs-sycl-fpga
This work has been ~submitted~ [accepted](https://www.iwocl.org/iwocl-2024/program/#conf-wed) to 12th International Workshop on OpenCL and SYCL (IWOCL24)

The Breadth-First Search implementations _memoryBFS_ and _streamingBFS_ using Intel oneAPI (SYCL2020) on Intel FPGAs
* **_memoryBFS_:** We applied the typical optimisations proposed in the official guidelines alongside an automatic cache to achieve proper pipelining and improve random memory accesses performance. However, limitations occurred with fine-grained parallelism, and it was competitive only to some related work that utilised hardware-description languages or established high-level synthesis tools.
* **_streamingBFS_:** We added bit-level representations of data in memory, banking in on-chip memory, and fine-grained control over parallel data streams to achieve higher throughput.

Authors : [Kaan Olgu](https://research-information.bris.ac.uk/en/persons/kaan-olgu-2) & [Tobias Kenter](https://www.uni-paderborn.de/en/person/3145)
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
python generator.py (dataset name) (partition) (nnz/row)
python generator.py wiki-Talk 1 nnz
```

## Build for Hardware Execution
```bash
cd *BFS
mkdir build
cd build
# for memoryBFS:
cmake .. -DFPGA_DEVICE=$AOCL_BOARD_PACKAGE_ROOT:$FPGA_BOARD_NAME -DNUM_COMPUTE_UNITS=4 
# for streamingBFS
cmake .. -DFPGA_DEVICE=$AOCL_BOARD_PACKAGE_ROOT:$FPGA_BOARD_NAME -DNUM_COMPUTE_UNITS=4 -DK_MEMORY_CACHE=131072
make fpga
```

## Build for Hardware Emulation
```bash
cd bfs_*cu
mkdir build
cd build
# for memoryBFS:
cmake .. -DFPGA_DEVICE=$AOCL_BOARD_PACKAGE_ROOT:$FPGA_BOARD_NAME -DNUM_COMPUTE_UNITS=4 
# for streamingBFS
cmake .. -DFPGA_DEVICE=$AOCL_BOARD_PACKAGE_ROOT:$FPGA_BOARD_NAME -DNUM_COMPUTE_UNITS=4 -DK_MEMORY_CACHE=131072
make fpga_emu
```

## AOCL Profiler
```bash
cd memoryBFS
aocl profile -output-dir /path/to/memoryBFS/aocl/ ./build/bfs.fpga (GraphName)  (Partition) (RootNode)
```

## Optimisations
[Optimisations Wiki Page](https://github.com/kaanolgu/bfs-sycl-fpga/wiki/Optimisation-Guide)


## Acknowledgements
* [University of Paderborn Noctua 2](https://pc2.uni-paderborn.de/hpc-services/available-systems/noctua2)
* [Intel DevCloud for OneAPI](https://devcloud.intel.com/oneapi/)
<img src="/docs/images/bristol-alumni-and-friends.png" alt="drawing" width="200" />

