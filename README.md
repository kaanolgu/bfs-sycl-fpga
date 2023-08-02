# bfs-sycl-fpga
The Breadth-First Search algorithm implementation using Intel oneAPI (SYCL2020) on Intel Arria10 and Stratix10 FPGAs


## Build for Hardware Execution
cd bfs_*cu
mkdir build
cd build
cmake .. -DFPGA_DEVICE=$AOCL_BOARD_PACKAGE_ROOT:$FPGA_BOARD_NAME 
make fpga

## AOCL Profiler
cd bfs_*cu
aocl profile -output-dir /path/to/bfs_*cu/aocl/ ./build/bfs.fpga (GraphName) (Root) (Partition) 