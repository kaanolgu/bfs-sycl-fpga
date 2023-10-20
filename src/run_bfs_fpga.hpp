// using namespace sycl;
 using namespace std::chrono;
#include "exception_handler.hpp"
#include <math.h>
#include <iostream>
#include <bitset>
#include "GraphProcessing.h"











//----------------------------------------------------------
//--breadth first search on FPGA
//----------------------------------------------------------
// This function instantiates the vector add kernel, which contains
// a loop that adds up the two summand arrays and stores the result
// into sum. This loop will be unrolled by the specified unroll_factor.

template <int unroll_factor>
void run_bfs_fpga(int numCols, 
                  std::vector<unsigned int> &source_inds,
                  std::vector<unsigned int> &source_indptr,
	std::vector<unsigned int> &old_buffer_size_indptr,
	std::vector<unsigned int> &old_buffer_size_inds,
                  int start_node,
                  int hardwarePECount) noexcept(false) {
                     try {
    // int target = 100;

    GraphProcessing acc(hardwarePECount,numCols,start_node,source_indptr,source_inds,	old_buffer_size_indptr,old_buffer_size_inds);
    acc.initQue();
    acc.initMem(source_indptr,source_inds);
        std::cout << "| "<<"Mode of Operation : BFS\n";
    acc.runBFS(hardwarePECount,numCols);
    acc.printDist();
    acc.perflog(hardwarePECount,"BFS");
      }
        catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
  }


 



