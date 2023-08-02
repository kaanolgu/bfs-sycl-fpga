using namespace sycl;
 using namespace std::chrono;

#include <math.h>
#include <iostream>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <bitset>
#include "functions.hpp"

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>


// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new ext::intel::lsu class to specify LSU style and modifiers
using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,ext::intel::statically_coalesce<false>>;
using PipelinedLSU = ext::intel::lsu<>;
using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>,ext::intel::statically_coalesce<false>>;


// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int unroll_factor> class ExploreNeighbours;
template <int unroll_factor> class LevelGenerator;
//-------------------------------------------------------------------
// Return the execution time of the event, in seconds
//-------------------------------------------------------------------
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}

//-------------------------------------------------------------------
//-- initialize Kernel for Exploring the neighbours of next to visit 
//-- nodes
//-------------------------------------------------------------------
template <int krnl_id>
event parallel_explorer_kernel(queue &q,
                                int no_of_nodes,
                                unsigned int offset,
                                unsigned int offset_inds,
                                unsigned int* usm_nodes_start,
                                unsigned int *usm_edges,
                                int *usm_dist,
                                unsigned int* usm_pipe,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited)
    {
    
      // define global work size for parallel_for function
      range<1> gws (no_of_nodes);


      auto e = q.parallel_for<class ExploreNeighbours<krnl_id>>(gws, [=] (id<1> iter) [[intel::kernel_args_restrict]]  {
        device_ptr<unsigned int> DevicePtr_start(usm_nodes_start+offset);  
        device_ptr<unsigned int> DevicePtr_end(usm_nodes_start + 1+offset);  
        device_ptr<unsigned int> DevicePtr_edges(usm_edges+offset_inds);  
        device_ptr<MyUint1> DevicePtr_visited(usm_visited);  
  
          // Read from the pipe
          unsigned int idx = usm_pipe[iter];    
          // Process the current node in tiles      
          unsigned int nodes_start = DevicePtr_start[idx];
          unsigned int nodes_end = DevicePtr_end[idx];


          // Process the edges of the current nodes
          for (int j = nodes_start; j < nodes_end; j++) {
            int id = BurstCoalescedLSU::load(DevicePtr_edges + j);
            MyUint1 visited_condition = BurstCoalescedLSU::load(DevicePtr_visited + id);
            if (!visited_condition) {
                usm_updating_mask[id]=1;

            }
          }
          
        
    });
        
return e;
}


template <int krnl_id>
event parallel_levelgen_kernel(queue &q,
                                int no_of_nodes_start,
                                int no_of_nodes_end,
                                int *usm_dist,
                                unsigned int *usm_pipe,
                                unsigned int *d_over,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited,
                                int global_level
                                 ){
                                  


        auto e =q.single_task<class LevelGenerator<krnl_id>>( [=]() [[intel::kernel_args_restrict]] {
          
          #pragma unroll 8
          [[intel::initiation_interval(1)]]
          for(int tid =no_of_nodes_start; tid < no_of_nodes_end; tid++){
            unsigned int condition = usm_updating_mask[tid];
            if(condition){
              usm_dist[tid] = global_level;
              usm_visited[tid]=1;
           }
          }

        });

return e;
}
template <int unroll_factor>
event parallel_pipegen_kernel(queue &q,
                                int no_of_nodes,
                                unsigned int *usm_pipe,
                                unsigned int *d_over,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited,
                                int global_level
                                 ){
                                  


        auto e =q.single_task<class PipeGenerator>(  [=]() [[intel::kernel_args_restrict]] {
          int iter = 0;
          [[intel::initiation_interval(1)]]
          for(int tid =0; tid < no_of_nodes; tid++){
            unsigned int condition = usm_updating_mask[tid];
            if(condition){
              usm_pipe[iter] = tid;
              iter++;
            }
          }
          d_over[0] = iter;

        });

return e;
}
template <int unroll_factor>
event maskremove_kernel(queue &q,
                                int no_of_nodes,
                                MyUint1 *usm_updating_mask
                                 ){
                                  


        auto e =q.single_task<class MaskRemove>( [=]() [[intel::kernel_args_restrict]] {
          #pragma unroll 16
          // [[intel::initiation_interval(1)]]
          for(int tid =0; tid < no_of_nodes; tid++){
            unsigned int condition = usm_updating_mask[tid];
            if(condition){
              usm_updating_mask[tid]=0;  
           }
          }

        });

return e;
}

// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void initUSMvec(queue &Q, T *usm_arr,std::vector<T> &arr){
  Q.memcpy(usm_arr, arr.data(), arr.size() * sizeof(T));
}
//----------------------------------------------------------
//--breadth first search on FPGA
//----------------------------------------------------------
// This function instantiates the vector add kernel, which contains
// a loop that adds up the two summand arrays and stores the result
// into sum. This loop will be unrolled by the specified unroll_factor.
template <int unroll_factor>
void run_bfs_fpga(int numCols, 
                  GraphData &host_cu_data,
                  std::vector<unsigned int> &source_inds,
                  std::vector<unsigned int> &source_indptr,
                  std::vector<MyUint1> &h_updating_graph_mask,
                  std::vector<MyUint1> &h_graph_visited,
                  std::vector<int> &h_dist,
                  std::vector<unsigned int> &offset,
                   std::vector<unsigned int> &offset_inds,
                  int start_node,
                  const int numcu) noexcept(false) {
 

  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
 ext::intel::fpga_emulator_selector  device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif



 auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};
  try {

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(device_selector, fpga_tools::exception_handler, prop_list);

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::vector<unsigned int> h_graph_pipe(numCols,0);
    h_graph_pipe[0] = start_node;
    unsigned int pipe_size=1;

    unsigned int* usm_nodes_start = malloc_device<unsigned int>(source_indptr.size(), q);
    int *usm_dist = malloc_device<int>(h_dist.size(), q); 
    MyUint1 *usm_updating_mask = malloc_device<MyUint1>(h_updating_graph_mask.size(), q); 
    MyUint1 *usm_visited= malloc_device<MyUint1>(h_graph_visited.size(), q); 
    unsigned int *usm_edges = malloc_device<unsigned int>(source_inds.size(), q); 
    unsigned int *usm_pipe = malloc_device<unsigned int>(h_graph_pipe.size(), q); 
    unsigned int *usm_pipe_size = malloc_device<unsigned int>(1, q); 


    initUSMvec(q,usm_edges,source_inds);
    initUSMvec(q,usm_nodes_start,source_indptr);
    initUSMvec(q,usm_dist,h_dist);
    initUSMvec(q,usm_updating_mask,h_updating_graph_mask);
    initUSMvec(q,usm_visited,h_graph_visited);
    initUSMvec(q,usm_pipe,h_graph_pipe);




    unsigned int h_over = 1;
    unsigned int *d_over = malloc_device<unsigned int>(1, q);


    // Compute kernel execution time
    event e1_1,e1_2,e1_3,e_visited_update_0,e_visited_update_1,e1_4;
    event e_remove_0,e_remove_1,e_remove_2,e_remove_3;
    double time_kernel=0,time_kernel1=0,time_kernel2=0,time_kernel3=0,time_kernel_levelgen=0,time_kernel_levelgen_1=0,time_kernel_remove=0,time_kernel_remove1=0;

  

    int global_level = 1;

    
    for(int ijk=0; ijk < 15; ijk++){
      // std::cout << h_over << " -- \n";
     if(h_over == 0){
      std::cout << "total number of iterations" << ijk << "\n";
      break;
     }    
    int zero = 0;
    q.memcpy(d_over, &zero, sizeof(unsigned int)).wait();
      
      // q.memcpy(d_over, &h_over, sizeof(unsigned int)).wait();

    e1_1 = parallel_explorer_kernel<0>(q,h_over,offset[0],offset_inds[0],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e1_2 = parallel_explorer_kernel<1>(q,h_over,offset[1],offset_inds[1],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e1_3 = parallel_explorer_kernel<2>(q,h_over,offset[2],offset_inds[2],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e1_4 = parallel_explorer_kernel<3>(q,h_over,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
 
    q.wait();
 

    e_visited_update_0 =parallel_levelgen_kernel<0>(q,0,numCols/2,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
    e_visited_update_1 =parallel_levelgen_kernel<1>(q,numCols/2,numCols,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);


    e_remove_0 =parallel_pipegen_kernel<8>(q,numCols,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
    q.wait();


    e_remove_1 =maskremove_kernel<8>(q,numCols,usm_updating_mask);
 
// #############################################################################################              


        
         

    q.wait();
    q.memcpy(&h_over, d_over, sizeof(unsigned int)).wait();
    // h_over++;
 

    time_kernel  += GetExecutionTime(e1_1);
    // time_kernel1 += GetExecutionTime(e1_2);
    // time_kernel2 += GetExecutionTime(e1_3);
    // time_kernel3 += GetExecutionTime(e1_4);
    time_kernel_levelgen += GetExecutionTime(e_visited_update_0);
    time_kernel_levelgen_1 += GetExecutionTime(e_visited_update_1);
time_kernel_remove += GetExecutionTime(e_remove_0);
time_kernel_remove1 += GetExecutionTime(e_remove_1);
    global_level++;

    }





    // copy usm_visited back to hostArray
    q.memcpy(&h_dist[0], usm_dist, h_dist.size() * sizeof(int));

    q.wait();
    // sycl::free(usm_nodes_start, q);
    // sycl::free(usm_nodes_end, q);
    // sycl::free(usm_edges, q);
    sycl::free(usm_dist, q);
    sycl::free(usm_visited, q);
    // sycl::free(usm_updating_mask, q);
    // sycl::free(usm_mask, q);
 

    const char separator    = ' ';
    const int nameWidth     = 24;
    const int numWidth      = 24;

      printf(
         "|-------------------------+-------------------------|\n"
         "| Number of Vertices = %d | N = %d M            |\n"
         "|-------------------------+-------------------------|\n"
         "| Kernel                  |    Wall-Clock Time (ns) |\n"
         "|-------------------------+-------------------------|\n",numCols,123456);

  double fpga_execution_time = (max(max(time_kernel,time_kernel1),max(time_kernel2,time_kernel3)) + time_kernel_levelgen );

  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Time elapsed e1_1  : ";
  std::cout << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel) + " (s) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Time elapsed e1_2  : " << "| " << time_kernel1 << " (s) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Time elapsed e1_3  : " << "| " << time_kernel2 << " (s) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Time elapsed e1_4  : " << "| " << time_kernel3 << " (s) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " e_remove  : " << "| " << time_kernel_remove << " (s) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " e_remove1  : " << "| " << time_kernel_remove1 << " (s) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " LevelGen  : "          << "| " << time_kernel_levelgen << " (s) "<< "| "  << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Total Time Elapsed  :" << "| " << fpga_execution_time << " (s) "<< "| "  << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Throughput = "         << "| " <<(16.77/fpga_execution_time) << "| " << "\n";


    // The queue destructor is invoked when q passes out of scope.
    // q's destructor invokes q's exception handler on any device exceptions.
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