namespace bfs {
// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int krnl_id> class ExploreNeighbours;
template <int krnl_id> class LevelGenerator;
template <int krnl_id> class kernelComputeSIMD;
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
                                  


        auto e =q.single_task<class PipeGenerator>( [=]() [[intel::kernel_args_restrict]] {
          int iter = 0;
          // #pragma unroll 8
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
                                  


        auto e =q.single_task<class MaskRemove>(  [=]() [[intel::kernel_args_restrict]] {
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
//-------------------------------------------------------------------
// Return the execution time of the event, in seconds
//-------------------------------------------------------------------
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}


} // end namespace bfs