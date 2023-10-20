#include "bfs.hpp"
namespace bfs {
namespace two{
inline void  explorer_wrapper(
                                queue &q,
                                                                unsigned int &h_over,
                                event &e_explore_0,
                                event &e_explore_1,
                                event &e_visited_update_0,
                                event &e_visited_update_1,
                                event &e_remove_0,
                                event &e_remove_1,
                                int no_of_nodes,
                                std::vector<unsigned int> &offset,
                                std::vector<unsigned int> &offset_inds,
                                unsigned int* usm_nodes_start,
                                unsigned int *usm_edges,
                                int *usm_dist,
                                unsigned int* usm_pipe,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited,
                                int &global_level,
                                int numCols,
                                unsigned int *d_over,
                                double &time_kernel,
                                double &time_kernel1,
                                double &time_kernel2,
                                double &time_kernel3,
                                double &time_kernel_levelgen,
                                double &time_kernel_levelgen_1,
                                double &time_kernel_remove, 
                                double &time_kernel_remove1){

    
     e_explore_0 = parallel_explorer_kernel<0>(q,no_of_nodes,offset[0],offset_inds[0],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_1 = parallel_explorer_kernel<1>(q,no_of_nodes,offset[1],offset_inds[1],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
         q.wait();

    e_visited_update_0 =parallel_levelgen_kernel<1>(q,0,numCols/2,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
    e_visited_update_1=parallel_levelgen_kernel<2>(q,numCols/2,numCols,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
        e_remove_0 = bfs::parallel_pipegen_kernel<8>(q,numCols,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
        q.wait();
    e_remove_1 = maskremove_kernel<8>(q,numCols,usm_updating_mask);

         q.wait();
    q.memcpy(&h_over, d_over, sizeof(unsigned int));
    // h_over++;
 

    time_kernel  += GetExecutionTime(e_explore_0);
    time_kernel1 += GetExecutionTime(e_explore_1);
    time_kernel_levelgen += GetExecutionTime(e_visited_update_0);
    time_kernel_levelgen_1 += GetExecutionTime(e_visited_update_1);
    time_kernel_remove += GetExecutionTime(e_remove_0);
    time_kernel_remove1 += GetExecutionTime(e_remove_1);
    global_level++;
                                }
} //end namespace 2 compute unit
} // end namespace bfs