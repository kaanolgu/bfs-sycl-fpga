#include "bfs.hpp"
namespace bfs {
namespace eight{
inline void  explorer_wrapper(
                                queue &q,
                                                                unsigned int &h_over,
                                event &e_explore_0,
                                event &e_explore_1,
                                event &e_explore_2,
                                event &e_explore_3,
                                event &e_explore_4,
                                event &e_explore_5,
                                event &e_explore_6,
                                event &e_explore_7,
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
                                double &time_kernel4,
                                double &time_kernel5,
                                double &time_kernel6,
                                double &time_kernel7,
                                double &time_kernel_levelgen,
                                double &time_kernel_levelgen_1,
                                double &time_kernel_remove, 
                                double &time_kernel_remove1){

    
     e_explore_0 = parallel_explorer_kernel<0>(q,no_of_nodes,offset[0],offset_inds[0],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_1 = parallel_explorer_kernel<1>(q,no_of_nodes,offset[1],offset_inds[1],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_2 = parallel_explorer_kernel<2>(q,no_of_nodes,offset[2],offset_inds[2],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_3 = parallel_explorer_kernel<3>(q,no_of_nodes,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_4 = parallel_explorer_kernel<4>(q,no_of_nodes,offset[4],offset_inds[4],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_5 = parallel_explorer_kernel<5>(q,no_of_nodes,offset[5],offset_inds[5],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_6 = parallel_explorer_kernel<6>(q,no_of_nodes,offset[6],offset_inds[6],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
     e_explore_7 = parallel_explorer_kernel<7>(q,no_of_nodes,offset[7],offset_inds[7],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
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
    time_kernel2 += GetExecutionTime(e_explore_2);
    time_kernel3 += GetExecutionTime(e_explore_3);
    time_kernel4 += GetExecutionTime(e_explore_4);
    time_kernel5 += GetExecutionTime(e_explore_5);
    time_kernel6 += GetExecutionTime(e_explore_6);
    time_kernel7 += GetExecutionTime(e_explore_7);

    time_kernel_levelgen += GetExecutionTime(e_visited_update_0);
    time_kernel_levelgen_1 += GetExecutionTime(e_visited_update_1);
    time_kernel_remove += GetExecutionTime(e_remove_0);
    time_kernel_remove1 += GetExecutionTime(e_remove_1);
    global_level++;
                                }
} //end namespace 2 compute unit
// namespace one{
// inline void  explorer_wrapper(
//                                 queue &q,
//                                                                 unsigned int &h_over,
//                                 event &e_explore_0,
//                                 event &e_visited_update_0,
//                                 event &e_visited_update_1,
//                                 event &e_remove_0,
//                                 event &e_remove_1,
//                                 int no_of_nodes,
//                                 std::vector<unsigned int> &offset,
//                                 std::vector<unsigned int> &offset_inds,
//                                 unsigned int* usm_nodes_start,
//                                 unsigned int *usm_edges,
//                                 int *usm_dist,
//                                 unsigned int* usm_pipe,
//                                 MyUint1 *usm_updating_mask,
//                                 MyUint1 *usm_visited,
//                                 int &global_level,
//                                 int numCols,
//                                 unsigned int *d_over,
//                                 double &time_kernel,
// double &time_kernel1,
// double &time_kernel2,
// double &time_kernel3,
// double &time_kernel_levelgen,
// double &time_kernel_levelgen_1,
// double &time_kernel_remove, 
// double &time_kernel_remove1){

    
//      e_explore_0 = parallel_explorer_kernel<0>(q,no_of_nodes,offset[0],offset_inds[0],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//          q.wait();

//     e_visited_update_0 =parallel_levelgen_kernel<1>(q,0,numCols/2,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
//     e_visited_update_1=parallel_levelgen_kernel<2>(q,numCols/2,numCols,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
//         e_remove_0 = bfs::parallel_pipegen_kernel<8>(q,numCols,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
//         q.wait();
//     e_remove_1 = maskremove_kernel<8>(q,numCols,usm_updating_mask);
//       q.wait();
//     q.memcpy(&h_over, d_over, sizeof(unsigned int));
//     // h_over++;
 

//     time_kernel  += GetExecutionTime(e_explore_0);
//     time_kernel_levelgen += GetExecutionTime(e_visited_update_0);
//     time_kernel_levelgen_1 += GetExecutionTime(e_visited_update_1);
//     time_kernel_remove += GetExecutionTime(e_remove_0);
//     time_kernel_remove1 += GetExecutionTime(e_remove_1);
//     global_level++;
   
//                                 }
// } //end namespace 1 compute unit

// namespace eight{
// inline void  explorer_wrapper(
//                                 queue &q,
//                                 unsigned int &h_over,
//                                 event &e_explore_0,
//                                 event &e_explore_1,
//                                 event &e_explore_2,
//                                 event &e_explore_3,
//                                 event &e_explore_4,
//                                 event &e_explore_5,
//                                 event &e_explore_6,
//                                 event &e_explore_7,
//                                 event &e_visited_update_0,
//                                 event &e_visited_update_1,
//                                 event &e_remove_0,
//                                 event &e_remove_1,
//                                 int no_of_nodes,
//                                 std::vector<unsigned int> &offset,
//                                 std::vector<unsigned int> &offset_inds,
//                                 unsigned int* usm_nodes_start,
//                                 unsigned int *usm_edges,
//                                 int *usm_dist,
//                                 unsigned int* usm_pipe,
//                                 MyUint1 *usm_updating_mask,
//                                 MyUint1 *usm_visited,
//                                 int &global_level,
//                                 int numCols,
//                                 unsigned int *d_over,
//                                 double &time_kernel,
// double &time_kernel1,
// double &time_kernel2,
// double &time_kernel3,
// double &time_kernel_levelgen,
// double &time_kernel_levelgen_1,
// double &time_kernel_remove, 
// double &time_kernel_remove1){

    
//     e_explore_0 = parallel_explorer_kernel<0>(q,no_of_nodes,offset[0],offset_inds[0],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     e_explore_1 = parallel_explorer_kernel<1>(q,no_of_nodes,offset[1],offset_inds[1],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     e_explore_2 = parallel_explorer_kernel<2>(q,no_of_nodes,offset[2],offset_inds[2],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     e_explore_3 = parallel_explorer_kernel<3>(q,no_of_nodes,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     e_explore_4 = parallel_explorer_kernel<4>(q,no_of_nodes,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     e_explore_5 = parallel_explorer_kernel<5>(q,no_of_nodes,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     e_explore_6 = parallel_explorer_kernel<6>(q,no_of_nodes,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     e_explore_7 = parallel_explorer_kernel<7>(q,no_of_nodes,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
//     q.wait();
//     e_visited_update_0 =parallel_levelgen_kernel<1>(q,0,numCols/2,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
//     e_visited_update_1=parallel_levelgen_kernel<2>(q,numCols/2,numCols,usm_dist,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
//         e_remove_0 = parallel_pipegen_kernel<8>(q,numCols,usm_pipe, d_over,usm_updating_mask,usm_visited,global_level);
//         q.wait();
//     e_remove_1 = maskremove_kernel<8>(q,numCols,usm_updating_mask);
//       q.wait();
//     q.memcpy(&h_over, d_over, sizeof(unsigned int));
//     // h_over++;
 

//     time_kernel  += GetExecutionTime(e_explore_0);
//     time_kernel1 += GetExecutionTime(e_explore_1);
//     time_kernel2 += GetExecutionTime(e_explore_2);
//     time_kernel3 += GetExecutionTime(e_explore_3);
//     time_kernel3 += GetExecutionTime(e_explore_4);
//     time_kernel3 += GetExecutionTime(e_explore_5);
//     time_kernel3 += GetExecutionTime(e_explore_6);
//     time_kernel3 += GetExecutionTime(e_explore_7);
//     time_kernel_levelgen += GetExecutionTime(e_visited_update_0);
//     time_kernel_levelgen_1 += GetExecutionTime(e_visited_update_1);
//     time_kernel_remove += GetExecutionTime(e_remove_0);
//     time_kernel_remove1 += GetExecutionTime(e_remove_1);
//     global_level++;

//                                 }
// } //end namespace 8 compute unit

} // end namespace bfs