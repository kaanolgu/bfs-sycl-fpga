
#include "GraphProcessing.h"
#include "modules/bfs_2cu.hpp"


GraphProcessing::GraphProcessing(int hardwarePECount,int numCols,int start_vertex,std::vector<unsigned int> &source_indptr,std::vector<unsigned int> &source_inds,	std::vector<unsigned int> &old_buffer_size_indptr,
	std::vector<unsigned int> &old_buffer_size_inds) {


	h_graph_pipe.resize(numCols,0);
  h_graph_pipe[0] = start_vertex;
  unsigned int pipe_size=1;

	  // allocate mem for the result on host side
  h_dist.resize(numCols,-1);
  h_dist[start_vertex]=0;  
  h_updating_graph_mask.resize(numCols,0);
  h_graph_visited.resize(numCols,0); 
  h_graph_visited[start_vertex]=1; 

  // explore_event.resize(hardwarePECount);

  

    

    //read the start_vertex node from the file
  //set the start_vertex node as 1 in the mask





  // iterate over num of compute units to generate the graph partition data
  for(int indexPE = 0; indexPE < hardwarePECount; indexPE++){
    int indptr_start = old_buffer_size_indptr[indexPE];
  int inds_start = old_buffer_size_inds[indexPE];
  offset[indexPE] = indptr_start;
  offset_inds_vec[indexPE] =inds_start;
  // initalize the memory
  }


    
}
// # TODO : Create a Graph Dataset class and call that class inside of this class to load graph dataset

// void GraphProcessing::loadDataset() {
// }
void GraphProcessing::initQue() {
	  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif


    auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};
	// initialize Multiple Queues 
	  auto devs = get_two_devices();
    auto q1 = sycl::queue{devs[0]};
                                   
    
    
    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(device_selector, 0, prop_list);     // FPGA

     Q.push_back(q1);
     Q.push_back(q);


    std::cout << "|--------------------+---------------------|\n" <<
                 "| Running on "<< Q.size() <<" devices |" << std::endl;
    for(int i =0; i < Q.size(); i++){
    std::cout << "| " << i <<":"<< std::left << std::setw(nameWidth) << std::setfill(separator) << Q[i].get_device().get_info<info::device::name>() << "|"
              << "\n";
              
    }
    std::cout<< "|--------------------+---------------------|\n";
}

std::vector<sycl::device>  GraphProcessing::get_two_devices() {
// This function returns a vector of two (not necessarily distinct) devices,
// allowing computation to be split across said devices.
  auto devs = sycl::device::get_devices(sycl::info::device_type::cpu);
  return {devs[0]};
}



void GraphProcessing::initMem(std::vector<unsigned int> &source_indptr,std::vector<unsigned int> &source_inds) {
	  usm_nodes_start   = 	malloc_device<unsigned int>(source_indptr.size(), Q[1]);
    usm_dist 	        =   malloc_device<int>(h_dist.size(), Q[1]); 
    usm_updating_mask = 	malloc_device<MyUint1>(h_updating_graph_mask.size(), Q[1]); 
    usm_visited	      = 	malloc_device<MyUint1>(h_graph_visited.size(), Q[1]); 
    usm_edges 		    = 	malloc_device<unsigned int>(source_inds.size(), Q[1]); 
    usm_pipe 			    = 	malloc_device<unsigned int>(h_graph_pipe.size(), Q[1]); 



    initUSMvec(Q[1],usm_edges,source_inds);
    initUSMvec(Q[1],usm_nodes_start,source_indptr);
    initUSMvec(Q[1],usm_dist,h_dist);
    initUSMvec(Q[1],usm_updating_mask,h_updating_graph_mask);
    initUSMvec(Q[1],usm_visited,h_graph_visited);
    initUSMvec(Q[1],usm_pipe,h_graph_pipe);



}
// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void GraphProcessing::initUSMvec(queue &Q, T *usm_arr,std::vector<T> &arr){
  Q.memcpy(usm_arr, arr.data(), arr.size() * sizeof(T));
}
void GraphProcessing::runBFS(int hardwarePECount,int numCols){
    event e_explore_0,e_explore_1,e_explore_2,e_explore_3,e_explore_4,e_explore_5,e_explore_6,e_explore_7;
    event e_visited_update_0,e_visited_update_1,e_visited_update_2,e_visited_update_3;
    event e_remove_0,e_remove_1,e_remove_2,e_remove_3;
    double time_kernel=0,time_kernel1=0,time_kernel2=0,time_kernel3=0,time_kernel_levelgen=0,time_kernel_levelgen_1=0,time_kernel_remove=0,time_kernel_remove1=0;
    unsigned int h_over = 1;
    unsigned int *d_over = malloc_device<unsigned int>(1, Q[1]);
    int global_level = 1;
    for(int ijk=0; ijk < 15; ijk++){
      // std::cout << h_over << " -- \n";
     if(h_over == 0){
      std::cout << "total number of iterations" << ijk << "\n";
      break;
     }    
    int zero = 0;
    Q[1].memcpy(d_over, &zero, sizeof(unsigned int)).wait();
      
      // Q[1].memcpy(d_over, &h_over, sizeof(unsigned int)).wait();
      // if(hardwarePECount == 4){
  //  bfs::four::explorer_wrapper(Q[1],h_over,e_explore_0,e_explore_1,e_explore_2,e_explore_3,e_visited_update_0,e_visited_update_1,e_remove_0,e_remove_1,h_over,offset,offset_inds_vec,usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited,global_level,numCols,d_over,time_kernel,time_kernel1,time_kernel2,time_kernel3,time_kernel_levelgen,time_kernel_levelgen_1,time_kernel_remove,time_kernel_remove1);
  //     } else if(hardwarePECount == 2){
   bfs::two::explorer_wrapper(Q[1],h_over,e_explore_0,e_explore_1,e_visited_update_0,e_visited_update_1,e_remove_0,e_remove_1,h_over,offset,offset_inds_vec,usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited,global_level,numCols,d_over,time_kernel,time_kernel1,time_kernel2,time_kernel3,time_kernel_levelgen,time_kernel_levelgen_1,time_kernel_remove,time_kernel_remove1);

  //     }else if(hardwarePECount == 8){
  //  bfs::eight::explorer_wrapper(Q[1],h_over,e_explore_0,e_explore_1,e_explore_2,e_explore_3,e_explore_4,e_explore_5,e_explore_6,e_explore_7,e_visited_update_0,e_visited_update_1,e_remove_0,e_remove_1,h_over,offset,offset_inds_vec,usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited,global_level,numCols,d_over,time_kernel,time_kernel1,time_kernel2,time_kernel3,time_kernel_levelgen,time_kernel_levelgen_1,time_kernel_remove,time_kernel_remove1);

  //     }else if(hardwarePECount == 1){
  //  bfs::one::explorer_wrapper(Q[1],h_over,e_explore_0,e_visited_update_0,e_visited_update_1,e_remove_0,e_remove_1,h_over,offset,offset_inds_vec,usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited,global_level,numCols,d_over,time_kernel,time_kernel1,time_kernel2,time_kernel3,time_kernel_levelgen,time_kernel_levelgen_1,time_kernel_remove,time_kernel_remove1);

  //     }
    // e_explore_0 = bfs::parallel_explorer_kernel<0>(Q[1],h_over,offset[0],offset_inds_vec[0],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e_explore_1 = bfs::parallel_explorer_kernel<1>(Q[1],h_over,offset[1],offset_inds_vec[1],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e_explore_2 = bfs::parallel_explorer_kernel<2>(Q[1],h_over,offset[2],offset_inds_vec[2],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e_explore_3 = bfs::parallel_explorer_kernel<3>(Q[1],h_over,offset[3],offset_inds_vec[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
 


   
 
// #############################################################################################              


        
         

  

    }



    // copy usm_visited back to hostArray
    Q[1].memcpy(&h_dist[0], usm_dist, h_dist.size() * sizeof(int));

        Q[1].wait();
    sycl::free(usm_nodes_start, Q[1]);
    sycl::free(usm_edges, Q[1]);
    sycl::free(usm_dist, Q[1]);
    sycl::free(usm_visited, Q[1]);
    sycl::free(usm_updating_mask, Q[1]);

     printf(
         "|-------------------------+-------------------------|\n"
         "| Number of Vertices = %d | N = %d M            |\n"
         "|-------------------------+-------------------------|\n"
         "| Kernel                  |    Wall-Clock Time (ns) |\n"
         "|-------------------------+-------------------------|\n",numCols,123456);

  double fpga_execution_time = 0;
  // double fpga_execution_time = (std::max(std::max(time_kernel,time_kernel1),max(time_kernel2,time_kernel3)) + time_kernel_levelgen );

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

}




void GraphProcessing::printDist(){
	  printLevels(h_dist,20,"fpga"); // CPU Results
}

//-------------------------------------------------------------------
//--Print Level Information
//-------------------------------------------------------------------
template<typename datatype>
void GraphProcessing::printLevels(std::vector<datatype> &A, int size,std::string name){
  for(int i = 0; i < size; i++){
    std::cout << std::setw(6) << std::left << " -> Level ("<< name << ") #" << i << " : "<< count(A.begin(), A.end(), i) << "\n";
  }
  std::cout<<std::endl;
}





void GraphProcessing::perflog(int hardwarePECount,std::string BenchmarkID){
   std::ofstream logfile;

// std::ios::app is the open mode "append" meaning
 // new data will be written to the end of the file.
 logfile.open("output/perflog.txt", std::ios::app);
  
  // for(int i = 0; i < limit; i++){
  // logfile << std::setw(6) << std::left << " -> Level ("<< "name" << ") #" << i << " : "<< count(h_dist.begin(), h_dist.end(), i) << "\n";
  // } 
  logfile << "\n";
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << "Benchmark";
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << "# Compute Units";
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << "Target Level";
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << "Target";
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << "Verify ";
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << "Elapsed Time ";
  logfile << "|\n";
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << BenchmarkID;
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << hardwarePECount;
  logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << h_dist[1013];
  // logfile << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << target;

}
