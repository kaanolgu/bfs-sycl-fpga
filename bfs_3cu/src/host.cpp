#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "exception_handler.hpp"
#include "run_bfs_fpga.hpp"
#include "run_bfs_cpu.hpp"
#include "graph_load.hpp"

using namespace std;
namespace ext_oneapi = sycl::ext::oneapi;
unsigned int hardwarePECount,matrixID;



#define DATA_SIZE 512
#define INDPTR_SIZE 2096648





 void setBit(unsigned int ind,std::vector<unsigned int> &m_bitmap) {
	m_bitmap[ind] = 1;
}

 bool testBit(unsigned int ind,std::vector<unsigned int> &m_bitmap) {
	return (m_bitmap[ind] == 1);
}




//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatype>
void print_levels(std::vector<datatype> &A, int size,std::string name){
  for(int i = 0; i < size; i++){
    std::cout << std::setw(6) << std::left << " -> Level ("<< name << ") #" << i << " : "<< count(A.begin(), A.end(), i) << "\n";
  }
  std::cout<<std::endl;
}

template<typename datatype>
void compare_results(std::vector<datatype> &cpu_results, std::vector<datatype> &gpu_results, const int size) {

  char passed = true; 
  for (int i=0; i<size; i++){
    if (cpu_results[i]!=gpu_results[i])
      passed = false; 
  }
  if (passed)
    std::cout << "TEST PASSED!" << std::endl;
  else
    std::cout << "TEST FAILED!" << std::endl;
  return ;
}


int main(int argc, char * argv[])
{
  // Measure execution times.
  // double elapsed_s = 0;
  // double elapsed_p = 0;
  
    
   
///////////////////////////////////////////////////////////////////////////
// ./build/fpga_bfs.emu rmat-19-32 #partitions

  datasetName = argv[1];  
  hardwarePECount = stoi(argv[2]);
  int start_vertex = stoi(argv[3]);

	std::vector<unsigned int> old_buffer_size_meta(1,0);
	std::vector<unsigned int> old_buffer_size_indptr(1,0);
	std::vector<unsigned int> old_buffer_size_inds(1,0);
	std::vector<unsigned int> old_buffer_size_config(1,0);
  unsigned int offset_meta =0;
  unsigned int offset_indptr =0;
  unsigned int offset_inds =0;
  
	std::cout << "######################LOADING MATRIX#########################" << std::endl;
	loadMatrix(hardwarePECount,old_buffer_size_meta,old_buffer_size_indptr,old_buffer_size_inds,old_buffer_size_config,offset_meta,offset_indptr,offset_inds);
	std::cout << "#############################################################\n" << std::endl;

  
  int cpuCUnum=0;
  numCols  = source_meta[1];  // cols -> total number of vertices
  // this is different than the kernel rows because we want to use the whole matrix to run on cpu not a partition
  std::vector<unsigned int> sw_bfs_results(numCols,-1);

  printf("run bfs (#nodes = %d) on host (cpu) \n", numCols);

  // initalize the memory again
  std::vector<unsigned int> host_nodes_start(numCols);
  std::vector<unsigned int> host_nodes_end(numCols);
  std::vector<unsigned int> host_graph_mask(numCols,0);
  std::vector<unsigned int> host_updating_graph_mask(numCols,0);
  std::vector<unsigned int> host_graph_visited(numCols,0);
  // allocate mem for the result on host side
  std::vector<int> host_level(numCols,-1);
    
  int indptr_start = old_buffer_size_indptr[cpuCUnum]/4;
  int indptr_end = old_buffer_size_indptr[cpuCUnum]/4 + old_buffer_size_indptr[cpuCUnum+1]/4;
// std::cout << " __ _ _ _ __ _ indptr ______host_____ \n";
//  DEBUG(indptr_start);
//  DEBUG(indptr_end);
//  DEBUG("");
  // initalize the memory
  for(int i = indptr_start,index=0; i < indptr_end-1; index++,i++){
    host_nodes_start[index] =  source_indptr[i];
    host_nodes_end[index] = source_indptr[i+1];
  }

  //set the start_vertex node as 1 in the mask

  host_graph_mask[start_vertex]=1;
  host_graph_visited[start_vertex]=1;
 host_level[start_vertex]=0; 



   for(int i=host_nodes_start[1013]; i<(host_nodes_end[1013]); i++){
          // int id = source_inds[i + 9140365];  //--h_graph_edges is source_inds
          int id = source_inds[i];  //--h_graph_edges is source_inds
          std::cout << ":: " << id << " num of neighbours of this ->"<<(host_nodes_end[id] - host_nodes_start[id]) << std::endl;
   }

 
  run_bfs_cpu(numCols,host_nodes_start,host_nodes_end,source_inds, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level);



////////////
// FPGA
///////////


 
    // allocate mem for the result on host side
  std::vector<int> h_dist(numCols,-1);
  h_dist[start_vertex]=0;  

std::vector<unsigned int> h_graph_nodes_start;


    

    //read the start_vertex node from the file
  //set the start_vertex node as 1 in the mask

const int numcu = hardwarePECount;
    std::vector<MyUint1> h_updating_graph_mask(numCols,0);
  std::vector<MyUint1> h_graph_visited(numCols,0); 
h_graph_visited[start_vertex]=1; 

    std::vector<unsigned int> offset(4,0);
    std::vector<unsigned int> offset_inds_vec(4,0);
  GraphData fpga_cu_data;
  // iterate over num of compute units to generate the graph partition data
  for(int indexPE = 0; indexPE < hardwarePECount; indexPE++){
    int indptr_start = old_buffer_size_indptr[indexPE];
    int indptr_end = old_buffer_size_indptr[indexPE+1];
  int inds_start = old_buffer_size_inds[indexPE];
  int inds_end = old_buffer_size_inds[indexPE+1];
  offset[indexPE] = indptr_start;
  offset_inds_vec[indexPE] =inds_start;
  // initalize the memory


  
  



    HostGraphDataGenerate(indexPE,start_vertex,fpga_cu_data,source_meta,source_indptr,source_inds,old_buffer_size_meta,old_buffer_size_indptr,old_buffer_size_inds);
  }
  run_bfs_fpga<8>(numCols,
                  fpga_cu_data,
                  source_inds,
                  source_indptr,
                  h_updating_graph_mask,
                  h_graph_visited,
                  h_dist,
                  offset,
                  offset_inds_vec,
                  start_vertex,numcu);  


 
  print_levels(h_dist,20,"fpga"); // CPU Results

  
  // unsigned int newarr[16] = {1013, 1083, 1260, 1385, 847, 484, 1351, 1050, 430, 742, 344, 331, 216, 12, 1386, 816};
      // cout << "Time sequential: " << (elapsed_seq/1000000000)  << " -  " <<CPU_bfstimeInSeconds << " sec\n";


  // verify

  std::cout << "\n ////////////VERIFY//////////// \n";
  print_levels(host_level,8,"cpu"); // CPU Results

  compare_results(host_level, h_dist, numRows);


    std::cout << "\n ////////////VERIFY//////////// \n";
    for(int i = 0; i < 10; i++)
    std::cout << h_dist[i] << "\n";
  
  return 0;
}