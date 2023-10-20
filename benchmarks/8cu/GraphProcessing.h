// A2DD.h
#ifndef GraphProcessing_H
#define GraphProcessing_H
#define REQD_WORK_GROUP_SIZE 1024
#define NUM_SIMD_WORK_ITEMS 16

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
#include "functions.hpp"
using namespace sycl;
 using namespace std::chrono;
#include <math.h>
#include <iostream>
#include <bitset>


// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new ext::intel::lsu class to specify LSU style and modifiers
using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,ext::intel::statically_coalesce<false>>;
using PipelinedLSU = ext::intel::lsu<>;
using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>,ext::intel::statically_coalesce<false>>;





class GraphProcessing 
{
// Load Graph

	std::vector<queue> Q;
	   // Compute kernel execution time

	
	// Printing Values
	const char separator    = ' ';
    const int nameWidth     = 24;
    const int numWidth      = 24;

	// initMEM
	  std::vector<int> h_dist;
	  std::vector<MyUint1> h_updating_graph_mask;
	  std::vector<MyUint1> h_graph_visited;
	  std::vector<unsigned int> h_graph_pipe;

	std::vector<unsigned int> offset{0,0,0,0,0,0,0,0};
    std::vector<unsigned int> offset_inds_vec{0,0,0,0,0,0,0,0};
unsigned int* usm_nodes_start=NULL;
int *usm_dist 			=NULL;	    
MyUint1 *usm_updating_mask =NULL;
MyUint1 *usm_visited			 =NULL; 
unsigned int *usm_edges 		=NULL;
unsigned int *usm_pipe 			=NULL;

// int lcl_numCols;

public:
	GraphProcessing(int hardwarePECount,int numCols,int start_node,std::vector<unsigned int> &source_indptr,std::vector<unsigned int> &source_inds,	std::vector<unsigned int> &old_buffer_size_indptr,
	std::vector<unsigned int> &old_buffer_size_inds);
	std::vector<sycl::device>  get_two_devices();
	void initQue();
	void initMem(std::vector<unsigned int> &source_indptr,std::vector<unsigned int> &source_inds);
	template <typename T>
	void initUSMvec(queue &Q, T *usm_arr,std::vector<T> &arr);
	void runBFS(int hardwarePECount,int numCols);
	void perflog(int hardwarePECount,std::string BenchmarkID);
	template<typename datatype>
	void printLevels(std::vector<datatype> &A, int size,std::string name);
	void printDist();
  // template <int krnl_id>
  // event parallel_explorer_kernel(queue &q,int no_of_nodes,unsigned int offset,unsigned int offset_inds,unsigned int* usm_nodes_start,unsigned int *usm_edges,int *usm_dist,unsigned int* usm_pipe,MyUint1 *usm_updating_mask,MyUint1 *usm_visited);
// template <int krnl_id>
// event parallel_levelgen_kernel(queue &q,int no_of_nodes_start,int no_of_nodes_end,event e1,event e2,event e3,event e4,int *usm_dist,unsigned int *usm_pipe,unsigned int *d_over,MyUint1 *usm_updating_mask,MyUint1 *usm_visited,int global_level);
// template <int unroll_factor>
// event parallel_pipegen_kernel(queue &q,int no_of_nodes,event e1,event e2,event e3,event e4,unsigned int *usm_pipe,unsigned int *d_over,MyUint1 *usm_updating_mask,MyUint1 *usm_visited,int global_level);
// template <int unroll_factor>
// event maskremove_kernel(queue &q,int no_of_nodes,event e1,event e2,event e3,MyUint1 *usm_updating_mask);
};
#endif