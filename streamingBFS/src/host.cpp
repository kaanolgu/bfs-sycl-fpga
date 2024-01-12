#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>
#include "exception_handler.hpp"
#include "graph_load.hpp"
#include "run_bfs_fpga.hpp"
#include "run_bfs_cpu.hpp"


#include <type_traits>

using namespace std;
namespace ext_oneapi = sycl::ext::oneapi;
unsigned int matrixID;

void setBit(unsigned int ind, std::vector<unsigned int>& m_bitmap) {
  m_bitmap[ind] = 1;
}

bool testBit(unsigned int ind, std::vector<unsigned int>& m_bitmap) {
  return (m_bitmap[ind] == 1);
}
template <typename datatype>
double checkRatio(int i,std::vector<datatype>& B,std::vector<unsigned int>&total){

   return(((float) count(B.begin(), B.end(), i+1) / (float) total[i]) * 100);

}
template <typename datatype>
double checkExplored(int i,std::vector<datatype>& total){
  
   return((double)total[i]);
  
}
// const char separator    = ' ';
// const int nameWidth     = 24;
// const int numWidth      = 24;
template <typename T>
class matrix
{
public:
    matrix(std::initializer_list<std::vector<T>> l) : v{transpose(l)} {}

    size_t getRows() const { return v.size(); }
    size_t getCols() const { return (v.size() > 0) ? v[0].size() : 0; }
    T at(size_t x, size_t y) const { return v.at(x).at(y); }

    void print(int colWidth) const
    {
       std::cout   <<"| " << std::setw(colWidth) << std::setfill(separator)<< "Level" 
                   <<" | " << std::setw(colWidth) << std::setfill(separator)<< "CPU"
                   <<" | " << std::setw(colWidth-1) << std::setfill(separator)<< "FPGA"
                   <<" | " << std::setw(colWidth-1) << std::setfill(separator)<< "Time(s)"
                   <<" | " << std::setw(colWidth-1) << std::setfill(separator)<< "#Checked"
                   <<" | " << std::setw(colWidth-1) << std::setfill(separator)<< "Rate(%)" 
                   <<" |\n";
  std::cout  <<std::setw(colWidth+2)<< std::setfill('-')  << "| "
             << std::setw(colWidth+3)<< std::setfill('-') << " | "  
             << std::setw(colWidth+2)<< std::setfill('-') << " | "    
             << std::setw(colWidth+2)<< std::setfill('-') << " | "   
             << std::setw(colWidth+2)<< std::setfill('-') << " | "    
             << std::setw(colWidth+2)<< std::setfill('-') << " | "   
             <<" |"<< std::endl;


        for (unsigned x = 0; x < getRows()-1; x++)
        {
          
             std::cout << "| "  << std::setw(colWidth) << std::setfill(separator)
              << "Level #" + std::to_string(x) << " | " ;
            for (unsigned y = 0; y < getCols(); y++)
            {
                std::cout << std::setw(colWidth) << at(x, y) << " |";
            }
            std::cout << "\n";
        }

        std::cout << "\n * time : FPGA execution time in seconds \n * checked : Number of total edges explored \n * rate : Number of non-visited edges/ total edges\n";
    }

private:
    std::vector<std::vector<T>> transpose(std::initializer_list<std::vector<T>> l) const
    {
        if (l.size() == 0)
        {
            return {};
        }

        size_t rows = l.begin()->size();
        size_t cols = l.size();

        std::vector<std::vector<T>> result(rows, std::vector<T>(cols));

        size_t i = 0;
        for (const auto &col : l)
        {
            size_t j = 0;
            for (const auto &element : col)
            {
                result[j][i] = element;
                j++;
            }
            i++;
        }

        return result;
    }

private:
    std::vector<std::vector<T>> v{};
};


//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template <typename datatype>
void print_levels(std::vector<datatype>& A,
                  std::string nameA,
                  std::vector<datatype>& B,
                  std::string nameB,
                  int size,
                  std::vector<double>& fpga_time_s,
                  std::vector<unsigned int>& hit_rate) {
    std::vector<double> col1; 
     std::vector<double>col2;
      std::vector<double> col3;
       std::vector<double> col4;
        std::vector<double>col5 ;
for(int i =0 ; i < size; i++){
  col1.push_back(count(A.begin(), A.end(), i));
  col2.push_back(count(B.begin(), B.end(), i));
  col3.push_back(fpga_time_s[i] );
  col4.push_back(checkExplored(i,hit_rate));
  col5.push_back(checkRatio(i,B,hit_rate));
}
    matrix m{col1, col2, col3,col4,col5};
    m.print(20);
//
  char passed = false;
  if (equal(A.begin(), A.end(), B.begin())) {
    passed = true;
  }
  if (passed)
  
    std::cout << "\n| " << std::left << std::setw(nameWidth/2 + 2) << std::setfill(separator)
              << " TEST PASSED!"
              << " |" << std::endl;
  else
    std::cout << "TEST FAILED!" << std::endl;
  printf("|-------------------|\n");
}

int main(int argc, char* argv[]) {
  // Measure execution times.
  // double elapsed_s = 0;
  // double elapsed_p = 0;

  ///////////////////////////////////////////////////////////////////////////
  // ./build/fpga_bfs.emu rmat-19-32 #partitions

  datasetName = argv[1];
  int start_vertex = stoi(argv[2]);

  std::vector<unsigned int> old_buffer_size_meta(1, 0);
  std::vector<unsigned int> old_buffer_size_indptr(1, 0);
  std::vector<unsigned int> old_buffer_size_inds(1, 0);

  unsigned int offset_meta = 0;
  unsigned int offset_indptr = 0;
  unsigned int offset_inds = 0;

  std::vector<unsigned int> old_buffer_size_meta_cpu(1, 0);
  std::vector<unsigned int> old_buffer_size_indptr_cpu(1, 0);
  std::vector<unsigned int> old_buffer_size_inds_cpu(1, 0);
  unsigned int offset_meta_cpu = 0;
  unsigned int offset_indptr_cpu = 0;
  unsigned int offset_inds_cpu = 0;

  std::cout << "######################LOADING MATRIX#########################" << std::endl;
  loadMatrix(NUM_COMPUTE_UNITS, old_buffer_size_meta, old_buffer_size_indptr, old_buffer_size_inds,
             offset_meta, offset_indptr, offset_inds);
  std::cout << "#############################################################\n" << std::endl;
  numCols = source_meta[1];  // cols -> total number of vertices
  loadMatrixCPU(1, old_buffer_size_meta_cpu, old_buffer_size_indptr_cpu, old_buffer_size_inds_cpu,
                offset_meta_cpu, offset_indptr_cpu, offset_inds_cpu);



//////////////
/// CPU RUN
/////////////

int cpuCUnum = 0;
  // this is different than the kernel rows because we want to use the whole matrix to run on cpu not a
  // partition

  // initalize the memory again
  std::vector<unsigned int> host_nodes_start(numCols);
  std::vector<unsigned int> host_nodes_end(numCols);
  std::vector<unsigned int> host_graph_mask(numCols, 0);
  std::vector<unsigned int> host_updating_graph_mask(numCols, 0);
  std::vector<unsigned int> host_graph_visited(numCols, 0);
  // allocate mem for the result on host side
  std::vector<int> host_level(numCols, -1);

  int indptr_start = old_buffer_size_indptr_cpu[cpuCUnum] / NUM_COMPUTE_UNITS;
  int indptr_end = old_buffer_size_indptr_cpu[cpuCUnum] / NUM_COMPUTE_UNITS +
                   old_buffer_size_indptr_cpu[cpuCUnum + 1] / NUM_COMPUTE_UNITS;
  // std::cout << " __ _ _ _ __ _ indptr ______host_____ \n";
  //  DEBUG(indptr_start);
  //  DEBUG(indptr_end);
  //  DEBUG("");
  // initalize the memory
  for (int i = indptr_start, index = 0; i < indptr_end - 1; index++, i++) {
    host_nodes_start[index] = source_indptr_cpu[i];
    host_nodes_end[index] = source_indptr_cpu[i + 1];
  }

  // set the start_vertex node as 1 in the mask

  host_graph_mask[start_vertex] = 1;
  host_graph_visited[start_vertex] = 1;
  host_level[start_vertex] = 0;

  std::vector<unsigned int> hit_rate(MAX_NUM_LEVELS, 0);
  std::vector<int> ranges(NUM_BITS_VISITED);
  run_bfs_cpu(numCols, source_indptr_cpu, source_inds_cpu, host_graph_mask, host_updating_graph_mask,
              host_graph_visited, hit_rate, host_level,ranges);

unsigned int exploredEdges = std::accumulate(hit_rate.begin(),hit_rate.end(),0);








  ////////////
  // FPGA
  ///////////

  // allocate mem for the result on host side
  std::vector<int> h_dist(numCols, -1);
  h_dist[start_vertex] = 0;

  std::vector<unsigned int> h_graph_nodes_start;

  // read the start_vertex node from the file
  // set the start_vertex node as 1 in the mask

  std::vector<visited_dt> h_updating_graph_mask(numCols, 0);
  // make this a different datatype and cast it to the kernel
  // hpm version of the stratix 10 try
  std::vector<visited_dt> h_graph_visited(numCols, 0);
  // h_graph_visited[start_vertex/16] ^= 1 << (start_vertex%16);

  // std::cout << index_for_bit(start_vertex) << " -- " <<bit_to_toggle(start_vertex) <<  std::endl;

  bitset(h_graph_visited[index_for_bit(start_vertex)], bit_to_toggle(start_vertex));
  // bitset(h_updating_graph_mask[index_for_bit(temp_start_vertex)],bit_to_toggle(temp_start_vertex));

  // std::cout << "index_for_bit: " << index_for_bit(start_vertex) <<" Toggle bit : " <<
  // bit_to_toggle(start_vertex) <<  " valvisited: " <<h_graph_visited[index_for_bit(start_vertex)] <<
  // std::endl; std::cout << "index_for_bit: " << index_for_bit(temp_start_vertex) <<" Toggle bit : " <<
  // bit_to_toggle(temp_start_vertex) <<  " valvisited: "
  // <<h_updating_graph_mask[index_for_bit(temp_start_vertex)] << std::endl;

  for (int j = 0; j < NUM_BITS_VISITED; j++) {
    //  std::cout << ": " << bit_compare(visited_new_val,visited_val,j)<<std::endl;
  }
  std::vector<unsigned int> offset(NUM_COMPUTE_UNITS, 0);
  std::vector<unsigned int> offset_inds_vec(NUM_COMPUTE_UNITS, 0);
  std::vector<unsigned int> offset_visited(NUM_COMPUTE_UNITS+1, source_meta[1]);
  // std::vector<unsigned int> offset_visited(NUM_COMPUTE_UNITS+1, 0);
  int numEdges = 0;
  // iterate over num of compute units to generate the graph partition data
  for (int indexPE = 0; indexPE < NUM_COMPUTE_UNITS; indexPE++) {
    int indptr_start = old_buffer_size_indptr[indexPE];
    // int indptr_end = old_buffer_size_indptr[indexPE+1];
    int inds_start = old_buffer_size_inds[indexPE];
    // int inds_end = old_buffer_size_inds[indexPE+1];
    offset[indexPE] = indptr_start;
    offset_inds_vec[indexPE] = inds_start;
   
    // initalize the memory
                
    numEdges += source_meta[2 + old_buffer_size_meta[indexPE]];  // nonZ count -> total edges
    numRows = source_meta[0 + old_buffer_size_meta[indexPE]];    // this it the value we want! (rows)
    numNonz = source_meta[2 + old_buffer_size_meta[indexPE]];    // nonZ count -> total edges
    // Sanity Check if we loaded the graph properly
    // assert(numRows <= numCols);
    std::cout << std::setw(6) << std::left << "# Graph Information"
              << "\n Vertices (nodes) = " << numRows << " \n Edges = " << numNonz << "\n";

    offset_visited[indexPE] = source_meta[3 + old_buffer_size_meta[indexPE]];
              

  }
 
    for(int i = 0; i < NUM_COMPUTE_UNITS+1;i++){
     std::cout << "offsetVisited[" << i << "]: "<< offset_visited[i] << std::endl;
  }
  std::vector<double> fpga_time_s;

  fpga_time_s =
      run_bfs_fpga<8>(numCols, source_inds, source_indptr, h_updating_graph_mask, h_graph_visited,
                      h_dist, offset, offset_inds_vec, start_vertex, numEdges,exploredEdges,offset_visited);


  // verify
  // For CPU test run

  
  // Select the element with the maximum value
  auto it = std::max_element(host_level.begin(), host_level.end());
  // Check if iterator is not pointing to the end of vector
  int maxLevelCPU = (*it + 2);

  print_levels(host_level, "cpu", h_dist, "fpga", maxLevelCPU, fpga_time_s, hit_rate);  // CPU Results

  return 0;
}