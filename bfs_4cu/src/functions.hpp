#define MAX_NUM_CU 4
// using MyUint1 = ac_int<1, false>;
using MyUint1 = char; 
using MyUint4 = ac_int<4, false>;
using MyUInt32 = ac_int<32, false>;
using MyUInt64 = ac_int<64, false>;
int numRows,numCols,numNonz;

#define DEBUG(x) std::cout <<" : "<< x << std::endl;
//Structure to hold a node information
struct ComputeUnit
{
  
  MyUint1 *usm_mask;


};
//Structure to hold a node information
struct HostGraphData
{
  
  // std::vector<unsigned int> h_graph_nodes_edges;
  std::vector<MyUint1> h_graph_mask;
  


};

typedef std::array<HostGraphData, MAX_NUM_CU> GraphData;


void HostGraphDataGenerate(int indexPE,int start_vertex,GraphData &fpga_cu_data,std::vector<unsigned int>&source_meta,std::vector<unsigned int>&source_indptr,std::vector<unsigned int>&source_inds,std::vector<unsigned int>& old_buffer_size_meta,
	std::vector<unsigned int>& old_buffer_size_indptr,
	std::vector<unsigned int>& old_buffer_size_inds) 
{

  numRows  = source_meta[0 + old_buffer_size_meta[indexPE]];  // this it the value we want! (rows)
	numNonz  = source_meta[2 + old_buffer_size_meta[indexPE]];  // nonZ count -> total edges
  // Sanity Check if we loaded the graph properly
  assert(numRows <= numCols);

  std::cout << std::setw(6) << std::left << "# Graph Information" << "\n Vertices (nodes) = " << numRows << " \n Edges = "<< numNonz << "\n";
	
  
  
  // allocate host memory


  fpga_cu_data[indexPE].h_graph_mask.resize(numCols);

 // initialise all the values to 0
  std::fill(fpga_cu_data[indexPE].h_graph_mask.begin(), fpga_cu_data[indexPE].h_graph_mask.end(), 0);  







    

   fpga_cu_data[indexPE].h_graph_mask[start_vertex]=1; 

}


// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void initialize(queue &Q,T val, T *arr,int gws,int pos = -1, T pos_val = -1)

{


    Q.parallel_for(gws, [=](id<1> i) [[intel::kernel_args_restrict]] {
                                  
                                      arr[i] = val;
    
                                      if (i == pos)
                                      {
                                          arr[pos] = pos_val;
                                      }
                                   }).wait(); 

}