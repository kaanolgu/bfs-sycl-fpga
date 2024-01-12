//----------------------------------------------------------
//--bfs on cpu
//--based on jianbin's work
//----------------------------------------------------------
template <typename T>
int median(T v)
{
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin()+n, v.end());
  int vn = v[n];
  if(v.size()%2 == 1)
  {
    return vn;
  }else
  {
    std::nth_element(v.begin(), v.begin()+n-1, v.end());
    return 0.5*(vn+v[n-1]);
  }
}
//---------------------------------------------------------- CPU COMPUTATION
void run_bfs_cpu(int no_of_nodes,
  std::vector<unsigned int> &source_indptr,
  std::vector<unsigned int>&source_inds, 
  std::vector<unsigned int>&h_graph_mask,
  std::vector<unsigned int>&h_updating_graph_mask, 
  std::vector<unsigned int>&fpga_visited,
  std::vector<unsigned int>&hit_rate,
  std::vector<int> &h_cost_ref,
  std::vector<int> &ranges){
  char stop;
  std::vector<std::vector<int>> modulo_ratios(NUM_BITS_VISITED);

  unsigned int itr=0; // number of levels cnt
  do{
    unsigned int cnt=0; // number of nodes checked cnt
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        for(int i=source_indptr[tid]; i<(source_indptr[tid+1]); i++){
          cnt++;
          // int id = source_inds[i+9140365];  //--h_graph_edges is source_inds
          int id = source_inds[i];  // Single Processing Element--h_graph_edges is source_inds
          modulo_ratios[id%NUM_BITS_VISITED].push_back(id);
          if(!fpga_visited[id]){  //--cambine: if node id has not been visited, enter the body below
            // std::cout << "HOST ID :"  << id  << " | index_for_bit: " << index_for_bit(id) <<" | Toggle bit : " << bit_to_toggle(id)  << std::endl;
            
            h_cost_ref[id]=h_cost_ref[tid]+1;
            h_updating_graph_mask[id]=1;
          }
        }
      }    
    }

    for(int tid=0; tid< no_of_nodes ; tid++ )
    {
      if (h_updating_graph_mask[tid] == 1){
        h_graph_mask[tid]=1;
        fpga_visited[tid]=1;
        stop=1;
        h_updating_graph_mask[tid]=0;
      }
    }
    hit_rate[itr++] = cnt;
    // stop = 0;
  }while(stop);

  std::vector<double> edge_ratios(NUM_BITS_VISITED);
  unsigned int total_newly_visited_edges_cnt = 0;
  for(int i = 0; i < NUM_BITS_VISITED; i++){
  total_newly_visited_edges_cnt += modulo_ratios[i].size();
  }

  for(int i = 0; i < NUM_BITS_VISITED; i++){
  edge_ratios[i] = (double) modulo_ratios[i].size()/total_newly_visited_edges_cnt*100;
  std::cout << "Lane ["<< i << "] : " << modulo_ratios[i].size() << " [ "<< edge_ratios[i]  <<" %]" <<  std::endl;
  }


std::cout << " MEDIAN " << median(edge_ratios) << std::endl;



for(int i = 0; i < NUM_BITS_VISITED; i++){
ranges[i] = ceil((float)edge_ratios[i]/median(edge_ratios));
std::cout <<edge_ratios[i] <<" / "<< median(edge_ratios)<< " = "<< ranges[i] << "(possible value for PARALLEL_FIFO_DEPTH*x) \n";
}

}
//

