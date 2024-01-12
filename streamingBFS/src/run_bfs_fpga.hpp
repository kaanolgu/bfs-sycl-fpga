using namespace sycl;
using namespace std::chrono;
#include <math.h>
#include <algorithm>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "functions.hpp"
#include "shift_reg.hpp"
constexpr int kCacheSize = 8;
constexpr int kMemoryDepth = K_MEMORY_CACHE;
constexpr int min_depth = 256;                    // minimum capacity of each pipe
constexpr int num_pipes_x = NUM_COMPUTE_UNITS;    // number of elements x dimension of the pipeArray
constexpr int num_pipes_y = PARALLEL_FIFO_DEPTH;  // number of elements y dimension of the pipeArray
// PIPES
using ReadtoFilterWritePipeData = StreamingData<SupplierPartSupplierJoined, NUM_BITS_VISITED>;
using FiltertoWritePipeData = StreamingData<unsigned int, NUM_BITS_VISITED>;
using PipeMatrix =
    fpga_tools::PipeArray<class MyPipe, sycl::vec<visited_dt, 2>, min_depth, 2, num_pipes_x>;
using VisitUpdate = fpga_tools::PipeArray<class PipeA, visited_dt, min_depth, num_pipes_x>;
using NextTOVisitPipe =
    fpga_tools::PipeArray<class PipeF, ReadtoFilterWritePipeData, min_depth, num_pipes_x>;
using FilterResultPipe =
    fpga_tools::PipeArray<class PipeFilterResult, ReadtoFilterWritePipeData, min_depth, num_pipes_x>;
// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new ext::intel::lsu class to specify LSU style and modifiers
using PrefetchingLSU =
    ext::intel::lsu<ext::intel::prefetch<true>, ext::intel::statically_coalesce<false>>;
using PipelinedLSU = ext::intel::lsu<>;
using BurstCoalescedLSU =
    ext::intel::lsu<ext::intel::burst_coalesce<true>, ext::intel::statically_coalesce<false>>;
using CacheLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>,
                                 ext::intel::cache<1024 * 1024>,
                                 ext::intel::statically_coalesce<false>>;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int unroll_factor>
class ExploreNeighboursRead;
template <int unroll_factor>
class ExploreNeighboursFilter;
template <int unroll_factor>
class ExploreNeighboursWrite;
template <int unroll_factor>
class LevelGenerator;
template <int unroll_factor>
class PipeGenerator;
class MaskRemove;

// Names of kernels launched by SubmitMVDRKernels.
// This enum should be used to access elements in the returned vector of events.
// Note: Only the ones as single kernel
enum class kernelNames {
  LevelGenerator,
  PipeGenerator,
  MaskRemove,
  // count must come last
  count
};

using eventArray = std::array<event, static_cast<int>(kernelNames::count)>;
using timesArray = std::array<double, static_cast<int>(kernelNames::count)>;
//-------------------------------------------------------------------
// This can be used for 2^x, which would eleminate the usage of modulo operator since the logical
// operations are inexpensive
// https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/optimization-guide/2023-1/avoid-expensive-functions.html
//-------------------------------------------------------------------
int customModulo(int x) {
  return x & (NUM_BITS_VISITED - 1);  // 7 is (2^3 - 1)
}
//-------------------------------------------------------------------
// Return the execution time of the event, in seconds
//-------------------------------------------------------------------
double GetExecutionTime(const event& e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  // save current precision
  std::streamsize ss = std::cout.precision(2);
  std::cout << "end time : " << std::setprecision(15) << end_k << ",\t start time : " << start_k << ",\t difference: " << end_k - start_k << ",\t end > start : " << (end_k > start_k)
            << std::endl;
  // Restore saved precision
  std::cout << std::setprecision(ss);
  double kernel_time = (end_k - start_k) * 1e-9;  // ns to s
  return kernel_time;
}

double getMinimumStartTime(const event& e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double kernel_time = (start_k) * 1e-9;  // ns to s
  return kernel_time;
}
double GetOverlapTotalTime(const event& e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9;  // ns to s
  return kernel_time;
}

void debug_parallel_fifo(const sycl::stream& out,
                         const char* name,
                         int (&parallel_fifo)[NUM_BITS_VISITED][PARALLEL_FIFO_DEPTH],
                         MyParCnt (&cnt)[NUM_BITS_VISITED]) {
  out << name << sycl::endl;
  for (int j = 0; j < NUM_BITS_VISITED; j++) {
    out << (int)cnt[j] << ": ";
    for (int k = 0; k < cnt[j]; k++) {
      out << parallel_fifo[j][k] << ", ";
    }
    out << sycl::endl;
  }
}

void debug_write_fifo(const sycl::stream& out,
                      const char* name,
                      int (&write_fifo)[WRITE_FIFO_SIZE],
                      int cnt) {
  out << name << sycl::endl;
  out << (int)cnt << ": ";
  for (int k = 0; k < cnt; k++) {
    out << write_fifo[k] << ", ";
  }
  out << sycl::endl;
}

//-------------------------------------------------------------------
//-- initialize Kernel for Exploring the neighbours of next to visit
//-- nodes
//-------------------------------------------------------------------
template <int krnl_id>
event submitExplorerKernelRead(queue& q,
                               int no_of_nodes,
                               unsigned int* usm_nodes_start,
                               unsigned int offset,
                               unsigned int offset_inds,
                               unsigned int* usm_edges,
                               unsigned int* usm_pipe) {
  auto e = q.submit([&](auto& h) {
#ifdef FPGA_EMULATOR
    sycl::stream out(1024 * 1024 * 5, 256, h);
#endif
    h.template single_task<class ExploreNeighboursRead<krnl_id>>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<unsigned int> DevicePtr_start(usm_nodes_start + offset);
      device_ptr<unsigned int> DevicePtr_end(usm_nodes_start + 1 + offset);
      device_ptr<unsigned int> DevicePtr_edges(usm_edges + offset_inds);

      // [[intel::initiation_interval(1)]]
      for (int titer = 0; titer < no_of_nodes; titer++) {
        // Read from the pipe
        unsigned int idx = usm_pipe[titer];
        // Process the current node in tiles
        unsigned int nodes_start = DevicePtr_start[idx];
        unsigned int nodes_end = DevicePtr_end[idx];
        unsigned int nodes_lim = nodes_end - nodes_start;
        for (int tid = 0; tid < nodes_lim; tid += NUM_BITS_VISITED) {
          StreamingData<SupplierPartSupplierJoined, NUM_BITS_VISITED> pipe_out(false, true);
          // initialize all outputs to false
          fpga_tools::UnrolledLoop<NUM_BITS_VISITED>(
              [&](auto j) { pipe_out.data.template get<j>().valid = false; });
          // read 'NUM_BITS_VISITED' elements from device memory
          fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
            if (tid + j < nodes_lim) {
              pipe_out.data.get<j>().value = DevicePtr_edges[nodes_start + (tid + j)];
              pipe_out.data.template get<j>().valid = true;
            }
          });
          // write to the output pipe
          NextTOVisitPipe::PipeAt<krnl_id>::write(pipe_out);
        }
      }
      // Write back the terminating (done,valid) bits so that the next kernel knows this one is finished
      NextTOVisitPipe::PipeAt<krnl_id>::write(ReadtoFilterWritePipeData(true, false));
    });
  });
  return e;
}
// Shift register size must be statically determinable
constexpr int II_CYCLES = 8;


template <int krnl_id>
event submitExplorerKernelFilter(queue& q, unsigned int offsetL,unsigned int offsetH, visited_dt* usm_visited) {
  auto e = q.submit([&](auto& h) {
#ifdef FPGA_EMULATOR
    sycl::stream out(1024 * 1024 * 5, 256, h);
#endif
    h.template single_task<class ExploreNeighboursFilter<krnl_id>>(
        [=]() [[intel::kernel_args_restrict]] {
          device_ptr<visited_dt> DevicePtr_visited(usm_visited);
          [[intel::fpga_memory]] visited_dt lcl_visited[kMemoryDepth];
      // Create an on-chip memory with cache, storing integers, with a depth of 16 and a cache size of 4
#ifdef FPGA_EMULATOR
      /*out << "filter krnl_id pre-init " << krnl_id << ": " <<
      sycl::endl;*/
#endif
            unsigned int offsetD = offsetH - offsetL;
          [[intel::initiation_interval(1)]] for (int i = 0; i < offsetD; i += NUM_BITS_VISITED) {
            lcl_visited[index_for_bit(i)] = usm_visited[index_for_bit((i+offsetL))];
            // #ifdef FPGA_EMULATOR
      // out << "indxforbit[j] :" << index_for_bit(j) << "indxforbit[i]: " << index_for_bit((i+offsetL)) << 
      // sycl::endl;
// #endif
          }
     
          /* keep track of target nodes that
           *  have been loaded in parallel
           *  and need to be sorted into parallel banks
           *  TODO: consider implementing a ring buffer instead
           */
          [[intel::fpga_register]]  fpga_tools::NTuple<fpga_tools::ShiftReg<unsigned int, PARALLEL_FIFO_DEPTH>,NUM_BITS_VISITED> sort_fifo_cache;
          MyParCnt sort_cnt[NUM_BITS_VISITED] = {0};
          bool sort_full = false;
          bool sort_full_next = false;
          
          [[intel::fpga_register]] fpga_tools::NTuple<fpga_tools::ShiftReg<unsigned int, PARALLEL_FIFO_DEPTH>,NUM_BITS_VISITED> filter_fifo_cache;
          MyParCnt filter_cnt[NUM_BITS_VISITED] = {0};

          // received done signal; will be sent twice
          bool end_cond = false;
          bool end_cond_next = false;

          // drained all data from pipeline
          bool drain_cond = false;
          bool drain_cond_next = false;
          MyDrainCnt drain_cnt = 0;

#ifdef FPGA_EMULATOR
      /*out << "filter krnl_id init " << krnl_id << ": " <<
      sycl::endl;*/
#endif
          // while loop to allow for stalling individual stages
          //[[intel::initiation_interval(2)]]
          while (!drain_cond) {

            end_cond = end_cond_next;
            drain_cond = drain_cond_next;
            //sort_full = sort_full_next;

            // decisions in which direction a fifo is updated
            bool sort_inc_cond[NUM_BITS_VISITED] = {false};
            bool sort_dec_cond[NUM_BITS_VISITED] = {false};
            bool filter_inc_cond[NUM_BITS_VISITED] = {false};
            bool filter_dec_cond[NUM_BITS_VISITED] = {false};

            // pre-calculate update values
            MyParCnt sort_cnt_plus[NUM_BITS_VISITED];
            MyParCnt sort_cnt_minus[NUM_BITS_VISITED];
            fpga_tools::UnrolledLoop<0, NUM_BITS_VISITED>([&](auto j) {
              sort_cnt_plus[j]  = sort_cnt[j] + 1;
              sort_cnt_minus[j] = sort_cnt[j] - 1;
            });
            MyParCnt filter_cnt_plus[NUM_BITS_VISITED];
            MyParCnt filter_cnt_minus[NUM_BITS_VISITED];
            fpga_tools::UnrolledLoop<0, NUM_BITS_VISITED>([&](auto j) {
              filter_cnt_plus[j]  = filter_cnt[j] + 1;
              filter_cnt_minus[j] = filter_cnt[j] - 1;
            });

            // read phase
            if (!sort_full && !end_cond) {
              ReadtoFilterWritePipeData from_read_pipe_data = NextTOVisitPipe::PipeAt<krnl_id>::read();
              end_cond_next = from_read_pipe_data.done;
              // bool overall_valid = from_read_pipe_data.valid;

              fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
                bool valid = from_read_pipe_data.data.get<j>().valid;
                if (valid) {
                  unsigned int temp = from_read_pipe_data.data.get<j>().value;
                  // sort_fifo[j][sort_cnt[j]] = temp;
                  fpga_tools::UnrolledLoop<PARALLEL_FIFO_DEPTH>([&](auto cnt) {
                  if(sort_cnt[j] == cnt){
                  sort_fifo_cache.template get<j>().template ShiftRight<cnt>(temp);
                  }
                  });
                  sort_inc_cond[j] = true;
                }
              });
            }
            //  #ifdef FPGA_EMULATOR
            //             debug_parallel_fifo(out, "sort after read", sort_fifo, tempSortInc);
            
            //  #endif

            // sort phase
            fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
              bool break_cond = true;
              MyParCnt tempFilterCount = filter_cnt[j];
              if (tempFilterCount < PARALLEL_FIFO_DEPTH) {
                fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto k) {                  
                  if (sort_cnt[k] > 0 && break_cond && (j == customModulo(sort_fifo_cache.template get<k>().template Get<0>() / NUM_BITS_VISITED))) {
                  fpga_tools::UnrolledLoop<PARALLEL_FIFO_DEPTH>([&](auto cnt) {
                  if(tempFilterCount == cnt){
                    unsigned int t = sort_fifo_cache.template get<k>().template Get<0>();
                    // filter_fifo[j][tempFilterCount] = sort_fifo_cache.template get<k>().template Get<0>();
                     filter_fifo_cache.template get<j>().template ShiftRight<cnt>(t);
                  }
                  });
                  filter_inc_cond[j] = true;
                  sort_dec_cond[k] = true;
                  break_cond = false;
                  // break;
                  }
                  // }
                });
              }
            });

            // update sort fifo
            fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
              if (sort_dec_cond[j]) {
                sort_fifo_cache.template get<j>().ShiftLeft();
                // sort_fifo_cache.template get<j>().ShiftRight();
                // fpga_tools::UnrolledLoop<(PARALLEL_FIFO_DEPTH - 1)>([&](auto k) {
                  // for (int k = 0; k < PARALLEL_FIFO_DEPTH - 1; k++) {
                  // sort_fifo[j][k] = sort_fifo[j][k + 1];
                  
                // });
              }
            });

            // update sort status
            bool sort_full_tmp = false;
            // sort_empty = true;

            fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
              sort_full_tmp |= (sort_cnt[j] >= PARALLEL_FIFO_DEPTH-2);
              sort_cnt[j] = (sort_inc_cond[j] && !sort_dec_cond[j]) ? sort_cnt_plus[j] : ((!sort_inc_cond[j] && sort_dec_cond[j]) ? sort_cnt_minus[j] : sort_cnt[j]);
              // Load ith element into end of shift register
              // sort_empty &= (sort_cnt[j] == 0);
            });
            sort_full = sort_full_tmp;
           
            // filter phase          
            ReadtoFilterWritePipeData pipe_out(false, true);
            bool filter_hit = false;
            fpga_tools::UnrolledLoop<0, NUM_BITS_VISITED>([&](auto j) {
              pipe_out.data.get<j>().valid = false;
              if (filter_cnt[j] > 0) {
                // int val = filter_fifo[j][0];
                
                int val = filter_fifo_cache.template get<j>().template Get<0>();
                // No need for this since we have aligned the offsetL and offsetH with NUM_BITS_VISITED
                // int val2 = val - offsetL;
                int entry= ((val / (NUM_BITS_VISITED * NUM_BITS_VISITED))* NUM_BITS_VISITED + j);
                int enty = ((offsetL / double(NUM_BITS_VISITED * NUM_BITS_VISITED))  * NUM_BITS_VISITED);
                bool visited = bitcheck(lcl_visited[entry-enty], bit_to_toggle(val));
//                 #ifdef FPGA_EMULATOR
//       out << "krnl_id :" << krnl_id  <<",val :" << val << ", val2 : "<< val2 << ", entry: " << entry << ",offsetL = " << offsetL <<", new index: " << enty << 
//       sycl::endl;
// #endif
                if (!visited) {
                  pipe_out.data.get<j>().value = val;
                  pipe_out.data.get<j>().valid = true;
                  filter_hit = true;
                }

                filter_dec_cond[j] = true;
// #pragma unroll
                // for (int k = 0; k < PARALLEL_FIFO_DEPTH - 1; k++) {
                //   filter_fifo[j][k] = filter_fifo[j][k + 1];
                // }
                filter_fifo_cache.template get<j>().ShiftLeft();
              }
            });
            // write to pipe
            if (filter_hit) {
              FilterResultPipe::PipeAt<krnl_id>::write(pipe_out);
            }

            // update filter status
            fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
              filter_cnt[j] = (filter_inc_cond[j] && !filter_dec_cond[j]) ? filter_cnt_plus[j] : ((!filter_inc_cond[j] && filter_dec_cond[j]) ? filter_cnt_minus[j] : filter_cnt[j]);
            });

            // exit after delay that guarantees that all entries are drained
            if(drain_cnt == (NUM_BITS_VISITED+1) * PARALLEL_FIFO_DEPTH){
              drain_cnt = 0;
              drain_cond_next = true;
            }

            if(end_cond) drain_cnt++;
          }  // end while loop
          // Forward end signal tell the downstream kernel we are done producing data
          FilterResultPipe::PipeAt<krnl_id>::write(ReadtoFilterWritePipeData(true, false));
        });
  });
  return e;
}
#define ELEMENTS 2
template <int krnl_id>
event submitExplorerKernelWrite(queue& q, int numCols, visited_dt* usm_visited) {
  auto e = q.submit([&](auto& h) {
#ifdef FPGA_EMULATOR
    sycl::stream out(1024 * 1024 * 5, 256, h);
#endif
    h.template single_task<class ExploreNeighboursWrite<krnl_id>>([=]() [[intel::kernel_args_restrict]] {
      constexpr int kCacheSize = 8;
      constexpr int literals_per_cycle = 8;

      device_ptr<visited_dt> DevicePtr_visited(usm_visited);

      // Create 8 copies of on-chip memory with cache, will use 8 banks in parallel
      // fpga_tools::OnchipMemoryWithCache<visited_dt, kMemoryDepth, kCacheSize>
      // mem_cache(0);
      fpga_tools::NTuple<fpga_tools::OnchipMemoryWithCache<visited_dt, kMemoryDepth, kCacheSize>,
                         literals_per_cycle>
          mem_cache;
      // initialize the accumulators
      fpga_tools::UnrolledLoop<literals_per_cycle>([&](auto j) { mem_cache.template get<j>().init(0); });

      bool done = false;

      // while loop to allow for stalling individual stages
      // [[intel::initiation_interval(1)]]
      while (!done) {
        // read from the input pipe
        ReadtoFilterWritePipeData filter_data = FilterResultPipe::PipeAt<krnl_id>::read();

        // check if the producer is done
        done = filter_data.done;

        fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
          // last address bits used for bit in word, next address bits here removed because they
          // correspond to j after filter stage
          // bool valid = filter_data.data.get<j>().valid;
          int index_in_bank = filter_data.data.get<j>().value / (NUM_BITS_VISITED * NUM_BITS_VISITED);

          // Read the value from OnchipMemoryWithCache
          // if (valid) { // resulted in II = 3
            visited_dt cacheValue = mem_cache.template get<j>().read(index_in_bank);

            // Manipulate a specific bit (e.g., bit X)
            bitset(cacheValue, bit_to_toggle(filter_data.data.get<j>().value));  // Set bit X

            // Write the modified value back to memory
            mem_cache.template get<j>().write(index_in_bank, cacheValue);
          // }
        });

      }  // end of while loop

      for (int tid = 0; tid < numCols; tid += NUM_BITS_VISITED) {
        sycl::vec<visited_dt, 2> wrt_data;
        wrt_data[0] = usm_visited[index_for_bit(tid)];
        fpga_tools::UnrolledLoop<NUM_BITS_VISITED>([&](auto j) {
          if (j == (tid / NUM_BITS_VISITED) % NUM_BITS_VISITED)
            wrt_data[1] = mem_cache.template get<j>().read(tid / (NUM_BITS_VISITED * NUM_BITS_VISITED));
        });
        // VisitUpdate::PipeAt<0, krnl_id>::write(lcl_visited[index_for_bit(tid)]);
        fpga_tools::UnrolledLoop<2>([&](auto i) { PipeMatrix::PipeAt<i, krnl_id>::write(wrt_data); });
      }
    });
  });

  return e;
}

/* OR can we do something like

for( int i = krnl_id*start; i < krnl_id*end; i+=NUM_VISITED_)

Write each chunk from one of the kernels and then do round robin to another chunk from other kernel

                chunk from kernel
iteration #0  :  0____1_____2____3
iteration #1  :  3____0_____1____2
iteration #2  :  2____3_____0____1

usm_vistied_new
+---------+--------+------+------+
| krnl0 | krnl1 | krnl2 | krnl 3 |
+---------+--------+------+------+
*/

template <int krnl_id>
event submitLevelGenKernel(queue& q,
                           int numCols,
                           int* usmDist,
                           //  visited_dt* usm_visited_new,
                           int global_level) {
  auto e = q.submit([&](auto& h) {
#ifdef FPGA_EMULATOR
    sycl::stream out(1024 * 1024 * 2, 256, h);
#endif
    h.template single_task<class LevelGenerator<krnl_id>>([=]() [[intel::kernel_args_restrict]] {
      /* New Feature Implemented
       * Instead of checking bit by bit, just check if the 8 bits match,
       * if not just copy whole. It should save us lots of iterations of checks.
       */

      [[intel::initiation_interval(1)]] for (int tid = 0; tid < numCols; tid += NUM_BITS_VISITED) {
        sycl::vec<visited_dt, 2> rd_data;
        rd_data = PipeMatrix::PipeAt<0, krnl_id>::read();
        // rd_data[0] : old visited
        // rd_data[1] : visited_new

#pragma unroll
        for (int j = 0; j < NUM_BITS_VISITED; j++) {
          bool condition = (bitcheck(rd_data[1], bit_to_toggle(j)));
          if (condition) {
#ifdef FPGA_EMULATOR
            if (usmDist[tid + j] > 0) {
              out << "ERROR: overwriting usmDist[" << tid + j << " ], formerly " << usmDist[tid + j]
                  << " with " << global_level;
              out << ", previous visited_val was " << bitcheck(rd_data[1], bit_to_toggle(j))
                  << sycl::endl;
            }
#endif
            usmDist[tid + j] = global_level;
          }
        }
        // visited_dt old_visited_value = VisitUpdate::PipeAt<0, krnl_id>::read();
        VisitUpdate::PipeAt<krnl_id>::write(rd_data[0] | rd_data[1]);
      }
    });
  });
  return e;
}
template <int krnl_id>
event submitPipeGenKernel(queue& q, int numCols, unsigned int* usm_pipe, unsigned int* usm_pipe_size
                          // visited_dt* usm_visited,
                          // visited_dt* usm_visited_new
) {
  auto e = q.submit([&](auto& h) {
#ifdef FPGA_EMULATOR
    sycl::stream out(1024 * 1024 * 2, 256, h);
#endif
    h.template single_task<class PipeGenerator<krnl_id>>([=]() [[intel::kernel_args_restrict]] {
      int iter = 0;
      [[intel::fpga_register]] int temp[BUFFER_SIZE * 2];
      d_type3 temp_pos = 0;
      // [[intel::initiation_interval(1)]]
      for (int tid = 0; tid < numCols; tid += NUM_BITS_VISITED) {
        char condition[NUM_BITS_VISITED];
        d_type3 increment = 0;
        sycl::vec<visited_dt, 2> rd_data;
        visited_dt visited_new_val = 0;
        visited_dt arrayVisitedVal[NUM_COMPUTE_UNITS];
        fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
          rd_data = PipeMatrix::PipeAt<1, krnlID>::read();
          arrayVisitedVal[krnlID] = rd_data[1];
        });
        fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>(
            [&](auto krnlID) { visited_new_val |= arrayVisitedVal[krnlID]; });
#pragma unroll
        for (int j = 0; j < NUM_BITS_VISITED; j++) {
          // if(bit_compare(visited_new_val,visited_val,j)){
          // if(!bitcheck(visited_val,bit_to_toggle(j)) &&
          // bitcheck(visited_new_val,bit_to_toggle(j))){
          if (bitcheck(visited_new_val, bit_to_toggle(j))) {
            condition[j] = 1;
          } else {
            condition[j] = 0;
          }
          if (condition[j] && ((tid + j) < numCols)) {
            // usm_visited[index_for_bit(tid)][j]=1;
            // bitset(usm_visited[index_for_bit(tid)],bit_to_toggle(j));
            increment++;
          }
        }
        d_type3 current = 0;
#pragma unroll
        for (int j = 0; j < NUM_BITS_VISITED; j++) {
          if (condition[j] && ((tid + j) < numCols)) {
            temp[temp_pos + current] = tid + j;
            current++;
          }
        }
        temp_pos += increment;
        if (temp_pos >= NUM_BITS_VISITED) {
#pragma unroll
          for (int j = 0; j < NUM_BITS_VISITED; j++) {
            usm_pipe[iter + j] = temp[j];
          }
          iter += NUM_BITS_VISITED;
#pragma unroll
          for (int j = 0; j < NUM_BITS_VISITED; j++) {
            temp[j] = temp[j + NUM_BITS_VISITED];
          }
          temp_pos -= NUM_BITS_VISITED;
        }
        // check if the buffer is filled
        // write buffer back to usm_pipe
      }
      // dump remaining inside the buffer to the output usm_pipe.
      for (int rest = 0; rest < temp_pos; rest++) {
        usm_pipe[iter + rest] = temp[rest];
      }

      usm_pipe_size[0] = iter + temp_pos;
    });
  });
  return e;
}

event submitVisitedGenKernel(queue& q, int no_of_nodes, visited_dt* usm_visited) {
  auto e = q.single_task<MaskRemove>([=]() [[intel::kernel_args_restrict]] {
    [[intel::initiation_interval(1)]] for (int tid = 0; tid < no_of_nodes; tid += NUM_BITS_VISITED) {
      visited_dt temp_val = 0;
      visited_dt temp_single[NUM_COMPUTE_UNITS];
      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        temp_single[krnlID] = VisitUpdate::PipeAt<krnlID>::read();
        // visited_dt tempVis = usm_visited[index_for_bit(tid)];
      });

      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) { temp_val |= temp_single[krnlID]; });
      usm_visited[index_for_bit(tid)] = temp_val;
    }
  });

  return e;
}

// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void initUSMvec(queue& Q, T* usm_arr, std::vector<T>& arr) {
  Q.memcpy(usm_arr, arr.data(), arr.size() * sizeof(T));
}
//----------------------------------------------------------
//--breadth first search on FPGA
//----------------------------------------------------------
// This function instantiates the vector add kernel, which contains
// a loop that adds up the two summand arrays and stores the result
// into sum. This loop will be unrolled by the specified unroll_factor.
template <int unroll_factor>
std::vector<double> run_bfs_fpga(int numCols,
                                 std::vector<unsigned int>& source_inds,
                                 std::vector<unsigned int>& source_indptr,
                                 std::vector<visited_dt>& h_updating_graph_mask,
                                 std::vector<visited_dt>& h_graph_visited,
                                 std::vector<int>& h_dist,
                                 std::vector<unsigned int>& offset,
                                 std::vector<unsigned int>& offset_inds,
                                 int start_node,
                                 int numEdges,
                                 unsigned int exploredEdges,
                                 std::vector<unsigned int> &offset_visited) noexcept(false) {
#if defined(FPGA_EMULATOR)
  auto device_selector = sycl::ext::intel::fpga_emulator_selector_v;
#else
  auto device_selector = sycl::ext::intel::fpga_selector_v;
#endif

  auto prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
  // track timing information in ms
  std::vector<double> time_ms(MAX_NUM_LEVELS, 0.0);

  try {
    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(device_selector, fpga_tools::exception_handler, prop_list);

    // Print out the device information.
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

    std::vector<unsigned int> h_graph_pipe(numCols, 0);
    h_graph_pipe[0] = start_node;

    unsigned int pipe_size = 1;

    unsigned int* usm_nodes_start = malloc_device<unsigned int>(source_indptr.size(), q);

    int* usm_dist = malloc_device<int>(h_dist.size(), q);
    visited_dt* usm_visited = malloc_device<visited_dt>(h_graph_visited.size(), q);

    unsigned int* usm_edges = malloc_device<unsigned int>(source_inds.size(), q);

    unsigned int* usm_pipe = malloc_device<unsigned int>(h_graph_pipe.size(), q);

    unsigned int* usm_pipe_size = malloc_device<unsigned int>(1, q);

    initUSMvec(q, usm_edges, source_inds);
    initUSMvec(q, usm_nodes_start, source_indptr);
    initUSMvec(q, usm_dist, h_dist);
    // initUSMvec(q, usm_visited_new, h_updating_graph_mask);
    initUSMvec(q, usm_visited, h_graph_visited);
    initUSMvec(q, usm_pipe, h_graph_pipe);


    // Compute kernel execution time
    eventArray events;
    timesArray execTimes;

    std::array<event, NUM_COMPUTE_UNITS> eventsExploreRead;
    std::array<double, NUM_COMPUTE_UNITS> execTimesExploreRead;
    std::array<event, NUM_COMPUTE_UNITS> eventsExploreFilter;
    std::array<double, NUM_COMPUTE_UNITS> execTimesExploreFilter;
    std::array<event, NUM_COMPUTE_UNITS> eventsExploreWrite;
    std::array<double, NUM_COMPUTE_UNITS> execTimesExploreWrite;
    std::array<event, NUM_COMPUTE_UNITS> eventsLevelGen;
    std::array<double, NUM_COMPUTE_UNITS> execTimesLevelGen;

    int global_level = 1;
    std::vector<double> start_time(NUM_COMPUTE_UNITS, 0.0);
    double end_time = 0;

    for (int itr = 0; itr < MAX_NUM_LEVELS; itr++) {
    // for (int itr = 0; itr < 1; itr++) {
      if (pipe_size == 0) {
        std::cout << std::endl;
        printInformation("Metric", "Value");
        printDivider();
        printInformation("# Levels", itr);
        break;
      }
      int zero = 0;
      q.memcpy(usm_pipe_size, &zero, sizeof(unsigned int)).wait();

      // std::cout << "starting iteration " << itr << " with pipe size " << pipe_size << "\n";
      // eventsExploreFilter[0] = submitExplorerKernelFilter<0>(q, numCols, usm_visited);

      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        eventsExploreRead[krnlID] = submitExplorerKernelRead<krnlID>(
            q, pipe_size, usm_nodes_start, offset[krnlID], offset_inds[krnlID], usm_edges, usm_pipe);
      });
      // q.memcpy(usm_pipe_size, &pipe_size, sizeof(unsigned int)).wait();
      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        eventsExploreFilter[krnlID] = submitExplorerKernelFilter<krnlID>(q,offset_visited[krnlID],offset_visited[krnlID+1], usm_visited);
      });
      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        eventsExploreWrite[krnlID] = submitExplorerKernelWrite<krnlID>(q, numCols, usm_visited);
      });
      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        eventsLevelGen[krnlID] = submitLevelGenKernel<krnlID>(q, numCols, usm_dist, global_level);
      });
      events[static_cast<int>(kernelNames::PipeGenerator)] =
          submitPipeGenKernel<0>(q, numCols, usm_pipe, usm_pipe_size);
      events[static_cast<int>(kernelNames::MaskRemove)] =
          submitVisitedGenKernel(q, numCols, usm_visited);

      // #############################################################################################
      q.wait();

      // duration in milliseconds

      // #############################################################################################
      // q.memcpy(&pipe_size, usm_pipe_size, sizeof(unsigned int)).wait();
      q.memcpy(&pipe_size, usm_pipe_size, sizeof(unsigned int)).wait();
      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        execTimesExploreRead[krnlID] += GetExecutionTime(eventsExploreRead[krnlID]);
        start_time[krnlID] = eventsExploreRead[krnlID]
                                 .template get_profiling_info<info::event_profiling::command_start>();
      });
      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        execTimesExploreFilter[krnlID] += GetExecutionTime(eventsExploreFilter[krnlID]);
      });
      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        execTimesExploreWrite[krnlID] += GetExecutionTime(eventsExploreWrite[krnlID]);
      });

      fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>(
          [&](auto krnlID) { execTimesLevelGen[krnlID] += GetExecutionTime(eventsLevelGen[krnlID]); });
      // execTimes[static_cast<int>(kernelNames::LevelGenerator)] +=
      //     GetExecutionTime(events[static_cast<int>(kernelNames::LevelGenerator)]);
      execTimes[static_cast<int>(kernelNames::PipeGenerator)] +=
          GetExecutionTime(events[static_cast<int>(kernelNames::PipeGenerator)]);

      execTimes[static_cast<int>(kernelNames::MaskRemove)] +=
          GetExecutionTime(events[static_cast<int>(kernelNames::MaskRemove)]);
      end_time = events[static_cast<int>(kernelNames::MaskRemove)]
                     .get_profiling_info<info::event_profiling::command_end>();
      time_ms[itr] +=
          (end_time - *std::min_element(start_time.begin(), start_time.end())) * 1e-9;  // ns to s

      global_level++;
    }

    // copy usm_visited back to hostArray
    q.memcpy(&h_dist[0], usm_dist, h_dist.size() * sizeof(int));

    q.wait();
    // sycl::free(usm_nodes_start, q);
    // sycl::free(usm_nodes_end, q);
    // sycl::free(usm_edges, q);
    sycl::free(usm_dist, q);
    // sycl::free(usm_visited, q);
    // sycl::free(usm_visited_new, q);
    // sycl::free(usm_mask, q);
    double fpga_exec_time = std::accumulate(time_ms.begin(), time_ms.end(), 0.0);
    printInformation("Root Node", start_node);
    printInformation("# Nodes", numCols);
    printInformation("Total # Edges", numEdges);
    printInformation("Explored # Edges", exploredEdges);
    printInformation("Execution Time", std::to_string(fpga_exec_time) + " (s)");
    printInformation("Throughput", std::to_string((exploredEdges / (1000000 * fpga_exec_time))) + " (MTEPS)");

    std::cout << std::endl;
    printInformation("Kernel", "Wall-Clock Time (s)");
    printDivider();

    fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>(
        [&](auto krnlID) { printKernelTime("ExploreRead", krnlID, execTimesExploreRead); });
    fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>(
        [&](auto krnlID) { printKernelTime("ExploreFilter", krnlID, execTimesExploreFilter); });
    fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>(
        [&](auto krnlID) { printKernelTime("ExploreWrite", krnlID, execTimesExploreWrite); });
    std::cout << std::endl;
    printInformation("Kernel", "Wall-Clock Time (s)");
    printDivider();
    printSingleKernelTime("PipeGenerator", static_cast<int>(kernelNames::PipeGenerator), execTimes);
    fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>(
        [&](auto krnlID) { printKernelTime("LevelGen", krnlID, execTimesLevelGen); });
    std::cout << std::endl;
    printInformation("Kernel", "Wall-Clock Time (s)");
    printDivider();
    printSingleKernelTime("MaskRemove", static_cast<int>(kernelNames::MaskRemove), execTimes);
    std::cout << std::endl;

    // The queue destructor is invoked when q passes out of scope.
    // q's destructor invokes q's exception handler on any device exceptions.
  } catch (sycl::exception const& e) {
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
  return time_ms;
}