#include "constexpr_math.hpp"
#include "onchip_memory_with_cache.hpp"  // DirectProgramming/C++SYCL_FPGA/include
#include "pipe_utils.hpp"                // DirectProgramming/C++SYCL_FPGA/include
#include "unrolled_loop.hpp"             // DirectProgramming/C++SYCL_FPGA/include
#include "tuple.hpp"
#include "StreamingData.hpp"
//
#define MAX_NUM_LEVELS 100           // number of times we run the for loop to iterate levels
#define NUM_BITS_VISITED 8
#define LOG_NUM_BITS_VISITED 3

// These values are retrieved from the CMake
// #define NUM_COMPUTE_UNITS 8          // Number of compute units 

#define bitset(byte, nbit) ((byte) |= (1 << (nbit)))
#define bitclear(byte, nbit) ((byte) &= ~(1 << (nbit)))
#define bitflip(byte, nbit) ((byte) ^= (1 << (nbit)))
#define bitcheck(byte, nbit) (bool((byte) & (1 << (nbit))))
#define index_for_bit(val) (val / NUM_BITS_VISITED)
#define bit_to_toggle(val) (val % NUM_BITS_VISITED)
#define bit_compare(byte1, byte2, nbit) (!(bitcheck(byte1, nbit) ^ !bitcheck(byte2, nbit)))
#define check_equal(byte1, byte2) ((((byte1 + byte2) % 2) + 1) % 2)  // equal = 1, not equal = 0


// output Print Related 
const char divider = '-';
const char separator = ' ';
const int nameWidth = 21;
const int numWidth = 25;
#define printKernelTime(name,id,kernelT) ( std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << name +std::to_string(krnlID) << "| " << std::setw(numWidth) << std::setfill(separator) << std::to_string(kernelT[krnlID]) + " (s) "<< "|" << std::endl)
#define printSingleKernelTime(name,index,kernelT) ( std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << name << "| " << std::setw(numWidth) << std::setfill(separator) << std::to_string(kernelT[index]) + " (s) "<< "|" << std::endl)
#define printInformation(name,val) ( std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << name << "| " << std::setw(numWidth) << std::setfill(separator) << val<< "| " << std::endl)
#define printDivider() ( std::cout << "| " << std::string((nameWidth-1), divider) <<" | "<<  std::string((numWidth-1), divider) << " |\n")


using MyUInt32 = ac_int<32, false>;
using MyUInt64 = ac_int<64, false>;
int numRows, numCols, numNonz;


constexpr int BUFFER_SIZE = 8;
constexpr int PARALLEL_FIFO_DEPTH = 8;
constexpr int WRITE_FIFO_SIZE = BUFFER_SIZE * 4;
using MyParCnt = ac_int<fpga_tools::BitsForMaxValue<PARALLEL_FIFO_DEPTH+1>(), false>;
// using MyParCnt = char; 
using MyDrainCnt = ac_int<fpga_tools::BitsForMaxValue<(NUM_BITS_VISITED+1) * (PARALLEL_FIFO_DEPTH+1)>(), false>;
//using MyWriteCnt = ac_int<fpga_tools::BitsForMaxValue<WRITE_FIFO_SIZE>(), false>;

using MyUint1 = char;
// using d_type3 = ac_int<log2(BUFFER_SIZE*2), false>;
using d_type3 = char;
// using visited_dt = ac_int<NUM_BITS_VISITED,false>;
using visited_dt = short;
