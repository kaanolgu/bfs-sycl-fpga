#include <sycl/ext/intel/ac_types/ac_int.hpp>

using MyUint1 = ac_int<1, false>;
using MyUint4 = ac_int<4, false>;
using MyUInt32 = ac_int<32, false>;
using MyUInt64 = ac_int<64, false>;


#define DEBUG(x) std::cout <<" : "<< x << std::endl;




