set(SOURCE_FILE host.cpp)
set(TARGET_NAME bfs)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(SIMULATOR_TARGET ${TARGET_NAME}.fpga_sim)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

# FPGA device selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA device ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on device selection.")
else()
    message(STATUS "Configuring the design to run on FPGA device ${FPGA_DEVICE}")
endif()

# These are Windows-specific flags:
# 1. /EHsc This is a Windows-specific flag that enables exception handling in host code
# 2. /Qactypes Include ac_types headers and link against ac_types emulation libraries
if(WIN32)
    set(WIN_FLAG "/EHsc")
    set(AC_TYPES_FLAG "/Qactypes")
else()
    set(AC_TYPES_FLAG "-qactypes")
endif()

# NUMBER OF COMPUTE UNITS
if(NOT DEFINED NUM_COMPUTE_UNITS)
    set(NUM_COMPUTE_UNITS "4")
    message(STATUS "NUM_COMPUTE_UNITS was not specified.\
                    \nConfiguring the design to run on the default ${NUM_COMPUTE_UNITS} \
                    \nPlease refer to the README for information on NUM_COMPUTE_UNITS selection.")
else()
    message(STATUS "Configuring the design to run on ${NUM_COMPUTE_UNITS} Compute Units")
endif()
# NUMBER OF COMPUTE UNITS
if(NOT DEFINED K_MEMORY_CACHE)
    set(K_MEMORY_CACHE "131072")
    message(STATUS "K_MEMORY_CACHE was not specified.\
                    \nConfiguring the design to run on the default ${K_MEMORY_CACHE} \
                    \nPlease refer to the README for information on K_MEMORY_CACHE selection.")
else()
    message(STATUS "Configuring the design to run on ${K_MEMORY_CACHE} Compute Units")
endif()
# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR -DNUM_COMPUTE_UNITS=${NUM_COMPUTE_UNITS} -DK_MEMORY_CACHE=${K_MEMORY_CACHE}")
set(EMULATOR_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG}")
set(HARDWARE_COMPILE_FLAGS "-Wall ${WIN_FLAG} -fsycl -fintelfpga ${AC_TYPES_FLAG} -O3 -DNUM_COMPUTE_UNITS=${NUM_COMPUTE_UNITS}  -DK_MEMORY_CACHE=${K_MEMORY_CACHE}")
set(HARDWARE_LINK_FLAGS "-fsycl -fintelfpga ${AC_TYPES_FLAG} -Xsauto-pipeline -Xshardware -Xssfc-exit-fifo-type=zero-latency  -Xsnum-reorder=1 -Xstarget=${FPGA_DEVICE} -Xsprofile ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR fpga_compile.cpp -o fpga_compile.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -DFPGA_EMULATOR -o fpga_compile.cpp.o -c fpga_compile.cpp
#    [link]    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} fpga_compile.cpp.o -o fpga_compile.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_include_directories(${EMULATOR_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${EMULATOR_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${EMULATOR_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/benchmarks/${NUMBER_CU}cu)
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early fpga_compile.cpp -o fpga_compile_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${FPGA_EARLY_IMAGE} ${SOURCE_FILE})
target_include_directories(${FPGA_EARLY_IMAGE} PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DCACHE_ENABLED")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsread-only-cache-size=4096 -Xsno-interleaving=default -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<FPGA_DEVICE> fpga_compile.cpp -o fpga_compile.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -o fpga_compile.cpp.o -c fpga_compile.cpp
#   [link]    icpx -fsycl -fintelfpga ${AC_TYPES_FLAG} -Xshardware -Xstarget=<FPGA_DEVICE> fpga_compile.cpp.o -o fpga_compile.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_include_directories(${FPGA_TARGET} PRIVATE ../include)
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET}")
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See C++SYCL_FPGA/GettingStarted/fast_recompile for details.



