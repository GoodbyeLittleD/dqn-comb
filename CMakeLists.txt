cmake_minimum_required(VERSION 3.16.0)
#==================
# Start project
#==================
project(dqn-comb)


#==================
# C++ Standard
#==================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#==================
# Archtecture and optimization settings
#==================
set(TARGET_ARCH "x64")
set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3")

#==================
# Find third-party packages
#==================
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


#==================
# Include headers
#==================
include_directories(src)

#==================
# Link libraries
#==================
link_libraries(${TORCH_LIBRARIES})

#===================
# Compile executables
#===================
set(CMAKE_BUILD_TYPE "Release") # Set build type to Release
add_executable(torrentcli src/game.cpp src/gendata_3.cpp)
add_executable(test src/game.cpp src/test.cpp)
add_executable(benchmark src/game.cpp src/benchmark.cpp)

#===================
# Post build actions
#===================
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET benchmark
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:benchmark>)
endif (MSVC)
