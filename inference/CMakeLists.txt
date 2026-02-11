cmake_minimum_required(VERSION 3.18)
project(inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "~/libtorch")

find_package(Torch REQUIRED)

add_executable(inference inference.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
set_property(TARGET inference PROPERTY CXX_STANDARD 17)