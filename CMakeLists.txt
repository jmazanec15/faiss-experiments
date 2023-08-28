#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.23.1)

project(FaissExperimental)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TARGET_HELLO_FAISS hellofaiss)

set(BUILD_TESTING OFF)          # Avoid building faiss tests
set(BLA_STATIC ON)              # Statically link BLAS
set(FAISS_OPT_LEVEL generic)    # Keep optimization level generic

if (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
    endif()
endif()

find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BLAS REQUIRED)
enable_language(Fortran)
find_package(LAPACK REQUIRED)

# Check if faiss exists
find_path(FAISS_REPO_DIR NAMES faiss PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)

# If not, pull the updated submodule
if (NOT EXISTS ${FAISS_REPO_DIR})
    message(STATUS "Could not find faiss. Pulling updated submodule.")
    execute_process(COMMAND git submodule update --init -- external/faiss WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

set(FAISS_ENABLE_GPU OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/faiss EXCLUDE_FROM_ALL)

add_executable(${TARGET_HELLO_FAISS} ${CMAKE_CURRENT_SOURCE_DIR}/cpp/hello_faiss.cpp)
target_link_libraries(${TARGET_HELLO_FAISS} faiss OpenMP::OpenMP_CXX)
target_include_directories(${TARGET_HELLO_FAISS} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)
set_target_properties(${TARGET_HELLO_FAISS} PROPERTIES POSITION_INDEPENDENT_CODE ON)
