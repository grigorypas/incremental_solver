cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(IncrementalSolver)

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
  include(ChooseMSVCCRT)

  # MLIR
  find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

  include(AddLLVM)
  include(TableGen) # required by AddMLIR
  include(AddMLIR)

  include(HandleLLVMOptions)

  include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_INCLUDE_DIR}")
  include_directories(${MLIR_INCLUDE_DIRS})
  link_directories("${LLVM_LIBRARY_DIR}")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. ")
endif()

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

add_subdirectory(lib)
add_subdirectory(unittest)
add_subdirectory(include/IncrementalSolver)