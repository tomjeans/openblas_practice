cmake_minimum_required (VERSION 3.8)
project(affine_test VERSION 1.0.0)

find_package (OpenCV REQUIRED)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(${BLAS_INCLUDE_DIRS})
    
endif(BLAS_FOUND)


include_directories (${OpenCV_INCLUDE_DIRS})
add_executable (affine_test test2.cc)

target_link_libraries (affine_test ${OpenCV_LIBS})
target_link_libraries(affine_test ${BLAS_LIBRARIES})