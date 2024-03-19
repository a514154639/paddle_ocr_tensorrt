cmake_minimum_required(VERSION 3.10)

project(OCR)
set(PROJECT_NAME infer_OCR)

add_definitions(-std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(TensorRT_DIR "" CACHE PATH "location of tensorrt")
set(CUDA_TOOLKIT_ROOT_DIR "" CACHE PATH "location of cuda")
set(OpenCV_DIR "" CACHE PATH "location of opencv")


find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GFLAGS REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

#message(${TensorRT_LIBRARIES})
#message(${TensorRT_INCLUDE_DIRS})
message(${OpenCV_VERSION})

file(GLOB SRC_FILE src/*.cpp)

# 生成库
cuda_add_library(${PROJECT_NAME} SHARED src/ocr_out.cpp src/preprocess_op.cpp src/utility.cpp src/postprocess_op.cpp)
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${GFLAGS_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${TensorRT_LIBRARIES})

add_executable(ocr demo.cpp ${SRC_FILE})
#add_executable(ocr main.cpp ${SRC_FILE})

target_include_directories(ocr PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
        ${TensorRT_DIR}/samples/common)
target_link_libraries(ocr  ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${GFLAGS_LIBRARIES} dl)

add_definitions(-O2 -pthread)