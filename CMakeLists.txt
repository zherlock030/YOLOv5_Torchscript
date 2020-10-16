cmake_minimum_required(VERSION 2.8)
project(yolov5s)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories("/home/zherlock/torchvision-abi/include/")
include_directories("/home/zherlock/torchvision-abi/include/torchvision")
include_directories("/home/zherlock/SLAM/libtorch/libtorch-abi/include/torch/csrc/api/include/")
include_directories("/home/zherlock/SLAM/libtorch/libtorch-abi/include/")
include_directories("/home/zherlock/anaconda3/envs/detectron2/include/python3.7m/")
include_directories(
${PROJECT_SOURCE_DIR}/include
)


set(CMAKE_PREFIX_PATH "/home/zherlock/SLAM/libtorch/libtorch-abi")
set(OpenCV_DIR "/home/zherlock/opencv-2.4.9/build/")
find_package(OpenCV 2.4.9 QUIET)
find_package(Torch REQUIRED)

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})



target_link_libraries(
${CMAKE_PROJECT_NAME}
${OpenCV_LIBS}
${TORCH_LIBRARIES}
/home/zherlock/torchvision-abi/lib/libtorchvision.so)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++14)


#add_library(nms SHARED
#${PROJECT_SOURCE_DIR}/include/src/nms.cpp
#)
#add_executable(yolo src.cpp)
