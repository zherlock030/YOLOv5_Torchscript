cmake_minimum_required(VERSION 2.8)
project(yolov5s)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_PREFIX_PATH "/home/zherlock/SLAM/libtorch/libtorch-abi")
set(OpenCV_DIR "/home/zherlock/opencv-2.4.9/build/")
find_package(OpenCV 2.4.9 QUIET)

include_directories("/home/zherlock/torchvision-abi/include")
include_directories("/home/zherlock/SLAM/libtorch/libtorch-abi/include/torch/csrc/api/include/")
include_directories("/home/zherlock/SLAM/libtorch/libtorch-abi/include/")
include_directories("/home/zherlock/anaconda3/envs/detectron2/include/python3.7m/")

find_package(Torch REQUIRED)

add_library(${PROJECT_NAME} SHARED
/home/zherlock/torchvision-abi/include/torchvision/nms.cpp
)


add_executable(yolo src.cpp)
target_link_libraries(yolo 
${PROJECT_NAME}
${OpenCV_LIBS}
${TORCH_LIBRARIES}
/home/zherlock/torchvision-abi/lib/libtorchvision.so)

#***zh
set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++14)
