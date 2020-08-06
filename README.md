# YOLOv5_Torchscript
C++ code for running a yolov5s model.

My steps are, 

1.generating a torchscript file using export.py in yolov5.
when u run export.py, Make sure u modify the detect layer to make it return the inputed list x, then we will implement detect layer in c++.

2.write codes for image pre_processing, detect layer, and nms.

src.cpp is a clean one that u can compile and run.

nms.cpp and nms.h are offered, I put them on the path /torchvision/include/torchvision/

my implementation is not perfect since I'm not familiar to libtorch, and I fixed some parameters in these functions.

IMAGES IN DIFFERENT SHAPES CAN BE FED LIEK THAT IN PYTHON.

