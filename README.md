# YOLOv5_Torchscript
C++ code for running a yolov5s model.

My steps are, 
1.generating a torchscript file using export.py in yolov5.
when u run export.py, Make sure u modify the detect layer to make it return the inputed list x, then we will implement detect layer in c++.

2.write codes for image pre_processing, detect layer, and nms.

