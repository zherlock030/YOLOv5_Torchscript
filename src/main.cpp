#include "yolo.h"

using namespace std;
using namespace cv;

int main(){
    string model_path = "/home/zherlock/c++ example/object detection/files/yolov5s.torchscript";
    YOLO detector = YOLO(model_path);
    string img_path = "/home/zherlock/c++ example/object detection/files/test.png";
    Mat img = imread(img_path);
    //detector.inference(img_path);
    detector.inference(img);
    
}