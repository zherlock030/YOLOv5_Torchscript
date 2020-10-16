#include "yolo.h"

using namespace std;

int main(){
    string model_path = "/home/zherlock/c++ example/object detection/files/yolov5s.torchscript";
    YOLO detector = YOLO();
    string img_path = "/home/zherlock/c++ example/object detection/files/test.png";
    detector.init_model(model_path);
    //detector.readmat(img_path);
    detector.inference(img_path);
}