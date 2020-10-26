#ifndef YOLO_H
#define YOLO_H

#include <memory>
#include <torch/script.h>
#include "torchvision/vision.h"
#include "torch/torch.h"
#include "torchvision/nms.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>
#include "sys/time.h"
#include <unistd.h>
#include <mutex>
#include <string>

class YOLO{
public:
    YOLO(std::string ModelPath);  // 
    at::Tensor box_area(at::Tensor box);  // 返回矩形面积
    at::Tensor box_iou(at::Tensor box1, at::Tensor box2);  // 返回两个矩形iou
    at::Tensor xywh2xyxy(at::Tensor x);  //
    at::Tensor non_max_suppression(at::Tensor pred, std::vector<std::string> labels, float conf_thres, float iou_thres,
                                   bool merge,  bool agnostic);  // nms
    at::Tensor make_grid(int nx, int ny);  //
    cv::Mat letterbox(cv::Mat img, int new_height, int new_width, cv::Scalar color, bool autos, bool scaleFill, bool scaleup);  //modify img shape
    at::Tensor clip_coords(at::Tensor boxes, auto img_shape);  // 
    float max_shape(int shape[], int len);  //
    at::Tensor scale_coords(int img1_shape[], at::Tensor coords, int img0_shape[]);  // 
    void readmat(std::string ImgPath);
    void readmat(cv::Mat &mat);
    void init_model(std::string ModelPath);
    int inference(std::string ImgPath);
    int inference(cv::Mat &mat);
    int inference();
    void show();
    void preprocess();


protected:
    float conf_thres;
    float iou_thres;
    bool merge;
    bool agnostic;
    std::string model_path;
    std::string img_path;
    cv::Mat img;
    cv::Mat im0;
    //auto tensor_img;
    at::Tensor tensor_img;
    std::vector<torch::jit::IValue> inputs;
    torch::jit::script::Module model;
    long start, end;
    at::Tensor anchor_grid;
    int ny, nx;
    int na = 3;int no = 85;int bs = 1;
    at::Tensor grid;
    at::Tensor op;
    at::Tensor y_0, y_1, y_2, y;
    std::vector<std::string> labels;
};


long time_in_ms(); // 返回当前时间
std::vector<std::string> split(const std::string& str, const std::string& delim); // 将字符串根据delim分割为vector
#endif

