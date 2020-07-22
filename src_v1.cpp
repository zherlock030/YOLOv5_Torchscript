//yolov5s的调用
//添加了图片的全部预处理

#include <memory>
#include <torch/script.h>
#include "torchvision/vision.h"
#include "torch/torch.h"
//#include "torchvision/PSROIAlign.h"
//#include "torchvision/PSROIPool.h"
//#include "torchvision/ROIAlign.h"
//#include "torchvision/ROIPool.h"
//#include "torchvision/empty_tensor_op.h"
//#include "torchvision/nms.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>
#include "sys/time.h"
#include <unistd.h>
#include <mutex>

using namespace std;
using namespace cv;
using namespace torch::indexing;

long time_in_ms(){
      struct timeval t;
      gettimeofday(&t, NULL);
      long time_ms = ((long)t.tv_sec)*1000+(long)t.tv_usec/1000;
      return time_ms;
}

cv::Mat letterbox(Mat img, int new_height = 640, int new_width = 640, Scalar color = (114,114,114), bool autos = true, bool scaleFill=false, bool scaleup=true){
  int width = img.cols;
  int height = img.rows;
  cout << "width is " << width << " and height is " << height << endl;
  float rh = float(new_height) / height;
  float rw = float(new_width) / width;
  float ratio;
  if(rw < rh){
      ratio = rw;}
  else{
      ratio = rh;}
  if (!scaleup){
    if(ratio >= 1.0){
      ratio = 1.0;
    }
  }
  cout << "ratio is " << ratio << " and rw is " << rw << " and rh is " << rh << endl;
  //new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r));
  int new_unpad_h = int(round(height * ratio));
  int new_unpad_w = int(round(width * ratio));
  cout << "new_h is " << new_unpad_h << " and new width is " << new_unpad_w << endl;
  int dh = new_height - new_unpad_h;
  int dw = new_width - new_unpad_w;
  cout << "dh is " << dh << " and dw is " << dw << endl;

  if(autos){
    dw = dw % 64;
    dh = dh % 64;
  }
  cout << "dh is " << dh << " and dw is " << dw << endl;
  
  dw /= 2;
  dh /= 2;//默认被二整除吧
  if( height != new_height or width != new_width){
    //Mat img = cv::resize(img, (new_unpad_h, new_unpad_w), interpolation=cv::INTER_LINEAR)
    resize(img, img, Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
  }
      //top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    //left, right = int(round(dw - 0.1)), int(round(dw + 0.1));
  int top = int(round(dh - 0.1));
  int bottom = int(round(dh + 0.1));
  int left = int(round(dw - 0.1));
  int right = int(round(dw + 0.1));

  //img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
  //img = cv::copyMakeBorder(img, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(114,114,114));
  cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(114,114,114));

  return img;
}

int main(int argc,char * argv[]){

  at::init_num_threads();//***zh,好像加上这句，速度有一点点提升

  string modelpath = "yolov5s.torchscript";
  cout << "before loading" << endl;
  long start = time_in_ms();
  torch::jit::script::Module model = torch::jit::load(modelpath);
  long end = time_in_ms();
  cout << "it took " << end - start << " ms to load the model" << endl;
  torch::jit::getProfilingMode() = false;
  torch::jit::getExecutorMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  
  //auto tensor_image = torch::rand((1, 3, 640,640));
  string img_path = "/home/zherlock/InstanceDetection/yolov5/test.png";
  Mat img = imread(img_path);
  //imshow("clock", img);
  
  // imshow之后必须有waitKey函数，否则显示窗内将一闪而过，不会驻留屏幕
  //waitKey(0);
  //cvtColor(img, img, CV_BGR2RGB);//***zh, bgr->rgb
  img = letterbox(img);//zh,,resize
  
  cout << "line 104, test number is " << int(img.at<Vec3b>(150, 200)[0]) << endl;
  cvtColor(img, img, CV_BGR2RGB);//***zh, bgr->rgb
  img.convertTo(img, CV_32FC3, 1.0f / 255.0f);//zh, 1/255
  cout << "line 105, " << img.size() << endl;
  cout << "line 111, width is "<< img.cols << " and height is " << img.rows << endl;

  auto tensor_img = torch::from_blob(img.data, {img.rows, img.cols, img.channels()});
  //tensor_img = tensor_img.index({Slice(), Slice(), Slice(None, None, -1)});//.permute({2, 0, 1});
  tensor_img = tensor_img.permute({2, 0, 1});
  //tensor_img = torch::unsqueeze(tensor_img, 0);
  tensor_img = tensor_img.unsqueeze(0);
  cout << "line 111, tensor size is " << tensor_img.sizes() << endl;//(1,3,384,640)
  cout << "line 118, test number is " << tensor_img.index({0,0,200,200}) << endl;

  return 0;


  start = time_in_ms();
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::rand({1, 3, 640, 640}));
  //inputs.emplace_back(tensor_image);
  torch::jit::IValue output = model.forward(inputs);
  end = time_in_ms();
  cout << "it took " << end - start << " ms to run the model once" << endl;
  
  

  return 0;
}
