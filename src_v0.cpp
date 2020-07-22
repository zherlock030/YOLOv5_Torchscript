//yolov5s的调用
//最基础版本，无输出

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

long time_in_ms(){
      struct timeval t;
      gettimeofday(&t, NULL);
      long time_ms = ((long)t.tv_sec)*1000+(long)t.tv_usec/1000;
      return time_ms;
}

//int gettimeofday(struct timeval *tv, struct timezone *tz);

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

  


  for(int i = 1; i < 10; i++){
  auto tensor_image = torch::rand((1, 3, 640,640));
  start = time_in_ms();
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::rand({1, 3, 640, 640}));
  //inputs.emplace_back(tensor_image);
  torch::jit::IValue output = model.forward(inputs);
  end = time_in_ms();
  cout << "it took " << end - start << " ms to run the model once" << endl;
  }
  

  return 0;
}
