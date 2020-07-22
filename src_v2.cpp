//yolov5s的调用
//增加模型运行后的后处理
//输入输出是搞得差不多了,现在c++输出和python输出是一致的

#include <memory>
#include <torch/script.h>
#include "torchvision/vision.h"
#include "torch/torch.h"
//#include "torchvision/PSROIAlign.h"
//#include "torchvision/PSROIPool.h"
//#include "torchvision/ROIAlign.h"
//#include "torchvision/ROIPool.h"
//#include "torchvision/empty_tensor_op.h"
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

using namespace std;
using namespace cv;
using namespace torch::indexing;

long time_in_ms(){
      struct timeval t;
      gettimeofday(&t, NULL);
      long time_ms = ((long)t.tv_sec)*1000+(long)t.tv_usec/1000;
      return time_ms;
}

at::Tensor box_area(at::Tensor box)
{
  return (box.index({2}) - box.index({0})) * (box.index({3}) - box.index({1}));
}
/*
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
*/

at::Tensor box_iou(at::Tensor box1, at::Tensor box2)
{
  //area1 = box_area(box1.t())
  //area2 = box_area(box2.t())
  cout << "line 69, box1.t() is " << box1.t() << endl;
  cout << "line 69, box2.t() is " << box2.t() << endl;
  at::Tensor area1 = box_area(box1.t());
  cout << "line 71, area1 is " << area1 << endl;
  at::Tensor area2 = box_area(box2.t());
  cout << "line 73, area2 is " << area2 << endl;
  //inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)


  at::Tensor temp1 =  torch::min( box1.index({Slice(), {None}, Slice(2,None)}), box2.index({Slice(), Slice(2,None)}));
  cout << "line 74, temp1 is " << temp1 << endl;

  at::Tensor temp2 = torch::max( box1.index({Slice(), {None}, Slice(None,2)}), box2.index({Slice(), Slice(None,2)}));
  cout << "line 75, temp2 is " << temp2 << endl;


  at::Tensor inter = ( torch::min( box1.index({Slice(), {None}, Slice(2,None)}), box2.index({Slice(), Slice(2,None)}))
  - torch::max( box1.index({Slice(), {None}, Slice(None,2)}), box2.index({Slice(), Slice(None,2)})) ).clamp(0).prod(2);
  cout << "line 77, inter is " << inter << endl;
  //return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
  return inter / (area1.index({Slice(), {None}}) + area2 - inter);

}

//Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
at::Tensor xywh2xyxy(at::Tensor x)
{
  at::Tensor y = torch::zeros_like(x);
  y.index({Slice(), 0}) = x.index({Slice(), 0}) - x.index({Slice(), 2}) / 2; // # top left x
  y.index({Slice(), 1}) = x.index({Slice(), 1}) - x.index({Slice(), 3}) / 2;
  y.index({Slice(), 2}) = x.index({Slice(), 0}) + x.index({Slice(), 2}) / 2;
  y.index({Slice(), 3}) = x.index({Slice(), 1}) + x.index({Slice(), 3}) / 2;

  return y;
}



at::Tensor non_max_suppression(at::Tensor pred, std::vector<string> labels, float conf_thres = 0.1, float iou_thres = 0.6,
                                   bool merge=false,  bool agnostic=false)
{
  cout << "line 38, input pred shape is " << pred.sizes() << endl;

  int nc = pred.sizes()[1] - 5;
  cout << "line 39, nc is " << nc << endl;

  at::Tensor xc = pred.index({Slice(None), 4}) > conf_thres; // # candidates,
  cout << "line 42, xc is " << xc.sizes() <<  endl; //xc size [15120], cpubooltype, 打印出来是0和1

  int min_wh = 2;
  int max_wh = 4096;
  int max_det = 300;
  float time_limit = 10.0;
  bool redundant = true;  //require redundant detections
  bool multi_label = true;
  at::Tensor output;

  //x = pred; xi = 0;
  at::Tensor x;
  x = pred.index({xc});
  cout << "line 55, x is " << x.sizes() <<  x.index({Slice(), Slice(None, 5)})  << endl;//shape (42,85)
  //match

  if(x.sizes()[0] <= 0){ //todo,这个判断有问题，如果完全没结果，如何判断这个空的tensor.
    return torch::zeros({1}); 
  }

  //x[:, 5:] *= x[:, 4:5]
  cout << "line 63, test num is " << x.index({0, Slice(5,10)}) << endl;
  cout << "line 64, test num is " << x.index({0, Slice(4,5)}) << endl;
  x.index({Slice(),Slice(5,None)}) *= x.index({Slice(), Slice(4,5)});
  cout << "line 65, test num is " << x.index({0, Slice(5,10)}) << endl;

  cout << "line 66, x is " << x.sizes() <<  x.index({Slice(), Slice(None, 5)})  << endl;

  //box = xywh2xyxy(x[:, :4])
  at::Tensor box = xywh2xyxy(x.index({Slice(), Slice(None,4)}));
  //unmatch
  cout << "line 95, box is " << box.sizes() << box << endl;

  if(true){ //here mulit label in python code
    //i, j = (x[:, 5:] > conf_thres).nonzero().t()
    auto temp = (x.index({Slice(), Slice(5,None)}) > conf_thres).nonzero().t();
    at::Tensor i = temp[0];
    at::Tensor j = temp[1];

    //cout << "line 99, temp is " << temp << endl;
    cout << "line 103, i is " << i << endl;
    cout << "line 104, j is " << j << endl;

    //x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    at::Tensor ba = box.index({i});
    cout << "line 109, ba shape is " << ba.sizes() << ba <<  endl;

    at::Tensor xa =  x.index({i,j+5,{None}});
    cout << "line 110, xa shape is " << xa.sizes() <<xa <<  endl;

    at::Tensor ja = j.index({Slice(),{None}});
    cout << "line 111, ja shape is " << ja.sizes() << ja.toType(torch::kFloat) << endl;
    x = torch::cat({  box.index({i}), x.index({i,j+5,{None}}), j.index({Slice(),{None}}).toType(torch::kFloat)  }, 1);
    //x = torch::cat((  box.index({i}), xa ), 1);
    cout << "line 108, x is " << x << endl;//shape (,)

    //at::Tensor ta = torch::ones({2,3});
    //at::Tensor tb = torch::ones({4,3});
    //at::Tensor tc = torch::cat( {ta, tb},0);
    //cout << "line 123, tc is " << tc.sizes() << endl;
  }

  if(x.sizes()[0] <= 0){
    return torch::zeros({1}); 
  }

  int n = x.sizes()[0];

  //c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
  at::Tensor c = x.index({Slice(), Slice(5,6)}) * max_wh;
  at::Tensor boxes = x.index({Slice(), Slice(None, 4)}) + c;
  at::Tensor scores = x.index({Slice(),4});

  at::Tensor i = nms(boxes, scores, iou_thres);
  cout << "line 136, i is " << i.sizes() << i  << endl;

  if(i.sizes()[0] > max_det){
    i = i.index({Slice(0,max_det)});
  }

  if(merge){
    if( n > 1 && n < 3000){
      cout << "line 195, boxes are " << boxes << endl;
      at::Tensor iou = box_iou(boxes.index({i}), boxes) > iou_thres;
      cout << "line 197, iou are " << iou.sizes() << iou << endl;
      //weights = iou * scores[None]  # box weights
      at::Tensor weights = iou * scores.index({ {None} });
      cout << "line 203, weights are " <<  weights << endl;

      cout << "line 203_1, before x is " << x.index({i, Slice(None, 4)}) << endl;
      //x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged 
      at::Tensor temp1 = torch::mm(weights, x.index({Slice(), Slice(None, 4)})).toType(torch::kFloat);
      cout << " line 204, temp1 is " << temp1 <<endl;
      at::Tensor temp2 = weights.sum(1, true);
      cout << " line 205, temp2 is " << temp2 <<endl;
      at::Tensor tempres = temp1 / temp2;
      cout << " line 205_1, tempres is " << tempres <<endl;
      x.index({i, Slice(None, 4)}) = torch::mm(weights, x.index({Slice(), Slice(None, 4)})).toType(torch::kFloat) / weights.sum(1, true); 
      x.index({i, Slice(None, 4)}) = tempres;
      //tensor.index_put_({Slice(1, None, 2)}, 1)
      x.index_put_({i, Slice(None, 4)}, tempres);
      cout << "line 206, x is " << x.index({i, Slice(None, 4)}) << endl;
      if(redundant){
        //i = i[iou.sum(1) > 1]  # require redundancy
        i = i.index({iou.sum(1) > 1});
        cout << "line 210, i is " << i << endl;
      }
    }
  }



  output = x.index({i}).squeeze();
  cout << "line 217, output is " << output << endl;

  


  return output;
}


cv::Mat letterbox(Mat img, int new_height = 640, int new_width = 640, Scalar color = (114,114,114), bool autos = true, bool scaleFill=false, bool scaleup=true){
  int width = img.cols;
  int height = img.rows;
  //cout << "width is " << width << " and height is " << height << endl;
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
  //cout << "ratio is " << ratio << " and rw is " << rw << " and rh is " << rh << endl;
  //new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r));
  int new_unpad_h = int(round(height * ratio));
  int new_unpad_w = int(round(width * ratio));
  //cout << "new_h is " << new_unpad_h << " and new width is " << new_unpad_w << endl;
  int dh = new_height - new_unpad_h;
  int dw = new_width - new_unpad_w;
  //cout << "dh is " << dh << " and dw is " << dw << endl;

  if(autos){
    dw = dw % 64;
    dh = dh % 64;
  }
  //cout << "dh is " << dh << " and dw is " << dw << endl;
  
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
  //cout << "before loading" << endl;
  long start = time_in_ms();
  torch::jit::script::Module model = torch::jit::load(modelpath);
  long end = time_in_ms();
  //cout << "it took " << end - start << " ms to load the model" << endl;
  torch::jit::getProfilingMode() = false;
  torch::jit::getExecutorMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  
  //auto tensor_image = torch::rand((1, 3, 640,640));
  string img_path = "/home/zherlock/InstanceDetection/yolov5_old/test.png";
  Mat img = imread(img_path);
  //imshow("clock", img);
  
  // imshow之后必须有waitKey函数，否则显示窗内将一闪而过，不会驻留屏幕
  //waitKey(0);
  //cvtColor(img, img, CV_BGR2RGB);//***zh, bgr->rgb
  img = letterbox(img);//zh,,resize
  
  //cout << "line 104, test number is " << int(img.at<Vec3b>(150, 200)[0]) << endl;
  cvtColor(img, img, CV_BGR2RGB);//***zh, bgr->rgb
  img.convertTo(img, CV_32FC3, 1.0f / 255.0f);//zh, 1/255
  //cout << "line 105, " << img.size() << endl;
  //cout << "line 111, width is "<< img.cols << " and height is " << img.rows << endl;

  auto tensor_img = torch::from_blob(img.data, {img.rows, img.cols, img.channels()});
  //tensor_img = tensor_img.index({Slice(), Slice(), Slice(None, None, -1)});//.permute({2, 0, 1});
  tensor_img = tensor_img.permute({2, 0, 1});
  //tensor_img = torch::unsqueeze(tensor_img, 0);
  tensor_img = tensor_img.unsqueeze(0);
  cout << "line 111, tensor size is " << tensor_img.sizes() << endl;//(1,3,384,640)
  //cout << "line 118, test number is " << tensor_img.index({0,0,300,200}) << endl;

  //return 0;
  //tensor_img = torch::ones({1, 3, 640,640});
  start = time_in_ms();
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor_img);
  torch::jit::IValue output = model.forward(inputs);
  end = time_in_ms();
  cout << "line 132, finish computing." << endl;
  //cout << "it took " << end - start << " ms to run the model once" << endl;

  //auto op1 = output.toTuple();
  //cout << "line 171, to tuple." << endl;
  //auto dets1 = out1->elements().at(0).toList().get(0);
  //auto op2 = op1->elements().at(0);
  //cout << "line 174, get item 0 " << endl;

  //at::Tensor op = op2.toTensor();
  //cout << "line 175, totensor " << endl;

  at::Tensor op = output.toTuple()->elements().at(0).toTensor();

  //auto op = output.toList().get(0).toTensor();
  //auto op = output.toTensor();


  cout << "line 132, op[0] is " << op.sizes() << endl;

  //cout << "line 133, test num is " << op.index({0,1000,Slice(None)}) << endl;

  //cout << "line 134, op_size[0] is " << op.sizes()[0] << " and op_size[-1] is " <<  op.sizes()[4] << endl; 

  //cout << "line 135, test num is " << op.index({0,0,0,0,Slice(0,5)}) << endl;

  //op = op.view({op.sizes()[0], -1 ,op.sizes()[4] });
  op = op.view({-1 ,op.sizes()[2] });
  cout << "line 144, op[0] is " << op.sizes() << endl;

  //for(int i = 0; i < op.sizes()[0] ; i ++ ){
    //cout << i << " row " << op.index({i,Slice(0,5) }) << endl;
  //}

  ifstream f;
  std::vector<string> labels;
  string labelpath = "/home/zherlock/InstanceDetection/yolov5_old/labels.txt";
  f.open(labelpath);
  string str;
  while (std::getline(f,str)){
    labels.push_back(str);
  }
  //cout << "we get " << labels.size() << " labels" << endl;


  for(int i = 0 ; i < 20 ; i++){
    //cout << "label " << i << " is " << labels[i] << endl;
  }

  op = non_max_suppression(op, labels, 0.4, 0.5, true,  false);

  

  int op_class = op.index({5}).item().toInt();
  cout << " op_class is " << op_class << ", and it's " << labels[op_class] << endl;

  int x1 = op.index({0}).item().toInt();
  int y1 = op.index({1}).item().toInt();

  int x2 = op.index({2}).item().toInt();
  int y2 = op.index({3}).item().toInt();

  cout << " the bounding box is x1 = " << x1 << ", and y1 = " << y1 << ", and x2 = " << x2 << ", and y2 = " << y2 << endl;




  
  

  return 0;
}
