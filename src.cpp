//yolov5s的调用
//删除冗余注释，优化代码,记录下时间消耗
//原始，load:129;pre_process:1;model:365;nms:2
//init_threads:365->313
//jit::设置，365->249
//似乎是jit的设置比较有用，两个同时设置后，也没有新的提升，和设置jit的性能一样

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

using namespace std;
using namespace cv;
using namespace torch::indexing;


vector<string> split(const string& str, const string& delim) {
	vector<string> res;
	if("" == str) return res;
	//先将要切割的字符串从string类型转换为char*类型
	char * strs = new char[str.length() + 1] ; //不要忘了
	strcpy(strs, str.c_str());

	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);
	while(p) {
		string s = p; //分割得到的字符串转换为string类型
		res.push_back(s); //存入结果数组
		p = strtok(NULL, d);
	}
	return res;
}

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

at::Tensor box_iou(at::Tensor box1, at::Tensor box2)
{
  at::Tensor area1 = box_area(box1.t());
  at::Tensor area2 = box_area(box2.t());

  at::Tensor inter = ( torch::min( box1.index({Slice(), {None}, Slice(2,None)}), box2.index({Slice(), Slice(2,None)}))
  - torch::max( box1.index({Slice(), {None}, Slice(None,2)}), box2.index({Slice(), Slice(None,2)})) ).clamp(0).prod(2);
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
  int nc = pred.sizes()[1] - 5;
  at::Tensor xc = pred.index({Slice(None), 4}) > conf_thres; // # candidates,

  int min_wh = 2;
  int max_wh = 4096;
  int max_det = 300;
  float time_limit = 10.0;
  bool redundant = true;  //require redundant detections
  bool multi_label = true;
  at::Tensor output;

  at::Tensor x;
  x = pred.index({xc});

  try{
    at::Tensor temp  = x.index({0});
  }catch(...){
    cout << "no objects, line 138 " << endl; 
    at::Tensor temp;
    return temp;
  }

 x.index({Slice(),Slice(5,None)}) *= x.index({Slice(), Slice(4,5)});
  at::Tensor box = xywh2xyxy(x.index({Slice(), Slice(None,4)}));
  if(true){ //here mulit label in python code
    //i, j = (x[:, 5:] > conf_thres).nonzero().t()
    auto temp = (x.index({Slice(), Slice(5,None)}) > conf_thres).nonzero().t();
    at::Tensor i = temp[0];
    at::Tensor j = temp[1];
    x = torch::cat({  box.index({i}), x.index({i,j+5,{None}}), j.index({Slice(),{None}}).toType(torch::kFloat)  }, 1);
}

  try{
    at::Tensor temp  = x.index({0});
  }catch(...){
    cout << "no objects, line 187 " << endl; 
    at::Tensor temp;
    return temp;
  }

  int n = x.sizes()[0];
  at::Tensor c = x.index({Slice(), Slice(5,6)}) * max_wh;
  at::Tensor boxes = x.index({Slice(), Slice(None, 4)}) + c;
  at::Tensor scores = x.index({Slice(),4});

  at::Tensor i = nms(boxes, scores, iou_thres);

  if(i.sizes()[0] > max_det){
    i = i.index({Slice(0,max_det)});
  }

  if(merge){
    if( n > 1 && n < 3000){
      at::Tensor iou = box_iou(boxes.index({i}), boxes) > iou_thres;
      at::Tensor weights = iou * scores.index({ {None} });
     at::Tensor temp1 = torch::mm(weights, x.index({Slice(), Slice(None, 4)})).toType(torch::kFloat);
      at::Tensor temp2 = weights.sum(1, true);
      at::Tensor tempres = temp1 / temp2;
      x.index_put_({i, Slice(None, 4)}, tempres);
      if(redundant){
        i = i.index({iou.sum(1) > 1});
      }
    }
  }
  output = x.index({i});
  return output;
}


cv::Mat letterbox(Mat img, int new_height = 640, int new_width = 640, Scalar color = (114,114,114), bool autos = true, bool scaleFill=false, bool scaleup=true){
  int width = img.cols;
  int height = img.rows;
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
  int new_unpad_h = int(round(height * ratio));
  int new_unpad_w = int(round(width * ratio));
  int dh = new_height - new_unpad_h;
  int dw = new_width - new_unpad_w;

  if(autos){
    dw = dw % 64;
    dh = dh % 64;
  }
  
  dw /= 2;
  dh /= 2;//默认被二整除吧
  if( height != new_height or width != new_width){
    resize(img, img, Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
  }
  int top = int(round(dh - 0.1));
  int bottom = int(round(dh + 0.1));
  int left = int(round(dw - 0.1));
  int right = int(round(dw + 0.1));

  cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(114,114,114));

  return img;
}


at::Tensor make_grid(int nx = 20, int ny = 20){
//yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
//        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

  std::vector<at::Tensor> temp = torch::meshgrid({torch::arange(ny), torch::arange(nx)});
  at::Tensor yv = temp[0];
  at::Tensor xv = temp[1];
  //cout << "line 202, " << temp[0].sizes() <<endl;
  at::Tensor retu = torch::stack({xv, yv},2).view({1,1,ny,nx,2}).toType(torch::kFloat);
  //cout << "line 202, " << retu.sizes() <<endl;
  //cout << "line 202, retu is " << retu.index({0,0,0,0,0}) <<endl;
  //return torch::ones_like(retu);
  return retu;
}

at::Tensor clip_coords(at::Tensor boxes, auto img_shape){
  /*
  # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
  */
  boxes.index({Slice(), 0}).clamp_(0, img_shape[1]);
  boxes.index({Slice(), 1}).clamp_(0, img_shape[0]);
  boxes.index({Slice(), 2}).clamp_(0, img_shape[1]);
  boxes.index({Slice(), 3}).clamp_(0, img_shape[0]);
  return boxes;
}

float max_shape(int shape[], int len){
  float max = -10000;
  for(int i = 0; i < len; i++){
    if (shape[i] > max){
      max = shape[i];
    }
  }
  return max;
}


at::Tensor scale_coords(int img1_shape[], at::Tensor coords, int img0_shape[]){
//gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
//pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
  cout << "line 240, im1 " << img1_shape << endl;
  cout << "line 241, im1 is " << max_shape(img1_shape, 2) << " and im0 is " << max_shape(img0_shape, 3) << endl;
  float gain = max_shape(img1_shape, 2) / max_shape(img0_shape, 3);
  float padw = (img1_shape[1] - img0_shape[1] * gain ) / 2.0;
  float padh = (img1_shape[0] - img0_shape[0] * gain ) / 2.0;
  cout << "line 244, gain is " << gain << ", padw is " << padw << " and padh is " << padh << endl;
/*
coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
*/
  coords.index_put_({Slice(), 0}, coords.index({Slice(), 0}) - padw);
  coords.index_put_({Slice(), 2}, coords.index({Slice(), 2}) - padw);
  coords.index_put_({Slice(), 1}, coords.index({Slice(), 1}) - padh);
  coords.index_put_({Slice(), 3}, coords.index({Slice(), 3}) - padh);
  coords.index_put_({Slice(), Slice(None, 4)}, coords.index({Slice(), Slice(None, 4)}) / gain);
  clip_coords(coords, img0_shape);
  return coords;
}

int main(int argc,char * argv[]){

  //at::init_num_threads();//***zh,好像加上这句，速度有一点点提升

  string modelpath = "/home/zherlock/c++ example/object detection/files/yolov5s.torchscript";
  long start = time_in_ms();
  torch::jit::script::Module model = torch::jit::load(modelpath);
  //long end = ;
  cout << "it took " << time_in_ms() - start << " ms to load the model" << endl;
  torch::jit::getProfilingMode() = false;
  torch::jit::getExecutorMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  
  string img_path = "/home/zherlock/c++ example/object detection/files/test.png";
  //string img_path = "/home/zherlock/InstanceDetection/yolov5_old/test.png";
  Mat img = imread(img_path);
  Mat im0 = imread(img_path);
  auto tim0 = torch::from_blob(img.data, {img.rows, img.cols, img.channels()});
  //cout << "line 269, tim0 shape is " << tim0.sizes() << endl;//(480,640,3)
  //return 0;
  img = letterbox(img);//zh,,resize
  
  cvtColor(img, img, CV_BGR2RGB);//***zh, bgr->rgb
  start = time_in_ms();
  img.convertTo(img, CV_32FC3, 1.0f / 255.0f);//zh, 1/255
  auto tensor_img = torch::from_blob(img.data, {img.rows, img.cols, img.channels()});
  tensor_img = tensor_img.permute({2, 0, 1});
  tensor_img = tensor_img.unsqueeze(0);
  cout << "line 201, " << tensor_img.sizes() << endl;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor_img);
  //end = ;
  cout << "it took " << time_in_ms() - start << " ms to preprocess image" << endl;
  start = time_in_ms();
  torch::jit::IValue output = model.forward(inputs);
  //end = time_in_ms();
  cout << "it took " << time_in_ms() - start << " ms to model forward" << endl;


  //Detect layer
  //set parameters
  int ny, nx;
  int na = 3;int no = 85;int bs = 1;
  at::Tensor op_1, op_2, op_3, op;
  at::Tensor y_0, y_1, y_2, y;
  at::Tensor grid_1, grid_2, grid_3, grid;
  float stride[3] = {8.0, 16.0, 32.0};
  at::Tensor anchor_grid = torch::ones({3, 1, 3, 1, 1, 2});
  ifstream f;
  string gridpath = "/home/zherlock/c++ example/object detection/files/anchor_grid.txt";
  f.open(gridpath);
  string str;
  while (std::getline(f,str)){
    vector<string> mp = split(str,",");
    anchor_grid.index_put_({stoi(mp[0]),stoi(mp[1]),stoi(mp[2]),stoi(mp[3]),stoi(mp[4]),stoi(mp[5])}, torch::ones({1}) * stof(mp[6]) );
  }
  f.close();

  //cout << "line 261, anchor grid is " << anchor_grid <<endl;


//at::Tensor a = torch::ones({1}) * 18.0;
//cout << "line 261, a is " << a <<endl;
  //cout << "line 225, stride, " << stride[0] <<endl;
  //grid_1 = torch::zeros({1});grid_2 = torch::zeros({1});grid_3 = torch::zeros({1});
  //grid = torch::cat({grid_1, grid_2, grid_3}, 0);
  //cout << "line 220, " << grid_1 << endl;

  //at::Tensor xx = output.toList().get(0).toTensor();
  //cout << "line 270, " << xx.sizes() << endl;
  //cout << "line 271, " << xx.index({0,0,0,0,Slice(None, 5)}) << endl; //it fits

  
  //run
  for(int i = 0; i < 3; i++){
    op = output.toList().get(i).toTensor().contiguous();
    //op = op.view({1, 255, op.sizes()[2],op.sizes()[3]});
    //cout << "line 286, shape is " << op.sizes() << endl;
    //cout << "line 287, test num is " << op.index({0,0,0,0,Slice(None, 5)}) << endl;
    //cout << "line 287, test num is " << op.index({0,Slice(None, 5),0,0}) << endl;
    //return 0;
    bs = op.sizes()[0]; ny = op.sizes()[2]; nx = op.sizes()[3];
    op = op.view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();
    grid = make_grid(nx, ny);
    //cout << "line 290, op shapes is " << op.sizes() << endl;
    //cout << "line 291, test num is " << grid.index({0,0,47,0,Slice(None, 5)}) << endl;
    //return 0;
    y = op.sigmoid();
    //cout << "line 292-0, test num is " << op.index({0,0,0,1,Slice(None, 5)}) << endl;
    //cout << "line 292-1, test num is " << y.index({0,0,0,1,Slice(None, 5)}) << endl;
    
    //cout << "line 228, " << y.sizes() << endl;
    //cout << "grid is " << grid.sizes() << endl;
    //cout << "stride is " << stride[i] << endl;
    //cout << "test num is " << grid.index({0,0,0,0,0}) << endl;
    at::Tensor test = y.index({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}) * 2.0 - 0.5;
    test = y.index({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}) * 2.0 - 0.5 + grid;
    //cout << "line 304, test num is " << grid.index({0,0,0,0,Slice(None, 5)}) << endl;
    //cout << "line 305, test num is " << test.index({0,0,0,0,Slice(None, 5)}) << endl;
    at::Tensor temp = (y.index({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}) * 2.0 - 0.5 + grid) * stride[i];
    y.index_put_({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}, temp);
    //cout << "line 305, test num is " << temp.index({0,0,0,0,Slice(None, 5)}) << endl;


    temp = (2.0 * y.index({Slice(),Slice(), Slice(),Slice(), Slice(2, 4)})).pow(2) * anchor_grid.index({i});
    //cout << "line 306, test num is " << temp.index({0,0,0,0,Slice(None, 5)}) << endl;
    y.index_put_({Slice(),Slice(), Slice(),Slice(), Slice(2, 4)}, temp);
    y = y.view({bs, -1, no});
    //cout << "line 308-0, test num is " << y.sizes() << endl;
    //cout << "line 309-1, test num is " << y.index({0,0,Slice(None, 5)}) << endl;
    if(i == 0){
      y_0 = y;
    } 
    else if (i == 1){
      y_1 = y;
    }
    else{
      y_2 = y;
    }
    //y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
    //y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    //z.append(y.view(bs, -1, self.no))
}

  op = torch::cat({y_0, y_1,y_2}, 1);
  op = op.view({-1, op.sizes()[2]});
  //cout << "line 297, op shape " << op.sizes() << endl;


  cout << "line 306, test number is " << op.index({100,Slice(None, 5)}) << endl;
  //return 0;
  
  


  
  //at::Tensor op = output.toTuple()->elements().at(0).toTensor();
  //op = op.view({-1 ,op.sizes()[2] });
  
  
  std::vector<string> labels;
  string labelpath = "/home/zherlock/InstanceDetection/yolov5_old/labels.txt";
  f.open(labelpath);
  while (std::getline(f,str)){
    labels.push_back(str);
  }
  f.close();
  start = time_in_ms();
  op = non_max_suppression(op, labels, 0.4, 0.5, true,  false);
  cout << "it took " << time_in_ms() - start << " ms to non_max_suppression" << endl;
  try{
    at::Tensor temp  = op.index({0});
  }catch(...){
    cout << "no objects, line 401 " << endl; 
    return -1;
  }
  int img_shape[2] = {tensor_img.sizes()[2], tensor_img.sizes()[3]};
  int im0_shape[3] = {im0.rows, im0.cols, im0.channels()};
  //cout << "line 422, img is " <<  img_shape[0] << " " << img_shape[1] << endl;
  //cout << "line 422, im0 is " <<  im0_shape[0] << " " << im0_shape[1] << im0_shape[2] << endl;
  //det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
  //cout << "line 423, " << tensor_img.sizes()(2,3) << endl;
  at::Tensor temp;
  temp = scale_coords(img_shape, op.index({Slice(), Slice(None, 4)}), im0_shape).round();
  cout << "line 421, temp is " << temp << endl;
  op.index_put_({Slice(), Slice(None, 4)}, temp);

  for( int i = 0; i < op.sizes()[0]; i++ ){

  cout << "start of object " << i << endl;

  at::Tensor opi = op.index({i});
  
  int op_class = opi.index({5}).item().toInt();
  cout << " op_class is " << op_class << ", and it's " << labels[op_class] << endl;

  int x1 = opi.index({0}).item().toInt();
  int y1 = opi.index({1}).item().toInt();

  int x2 = opi.index({2}).item().toInt();
  int y2 = opi.index({3}).item().toInt();

  cout << " the bounding box is x1 = " << x1 << ", and y1 = " << y1 << ", and x2 = " << x2 << ", and y2 = " << y2 << endl;

  cv::rectangle(im0, Point(x1,y1), Point(x2,y2), Scalar(0,0,255), 2, 8, 0);
  cv::putText(im0,labels[op_class],Point(x1,y1),1,1.0,cv::Scalar(0,255,0),1);
  }
  

  imshow("test image", im0);
  waitKey();




  return 0;
}
