#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

//using namespace std;
using namespace paddle;

namespace PaddleOCR {

class Normalize {
public:
  virtual void Run(cv::Mat *im, const std::vector<float> &mean,
                   const std::vector<float> &scale, const bool is_scale = true);
};

// RGB -> CHW
class Permute {
public:
  virtual void Run(const cv::Mat *im, float *data);
};

class ResizeImgType0 {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, int max_size_len,
                   float &ratio_h, float &ratio_w, bool use_tensorrt);
};

class CrnnResizeImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                   bool use_tensorrt = false,
                   const std::vector<int> &rec_image_shape = {3, 32, 320});
};

class ClsResizeImg {
public:
  virtual void Run(const cv::Mat &img, cv::Mat &resize_img,
                   bool use_tensorrt = false,
                   const std::vector<int> &rec_image_shape = {3, 48, 192});
};

} // namespace PaddleOCR