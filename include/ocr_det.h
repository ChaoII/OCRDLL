
#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <postprocess_op.h>
#include <preprocess_op.h>

using namespace paddle_infer;

namespace PaddleOCR {

class DBDetector {
public:
  explicit DBDetector(const std::string &model_dir, const bool &use_gpu,
                      const int &gpu_id, const int &gpu_mem,
                      const int &cpu_math_library_num_threads,
                      const bool &use_mkldnn, const int &max_side_len,
                      const double &det_db_thresh,
                      const double &det_db_box_thresh,
                      const double &det_db_unclip_ratio,
                      const bool &use_polygon_score, const bool &visualize,
                      const bool &use_tensorrt, const std::string &precision) {
    this->use_gpu_ = use_gpu;
    this->gpu_id_ = gpu_id;
    this->gpu_mem_ = gpu_mem;
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    this->use_mkldnn_ = use_mkldnn;

    this->max_side_len_ = max_side_len;

    this->det_db_thresh_ = det_db_thresh;
    this->det_db_box_thresh_ = det_db_box_thresh;
    this->det_db_unclip_ratio_ = det_db_unclip_ratio;
    this->use_polygon_score_ = use_polygon_score;

    this->visualize_ = visualize;
    this->use_tensorrt_ = use_tensorrt;
    this->precision_ = precision;

    LoadModel(model_dir);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir);

  // Run predictor
  void Run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes, std::vector<double> *times);

private:
  std::shared_ptr<Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;

  int max_side_len_ = 960;

  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.5;
  double det_db_unclip_ratio_ = 2.0;
  bool use_polygon_score_ = false;

  bool visualize_ = true;
  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";

  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  bool is_scale_ = true;

  // pre-process
  ResizeImgType0 resize_op_;
  Normalize normalize_op_;
  Permute permute_op_;

  // post-process
  PostProcessor post_processor_;
};

} // namespace PaddleOCR