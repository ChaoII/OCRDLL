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

#include <preprocess_op.h>
#include <utility.h>

using namespace paddle_infer;

namespace PaddleOCR {

    class Classifier {
    public:
        explicit Classifier(const std::string& model_dir, const bool& use_gpu,
            const int& gpu_id, const int& gpu_mem,
            const int& cpu_math_library_num_threads,
            const bool& use_mkldnn, const double& cls_thresh,
            const bool& use_tensorrt, const std::string& precision) {
            this->use_gpu_ = use_gpu;
            this->gpu_id_ = gpu_id;
            this->gpu_mem_ = gpu_mem;
            this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
            this->use_mkldnn_ = use_mkldnn;

            this->cls_thresh = cls_thresh;
            this->use_tensorrt_ = use_tensorrt;
            this->precision_ = precision;

            LoadModel(model_dir);
        }

        // Load Paddle inference model
        void LoadModel(const std::string& model_dir);

        cv::Mat Run(cv::Mat& img);

    private:
        std::shared_ptr<Predictor> predictor_;

        bool use_gpu_ = false;
        int gpu_id_ = 0;
        int gpu_mem_ = 4000;
        int cpu_math_library_num_threads_ = 4;
        bool use_mkldnn_ = false;
        double cls_thresh = 0.5;

        std::vector<float> mean_ = { 0.5f, 0.5f, 0.5f };
        std::vector<float> scale_ = { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
        bool is_scale_ = true;
        bool use_tensorrt_ = false;
        std::string precision_ = "fp32";
        // pre-process
        ClsResizeImg resize_op_;
        Normalize normalize_op_;
        Permute permute_op_;

    }; // class Classifier

} // namespace PaddleOCR
