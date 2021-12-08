#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>
#include <glog/logging.h>


class AutoLogger {
public:
    AutoLogger(std::string model_name, 
               bool use_gpu,
               bool enable_tensorrt,
               bool enable_mkldnn,
               int cpu_threads,
               int batch_size,
               std::string input_shape,
               std::string model_precision,
               std::vector<double> time_info,
               int img_num) {
        this->model_name_ = model_name;
        this->use_gpu_ = use_gpu;
        this->enable_tensorrt_ = enable_tensorrt;
        this->enable_mkldnn_ = enable_mkldnn;
        this->cpu_threads_ = cpu_threads;
        this->batch_size_ = batch_size;
        this->input_shape_ = input_shape;
        this->model_precision_ = model_precision;
        this->time_info_ = time_info;
        this->img_num_ = img_num;
    }
    void report() {
        LOG(INFO) << "----------------------- Config info -----------------------";
        LOG(INFO) << "runtime_device: " << (this->use_gpu_ ? "gpu" : "cpu");
        LOG(INFO) << "ir_optim: " << "True";
        LOG(INFO) << "enable_memory_optim: " << "True";
        LOG(INFO) << "enable_tensorrt: " << this->enable_tensorrt_;
        LOG(INFO) << "enable_mkldnn: " << (this->enable_mkldnn_ ? "True" : "False");
        LOG(INFO) << "cpu_math_library_num_threads: " << this->cpu_threads_;
        LOG(INFO) << "----------------------- Data info -----------------------";
        LOG(INFO) << "batch_size: " << this->batch_size_;
        LOG(INFO) << "input_shape: " << this->input_shape_;
        LOG(INFO) << "data_num: " << this->img_num_;
        LOG(INFO) << "----------------------- Model info -----------------------";
        LOG(INFO) << "model_name: " << this->model_name_;
        LOG(INFO) << "precision: " << this->model_precision_;
        LOG(INFO) << "----------------------- Perf info ------------------------";
        LOG(INFO) << "Total time spent(ms): "
                  << std::accumulate(this->time_info_.begin(), this->time_info_.end(), 0);
        LOG(INFO) << "preprocess_time(ms): " << this->time_info_[0] / this->img_num_
                  << ", inference_time(ms): " << this->time_info_[1] / this->img_num_
                  << ", postprocess_time(ms): " << this->time_info_[2] / this->img_num_;
    }
        
private:
    std::string model_name_;
    bool use_gpu_ = false;
    bool enable_tensorrt_ = false;
    bool enable_mkldnn_ = true;
    int cpu_threads_ = 10;
    int batch_size_ = 1;
    std::string input_shape_ = "dynamic";
    std::string model_precision_ = "fp32";
    std::vector<double> time_info_;
    int img_num_;
};
