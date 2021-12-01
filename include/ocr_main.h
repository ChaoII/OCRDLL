#pragma once
// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
//#include "glog/logging.h"
#include "omp.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <ostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <numeric>
//#include <glog/logging.h>
#include <include/ocr_det.h>
#include <include/ocr_cls.h>
#include <include/ocr_rec.h>
#include <include/utility.h>
#include <sys/stat.h>

//#include <gflags/gflags.h>
//#include "auto_log/autolog.h"


using namespace PaddleOCR;




int ocr_pipline(const char* image_dir);

void initModelSub(
    const char* det_model_dir,
    const char* rec_model_dir,
    bool use_gpu,
    int gpu_id,
    int gpu_mem,
    int cpu_threads,
    bool enable_mkldnn,
    bool use_tensorrt,
    const char* precision,
    bool benchmark,
    int max_side_len,
    float det_db_thresh,
    float det_db_box_thresh,
    float det_db_unclip_ratio,
    bool use_polygon_score,
    bool visualize,
    bool use_angle_cls,
    const char* cls_model_dir,
    float cls_thresh,
    int rec_batch_num,
    const char* char_list_file);

void uninitModelSub();
