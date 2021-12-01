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

#include "glog/logging.h"
#include "omp.h"
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

#include <glog/logging.h>
#include <include/ocr_det.h>
#include <include/ocr_cls.h>
#include <include/ocr_rec.h>
#include <include/utility.h>
#include <sys/stat.h>
#include "ocr_main.h"
#include <gflags/gflags.h>
//#include "auto_log/autolog.h"




using namespace std;
using namespace cv;
using namespace PaddleOCR;


int ocr_pipline(std::vector<cv::String> cv_all_img_names) {
    DBDetector det("", false, 0,
                   4000, 10, 
                   true, 960, 0.3,
                   0.5, 1.6,
                   0.5, true,
                   false, "fp32");

    Classifier *cls = nullptr;
    if (true) {
      cls = new Classifier("", false, 0,
                           4000, 10,
                           true, 0.3,
                           false, "fp32");
    }

    CRNNRecognizer rec("", true, 0,
                       4000, 0.3,
                       true, "",
                       false, "fp32");

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < cv_all_img_names.size(); ++i) {
      LOG(INFO) << "The predict img: " << cv_all_img_names[i];

      cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
        exit(1);
      }
      std::vector<std::vector<std::vector<int>>> boxes;
      std::vector<double> det_times;
      std::vector<double> rec_times;
        
      det.Run(srcimg, boxes, &det_times);
    
      cv::Mat crop_img;
      for (int j = 0; j < boxes.size(); j++) {
        crop_img = Utility::GetRotateCropImage(srcimg, boxes[j]);

        if (cls != nullptr) {
          crop_img = cls->Run(crop_img);
        }
        rec.Run(crop_img, &rec_times);
      }
        
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cout << "Cost  "
                << double(duration.count()) *
                       std::chrono::microseconds::period::num /
                       std::chrono::microseconds::period::den
                << "s" << std::endl;
    }
      
    return 0;
}
