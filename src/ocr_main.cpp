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
#include "ocr_main.h"



using namespace std;
using namespace cv;
using namespace PaddleOCR;

Classifier* cls = nullptr;
DBDetector* det = nullptr;
CRNNRecognizer* rec = nullptr;


void initModelSub(
    const char* det_model_dir = "",
    const char* rec_model_dir = "",
    bool use_gpu = false,
    int gpu_id = 0,
    int gpu_mem = 4000,
    int cpu_threads = 10,
    bool enable_mkldnn = false,
    bool use_tensorrt = false,
    const char* precision = "fp32",
    bool benchmark = true,
    int max_side_len = 960,
    float det_db_thresh = 0.3,
    float det_db_box_thresh = 0.5,
    float det_db_unclip_ratio = 1.6,
    bool use_polygon_score = false,
    bool visualize = true,
    bool use_angle_cls = false,
    const char* cls_model_dir = "",
    float cls_thresh = 0.9,
    int rec_batch_num = 1,
    const char* char_list_file = "ppocr_keys_v1.txt") {

    det = new DBDetector(det_model_dir, use_gpu, gpu_id,
        gpu_mem, cpu_threads,
        enable_mkldnn, max_side_len, det_db_thresh,
        det_db_box_thresh, det_db_unclip_ratio,
        use_polygon_score, visualize,
        use_tensorrt, precision);


    if (use_angle_cls) {
        cls = new Classifier(cls_model_dir, use_gpu, gpu_id,
            gpu_mem, cpu_threads,
            enable_mkldnn, cls_thresh,
            use_tensorrt, precision);
    }


    rec = new CRNNRecognizer(rec_model_dir, use_gpu, gpu_id,
        gpu_mem, cpu_threads,
        enable_mkldnn, char_list_file,
        use_tensorrt, precision);

}

void uninitModelSub() {

    if (cls != nullptr) {
        delete cls;
        cls = nullptr;
    }
    if (det != nullptr) {
        delete det;
        det = nullptr;
    }
    if (rec != nullptr) {
        delete rec;
        rec = nullptr;
    }
}

int ocr_pipline(const char* image_dir) {


    std::vector<std::string> cv_all_img_names;
    std::string s = image_dir;
    cv_all_img_names.push_back(s);
    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < cv_all_img_names.size(); ++i) {
//        LOG(INFO) << "The predict img: " << cv_all_img_names[i];

        cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
        if (!srcimg.data) {
            std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
            exit(1);
        }
        std::vector<std::vector<std::vector<int>>> boxes;
        std::vector<double> det_times;
        std::vector<double> rec_times;

        det->Run(srcimg, boxes, &det_times);

        cv::Mat crop_img;
        for (int j = 0; j < boxes.size(); j++) {
            crop_img = Utility::GetRotateCropImage(srcimg, boxes[j]);

            if (cls != nullptr) {
                crop_img = cls->Run(crop_img);
            }
            rec->Run(crop_img, &rec_times);
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
