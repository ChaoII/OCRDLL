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
RecResultArray res;


//使用glog严重错误DUMP功能





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
        const char* char_list_file) {

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
    if(res.rec_res != nullptr){
        delete res.rec_res;
        res.rec_res = nullptr;
    }
}

RecResultArray ocr_pipline(const char* image_dir) {


//    google::InitGoogleLogging("123");


//    FLAGS_logbufsecs = 0;  //日志实时输出
//    FLAGS_max_log_size=10; //最大日志文件大小 10M
//    FLAGS_logtostderr = true; // 设置日志消息是否转到标准输出而不是日志文件
//    FLAGS_colorlogtostderr = true; // 设置记录到标准输出的颜色消息（如果终端支持）
//    FLAGS_alsologtostderr = true;

    std::vector<std::string> cv_all_img_names;

    cv_all_img_names.push_back(image_dir);

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < cv_all_img_names.size(); ++i) {

        cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
        if (!srcimg.data) {
            std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
            exit(1);
        }
        std::vector<std::vector<std::vector<int>>> boxes;
        std::vector<double> det_times;
        std::vector<double> rec_times;

        det->Run(srcimg, boxes, &det_times);
//        LOG(INFO)<<"------------------------------------------";
        std::cout<< "det前处理："<<det_times[0]<<std::endl;
        std::cout<< "det推理："<<det_times[1]<<std::endl;
        std::cout<< "det后处理："<<det_times[2]<<std::endl;
        //------------------------数据声名----------------------
        cv::Mat crop_img;
        std::vector<std::string> rec_strs;
        std::vector<double> rec_scores;
        std::string rstr;
        double score;
        //-----------------------------------------------------
        for (int j = 0; j < boxes.size(); j++) {
            crop_img = Utility::GetRotateCropImage(srcimg, boxes[j]);
            //            cv::imwrite(std::string("im")+std::to_string(j)+std::string(".jpg"),crop_img);
            if (cls != nullptr) {
                crop_img = cls->Run(crop_img);
            }
            rec->Run(crop_img, &rec_times,rstr,score);
            rec_strs.push_back(rstr);
            rec_scores.push_back(score);
        }

        std::cout<< "rec前处理："<<rec_times[0]<<std::endl;
        std::cout<< "rec推理："<<rec_times[1]<<std::endl;
        std::cout<< "rec后处理："<<rec_times[2]<<std::endl;

        // ----------------------赋值-----------------------------
        RecResult re_sub;
        BoxR box_r;
        res.res_nums = boxes.size();
        res.rec_res = new  RecResult[boxes.size()];
        for (int n = 0; n < boxes.size(); n++) {
            PointR left_top,right_top,right_bottom,left_bottom;
            left_top.x=int(boxes[n][0][0]);
            left_top.y=int(boxes[n][0][1]);
            right_top.x=int(boxes[n][1][0]);
            right_top.y=int(boxes[n][1][1]);
            right_bottom.x = int(boxes[n][2][0]);
            right_bottom.y=int(boxes[n][2][1]);
            left_bottom.x=int(boxes[n][3][0]);
            left_bottom.y=int(boxes[n][3][1]);
            box_r.left_top = left_bottom;
            box_r.right_top = right_top;
            box_r.right_bottom=right_bottom;
            box_r.left_bottom=left_bottom;
            re_sub.box = box_r;
            re_sub.rec_str = rec_strs[n].c_str();
            re_sub.score = rec_scores[n];
            res.rec_res[n] = re_sub;
        }

        //------------------------------------------------------
        auto end = std::chrono::system_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "总耗时  "
                  << double(duration.count()) *
                     std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
                  << "s" << std::endl;
    }

    return res;

}

