#pragma once

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
#include <ocr_det.h>
#include <ocr_cls.h>
#include <ocr_rec.h>
#include <utility.h>
#include <sys/stat.h>

//#include <gflags/gflags.h>
//#include <auto_log/autolog.h>


//using namespace PaddleOCR;
namespace PaddleOCR {
    extern "C" {
    struct PointR {
        int x;
        int y;
    };
    struct BoxR {
        PointR left_top;
        PointR right_top;
        PointR right_bottom;
        PointR left_bottom;
    };

    struct RecResult {
        const char *rec_str;
        BoxR box;
        double score;
    };

    struct RecResultArray {
        RecResult *rec_res;
        int res_nums;
    };
    }

    RecResultArray ocr_pipline(const char *image_dir);

    void initModelSub(
            const char *det_model_dir = "",
            const char *rec_model_dir = "",
            bool use_gpu = false,
            int gpu_id = 0,
            int gpu_mem = 4000,
            int cpu_threads = 10,
            bool enable_mkldnn = false,
            bool use_tensorrt = false,
            const char *precision = "fp32",
            bool benchmark = true,
            int max_side_len = 960,
            float det_db_thresh = 0.3,
            float det_db_box_thresh = 0.5,
            float det_db_unclip_ratio = 1.6,
            bool use_polygon_score = false,
            bool visualize = false,
            bool use_angle_cls = false,
            const char *cls_model_dir = "",
            float cls_thresh = 0.9,
            int rec_batch_num = 1,
            const char *char_list_file = "ppocr_keys_v1.txt"
    );

    void uninitModelSub();
}