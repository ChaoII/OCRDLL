#ifndef OCRDLL_H
#define OCRDLL_H

#include "ocrdll_global.h"
#include "ocr_main.h"

#ifdef __cplusplus

extern "C"
{
#endif
    // 写自己的逻辑


    OCRDLL_EXPORT RecResultArray dll_ocr_pipline(const char* image);

    OCRDLL_EXPORT void dll_init_model(
            const char* det_model_dir ,
            const char* rec_model_dir ,
            const char* char_list_file);

//    const char* det_model_dir = "",
//    const char* rec_model_dir = "",
//    bool use_gpu = false,
//    int gpu_id = 0,
//    int gpu_mem = 4000,
//    int cpu_threads = 10,
//    bool enable_mkldnn = false,
//    bool use_tensorrt = false,
//    const char* precision = "fp32",
//    bool benchmark = true,
//    int max_side_len = 960,
//    float det_db_thresh = 0.3,
//    float det_db_box_thresh = 0.5,
//    float det_db_unclip_ratio = 1.6,
//    bool use_polygon_score = false,
//    bool visualize = true,
//    bool use_angle_cls = false,
//    const char* cls_model_dir = "",
//    float cls_thresh = 0.9,
//    int rec_batch_num = 1,
//    const char* char_list_file = "ppocr_keys_v1.txt");

    OCRDLL_EXPORT void dll_uninit_model();



#ifdef __cplusplus
}
#endif

#endif // OCRDLL_H
