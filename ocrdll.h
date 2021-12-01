#ifndef OCRDLL_H
#define OCRDLL_H

#include "ocrdll_global.h"
#include "ocr_main.h"

#ifdef __cplusplus

extern "C"
{
#endif
    // 写自己的逻辑

    OCRDLL_EXPORT int dll_ocr_pipline(const char * image);

    OCRDLL_EXPORT void dll_init_model(
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

    OCRDLL_EXPORT void dll_uninit_model();



#ifdef __cplusplus
}
#endif

#endif // OCRDLL_H
