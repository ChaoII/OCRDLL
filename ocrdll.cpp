#include "ocrdll.h"
#include <iostream>



int dll_ocr_pipline(const char *image)
{
    return ocr_pipline(image);
}

void dll_init_model(const char *det_model_dir, const char *rec_model_dir, bool use_gpu, int gpu_id,
                    int gpu_mem, int cpu_threads, bool enable_mkldnn, bool use_tensorrt, const char *precision,
                    bool benchmark, int max_side_len, float det_db_thresh, float det_db_box_thresh,
                    float det_db_unclip_ratio, bool use_polygon_score, bool visualize, bool use_angle_cls,
                    const char *cls_model_dir, float cls_thresh, int rec_batch_num, const char *char_list_file)
{

    initModelSub(det_model_dir,rec_model_dir,use_gpu,gpu_id,gpu_mem,cpu_threads,enable_mkldnn,use_tensorrt,precision,
                 benchmark,max_side_len,det_db_thresh,det_db_box_thresh,det_db_unclip_ratio,use_polygon_score,visualize,
                 use_angle_cls,cls_model_dir,cls_thresh,rec_batch_num,char_list_file);

}

void dll_uninit_model()
{
    uninitModelSub();
}
