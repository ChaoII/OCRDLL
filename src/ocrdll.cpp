#include "ocrdll.h"

using namespace PaddleOCR;

RecResultArray dll_ocr_pipline(const char *image) {
    return ocr_pipline(image);
}

void dll_init_model(const char *det_model_dir, const char *rec_model_dir, const char *char_list_file) {
    initModelSub(det_model_dir, rec_model_dir);

}

void dll_uninit_model() {
    uninitModelSub();
}
