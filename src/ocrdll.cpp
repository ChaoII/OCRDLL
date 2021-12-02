#include "ocrdll.h"
#include <iostream>



RecResultArray dll_ocr_pipline(const char *image)
{
    return ocr_pipline(image);
}

void dll_init_model(const char *det_model_dir, const char *rec_model_dir, const char *char_list_file)
{
    std::cout<<det_model_dir<<"\n"<<rec_model_dir<<"\n"<<char_list_file<<std::endl;

    initModelSub(det_model_dir,rec_model_dir);

}

void dll_uninit_model()
{
    uninitModelSub();
}
