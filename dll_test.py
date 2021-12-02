from ctypes import *
from typing import List


class PointR(Structure):
    x: int
    y: int
    _fields_ = [("x", c_int), ("y", c_int)]


class BoxR(Structure):
    left_top: PointR
    right_top: PointR
    right_bottom: PointR
    left_bottom: PointR
    _fields_ = [("left_top", PointR), ("right_top", PointR), ("right_bottom", PointR), ("left_bottom", PointR)]


class RecResult(Structure):
    rec_str: str
    box: BoxR
    score: float
    _fields_ = [("rec_str", c_char_p), ("box", BoxR), ("score", c_double)]


class RecResultArray(Structure):
    rec_res: List[RecResult]
    res_nums: int
    _fields_ = [("rec_res", POINTER(RecResult)), ("res_nums", c_int)]


# 加载API库
api = CDLL(r"D:\QtCreatorProjects\ocrdll\build\release\ocrdll.dll")

'''
    bool use_gpu = false,
    int gpu_id = 0,
    int gpu_mem = 4000,
    int cpu_threads = 10,
    bool enable_mkldnn = false,
    bool use_tensorrt = false,
    const char* precision = "fp32",
    bool benchmark = true,
    const char* save_log_path = "./log_output/",
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
    const char* char_list_file = "ppocr_keys_v1.txt"

'''

'''
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

'''
api.dll_init_model(
    c_char_p("det_infer".encode("utf8")),
    c_char_p("rec_infer".encode("utf8")),
    c_char_p("ppocr_keys_v1.txt".encode("utf8"))
    # c_bool(False),
    # c_int(0),
    # c_int(4000),
    # c_int(10),
    # c_bool(False),
    # c_bool(False),
    # c_char_p("fp32".encode("utf8")),
    # c_bool(False),
    # c_int(960),
    # c_float(0.3),
    # c_float(0.5),
    # c_float(1.6),
    # c_bool(False),
    # c_bool(True),
    # c_bool(False),
    # c_char_p("".encode("utf8")),
    # c_float(0.9),
    # c_int(1),
    # c_char_p("ppocr_keys_v1.txt".encode("utf8"))
)
# (c_char_p("33.png".encode("utf8")))

ocr_pipline = api.dll_ocr_pipline
ocr_pipline.argtypes = (c_char_p,)
ocr_pipline.restype = RecResultArray

RecResultArray = ocr_pipline(c_char_p("43.png".encode("utf8")))
nums = RecResultArray.res_nums

for i in range(nums):
    RecResult = RecResultArray.rec_res[i]
    print(RecResult.rec_str.decode("u8"), RecResult.score)
    print(RecResult.box.left_top.x)
# print(res,"444")

api.dll_uninit_model()
