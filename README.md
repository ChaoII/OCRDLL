# OCRDLL
base on paddleocr dll

- base on paddleocr dll
- 在paddleocr的基础上进行了简单的封装，实现基本的逻辑功能，将输出转为C类型的结构体,并且提供了python语言ctypes的调用实现 paddleocr地址，paddle_inference 高性能预测库
CMakelist.txt 参考paddleocr部署cmakelist文件进行了大幅度的简化（其实复杂的不会），本预测仅采用CPU的mkldnn进行加速，如果是要使用cuda，trt等预测加速，请参考paddleocrgithub地址
- 这仅仅是demo能够生成供外部调用的dll，其中dll输入输出接口，请按需调整。
