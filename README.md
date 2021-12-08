# OCRDLL
base on paddleocr dll   
在paddleocr的基础上进行了简单的封装，实现基本的逻辑功能，将输出转为C类型的结构体,并且提供了`python`语言`ctypes`的调用实现
[paddleocr地址](https://github.com/PaddlePaddle/PaddleOCR)，`paddle_inference` [高性能预测库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows)

## CMakelist.txt 参考[paddleocr部署cmakelist文件](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/deploy/cpp_infer/CMakeLists.txt)进行了大幅度的简化（其实复杂的不会），本预测仅采用CPU的mkldnn进行加速，如果是要使用cuda，trt等预测加速，请参考`paddleocr`[github地址](https://github.com/PaddlePaddle/PaddleOCR)
