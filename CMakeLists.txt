cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
project(ocrdll)

# 文件较大存放自己需要的位置
set(PADDLE_LIB ${PROJECT_SOURCE_DIR}/paddle_inference)
set(CMAKE_CXX_FLAGS_RELEASE "/MT")
set(CMAKE_C_FLAGS_RELEASE "/MT")
set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
set(CMAKE_C_FLAGS_DEBUG "/MTd")

# 用的MSVC有该死的乱码情况,只能用该挫方法处理
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 设置输出目录，dll输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/install)

#添加本地文件包括include和src
include_directories(${PROJECT_SOURCE_DIR}/include)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src SRC)

# opencv 软链接
set(OpenCV_DIR C:/opencv)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

#paddle_inference
include_directories(${PADDLE_LIB}/paddle/include)
include_directories(${PADDLE_LIB}/third_party/install/cryptopp/include)
include_directories(${PADDLE_LIB}/third_party/install/gflags/include)
#include_directories(${PADDLE_LIB}/third_party/install/glog/include)
include_directories(${PADDLE_LIB}/third_party/install/mkldnn/include)
include_directories(${PADDLE_LIB}/third_party/install/protobuf/include)
include_directories(${PADDLE_LIB}/third_party/install/xxhash/include)
include_directories(${PADDLE_LIB}/third_party/install/mklml/include)

link_directories(${PADDLE_LIB}/paddle/lib/)
link_directories(${PADDLE_LIB}/third_party/install/cryptopp/lib)
link_directories(${PADDLE_LIB}/third_party/install/gflags/lib)
#link_directories(${PADDLE_LIB}/third_party/install/glog/lib)
link_directories(${PADDLE_LIB}/third_party/install/mkldnn/lib)
link_directories(${PADDLE_LIB}/third_party/install/protobuf/lib)
link_directories(${PADDLE_LIB}/third_party/install/xxhash/lib)
link_directories(${PADDLE_LIB}/third_party/install/mklml/lib)


add_library(ocrdll SHARED ${SRC})

target_link_libraries(ocrdll ${OpenCV_LIBS}
        paddle_inference
        cryptopp-static
        gflags_static
        mkldnn
        mklml
        libprotobuf
        xxhash
        libiomp5md
        )

# 依赖打包
add_custom_command(TARGET ocrdll POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mkldnn.dll
        )
