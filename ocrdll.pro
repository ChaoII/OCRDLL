QT -= gui

TEMPLATE = lib
DEFINES += OCRDLL_LIBRARY

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    include/clipper.cpp \
    ocrdll.cpp \
    src/ocr_cls.cpp \
    src/ocr_det.cpp \
    src/ocr_main.cpp \
    src/ocr_rec.cpp \
    src/postprocess_op.cpp \
    src/preprocess_op.cpp \
    src/utility.cpp

HEADERS += \
    include/autolog.h \
    include/clipper.h \
    include/dirent.h \
    include/ocr_cls.h \
    include/ocr_det.h \
    include/ocr_rec.h \
    include/postprocess_op.h \
    include/preprocess_op.h \
    include/utility.h \
    ocrdll_global.h \
    ocrdll.h

# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target

LIBS += -L$$PWD/paddle_inference/paddle/lib/ -lpaddle_inference
LIBS += -L$$PWD/paddle_inference/third_party/install/cryptopp/lib/ -lcryptopp-static
LIBS += -L$$PWD/paddle_inference/third_party/install/gflags/lib/ -lgflags_static
LIBS += -L$$PWD/paddle_inference/third_party/install/glog/lib/ -lglog
LIBS += -L$$PWD/paddle_inference/third_party/install/mkldnn/lib/ -lmkldnn
LIBS += -L$$PWD/paddle_inference/third_party/install/mklml/lib/ -lmklml
LIBS += -L$$PWD/paddle_inference/third_party/install/protobuf/lib/ -llibprotobuf
LIBS += -L$$PWD/paddle_inference/third_party/install/xxhash/lib/ -lxxhash
LIBS += -L$$PWD/paddle_inference/third_party/install/opencv/lib/ -lopencv_world451

INCLUDEPATH += $$PWD/include

INCLUDEPATH += $$PWD/paddle_inference/paddle/include
DEPENDPATH += $$PWD/paddle_inference/paddle/include



INCLUDEPATH += $$PWD/paddle_inference/third_party/install/cryptopp/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/cryptopp/include



INCLUDEPATH += $$PWD/paddle_inference/third_party/install/gflags/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/gflags/include




INCLUDEPATH += $$PWD/paddle_inference/third_party/install/glog/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/glog/include




INCLUDEPATH += $$PWD/paddle_inference/third_party/install/mkldnn/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/mkldnn/include



INCLUDEPATH += $$PWD/paddle_inference/third_party/install/mklml/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/mklml/include



INCLUDEPATH += $$PWD/paddle_inference/third_party/install/protobuf/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/protobuf/include


INCLUDEPATH += $$PWD/paddle_inference/third_party/install/xxhash/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/xxhash/include



INCLUDEPATH += $$PWD/paddle_inference/third_party/install/opencv/include
DEPENDPATH += $$PWD/paddle_inference/third_party/install/opencv/include
