#!/bin/bash
# setting NDK_ROOT root
export NDK_ROOT=/opt/android-ndk-r20b
echo "NDK_ROOT is ${NDK_ROOT}"
# build
cd $(pwd)/src
# configure
ARM_ABI=arm64-v8a
# ARM_ABI=armeabi-v7a
# ARM_TARGET_LANG=gcc
ARM_TARGET_LANG=clang
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"

if [ "x$1" != "x" ]; then
    ARM_ABI=$1
fi
export ARM_TARGET_LANG
export ARM_ABI
export PADDLE_LITE_DIR
export OPENCV_LITE_DIR

echo "ARM_TARGET_LANG is ${ARM_TARGET_LANG}"
echo "ARM_ABI is ${ARM_ABI}"
echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
rm -rf build
mkdir build
make clean
cd build
cmake -DANDROID_PLATFORM=android-21 -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} -DARM_ABI=${ARM_ABI} -DARM_TARGET_LANG=${ARM_TARGET_LANG} -DOPENCV_LITE_DIR=${OPENCV_LITE_DIR} -DNDK_ROOT=${NDK_ROOT} ..
make -j10

echo "make successful!"

# 检查生成的文件
echo "Generated files in build directory:"
find . -type f \( -name "ppocr_demo" -o -name "libppocr_shared.so" -o -name "ppocr_shared.so" \) -exec ls -la {} \;

# mkdir
cd ../../
if [ ! -d "./ppocr_demo" ]; then
mkdir ppocr_demo
fi

# 复制可执行文件
if [ -f "./src/build/ppocr_demo" ]; then
    cp ./src/build/ppocr_demo ./ppocr_demo
    echo "copied ppocr_demo executable"
else
    echo "Warning: ppocr_demo executable not found"
fi

# 复制共享库
if [ -f "./src/build/libppocr_shared.so" ]; then
    cp ./src/build/libppocr_shared.so ./ppocr_demo/
    echo "copied libppocr_shared.so"
elif [ -f "./src/build/ppocr_shared.so" ]; then
    cp ./src/build/ppocr_shared.so ./ppocr_demo/libppocr_shared.so
    echo "copied and renamed ppocr_shared.so to libppocr_shared.so"
else
    echo "Warning: ppocr_shared library not found"
    # 列出 build 目录中的所有文件来调试
    echo "Files in build directory:"
    ls -la ./src/build/
fi

# 复制依赖库
cp ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so ./ppocr_demo
cp ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so ./ppocr_demo

echo "copy successful!"

# 显示最终生成的文件
echo "Final files in ppocr_demo directory:"
ls -la ./ppocr_demo/
