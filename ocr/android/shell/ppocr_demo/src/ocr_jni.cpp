#include <jni.h>
#include <opencv2/opencv.hpp>
#include "pipeline.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_autojs_ocr_OCREngine_recognize(
    JNIEnv* env, jobject thiz,
    jlong matAddr,  // OpenCV Mat 对象的地址
    jstring detModel, jstring recModel, jstring clsModel,
    jstring dictPath, jstring configPath) {
    
    try {
        // 从地址获取 OpenCV Mat 对象
        cv::Mat& image = *(cv::Mat*)matAddr;
        
        // 转换字符串
        const char* det_model = env->GetStringUTFChars(detModel, nullptr);
        const char* rec_model = env->GetStringUTFChars(recModel, nullptr);
        const char* cls_model = env->GetStringUTFChars(clsModel, nullptr);
        const char* dict_path = env->GetStringUTFChars(dictPath, nullptr);
        const char* config_path = env->GetStringUTFChars(configPath, nullptr);
        
        // 创建 Pipeline
        Pipeline pipeline(det_model, cls_model, rec_model, "", 1, config_path, dict_path);
        
        // 识别图像
        std::string result = pipeline.ProcessWithJson(image, "");
        
        // 释放字符串
        env->ReleaseStringUTFChars(detModel, det_model);
        env->ReleaseStringUTFChars(recModel, rec_model);
        env->ReleaseStringUTFChars(clsModel, cls_model);
        env->ReleaseStringUTFChars(dictPath, dict_path);
        env->ReleaseStringUTFChars(configPath, config_path);
        
        return env->NewStringUTF(result.c_str());
        
    } catch (const std::exception& e) {
        return env->NewStringUTF("{\"error\": \"Exception occurred\"}");
    }
}
