/**
* This file is part of Crowd-SLAM.
* Copyright (C) 2020 João Carlos Virgolino Soares - Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: Marcelo Gattass and Marco Antonio Meggiolaro
* For more information see <https://github.com/virgolinosoares/Crowd-SLAM>.
* Please report suggestions and comments to virgolinosoares@gmail.com
* 
* YOLO detection and processing functions are based on the OpenCV project. They are subject to the license terms at http://opencv.org/license.html
*/

//这是yolo检测器的源代码
#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <thread>

#include <fstream>
#include <sstream>
#include <iostream>


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <cstdio>
#include <boost/thread.hpp>

#include <mutex>

#include "Tracking.h"
#include <opencv2/opencv.hpp>	

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "logging.h"
#include "NvInfer.h"


namespace ORB_SLAM3
{
//定义了对象跟踪类,这个是调用了orb-slam中的tracking
class Tracking;

//定义了目标检测器类
class Detector
{

//定义了其中的公有方法
public:
	// 初始化目标检测器类
	Detector();
	// 创建对象跟踪类的指针对象
	Tracking *mpTracker;

	// 预定义Run方法
	void Run();
	// 预定义对象跟踪设置方法
	void SetTracker(Tracking *pTracker);
	// 预定义是否是新的图像方法
	bool isNewImgArrived();
	// 预定义设置目标检测器标志方法
	void SetDetectionFlag();
	// 预定义目标检测方法
	void Detect();

	// 预定义检测是否完成的方法
	bool isFinished();
	// 预定义完成请求的方法
	void RequestFinish();

	// 定义一个矩阵类来存放对象
	cv::Mat mImg;	
	// mutex定义了C++11标准中的一些互斥访问的类与方法等
	// 这个表示异步对新的图像进行目标检测的操作
	std::mutex mMutexNewImgDetection;
	// 这个表示异步获取新的图像的操作
	std::mutex mMutexGetNewImg;
	// 定义属性来标记这个图像是不是新来的
	bool mbNewImgFlag;

	// 定义属性来标记这个请求是否已经完成（是不是目标检测请求已经完成？）
	bool mbFinishRequested;
	// 定义异步操作，完成整个语义分割
	std::mutex mMutexFinish;

	// 这个aprox是啥意思？
	double aprox_area;	
	double fig_area;

	// 可能是图片大小参数？
	int fig_param = 640*480;
	int fig_factor;

	//------------------------------------------------------------

	// YOLO 的参数，先定义net变量
	cv::dnn::Net net;
	
/**		//Tensorrt参数
	#define DEVICE 0  // GPU id GPU设备的编号，需要在终端中查看
	// 定义置信区间阈值变量
	#define BBOX_CONF_THRESH 0.3 //目标检测框概率阈值

	// 定义非极大值抑制的阈值(为了除去对于同一个物体的其他目标检测框，让一个物体只有一个目标检测框)
	#define NMS_THRESH 0.45  //非极大值抑制阈值
	
	static const int INPUT_W = 640;
        static const int INPUT_H = 640;
        static const int NUM_CLASSES = 80;
        const char* INPUT_BLOB_NAME = "input_0";
        const char* OUTPUT_BLOB_NAME = "output_0";
        static Logger gLogger;
        static const char* class_names[];
        static const char* engine;
        **/
	
	// 表示输入的图像的宽度
	int inpWidth; 
	// 表示输入的图像的长度
	int inpHeight;
	
	// vector是可以改变大小的数组的序列容器，在末尾增删改效率高
	// 这个数组存的是检测的类别
	std::vector<std::string> classes;

	// Rect是长方形对象，表示里面被识别为人的矩形框
//	std::vector<cv::Rect> people_boxes;
	
	std::vector<cv::Rect> dynamic_boxes;
	
//	std::vector<cv::Rect> half_boxes;
	
//	std::vector<cv::Rect> stay_boxes;				


	std::vector<cv::Rect> boxes1;
	int label[80] = {0};
	// YOLO 的功能函数定义
	/************************************************************************
		函数功能：得到输出层的名称(识别为的种类名)
		输入：需要遍历的网络Net
		输出：返回输出层的名称
	************************************************************************/
	std::vector<cv::String> getOutsNames(const cv::dnn::Net& net);
	// 对检测后的结果进行处理，使用非最大值抑制移除置信度较低的边界框，centersx指的是置信框中心的坐标，centersY同理。boxes是检测框
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<int>& classIds, std::vector<int> &centersX, std::vector<int> &centersY, std::vector<cv::Rect>& boxes);
	//画出预测框
        void drawPred (int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

};

}// namespace ORB_SLAM3

#endif // DETECTOR_H
