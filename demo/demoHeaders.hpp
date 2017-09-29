#pragma once

// Standard libraries
#include <sys/stat.h>
#include <iostream>
#include <queue>
#include <ctime>
#include <string>
#include <fstream>
#include <cstdlib>
#include <experimental/filesystem>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

// Dlib headers
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

// Third party
#include <boost/filesystem.hpp>

// Costants
#define N_LANDMARKS 68

// Namespaces
using namespace dlib;
using namespace std;

// Local headers

// Functions
cv::Mat GetSquareImage(const cv::Mat& img, int target_width);
inline bool exists (const std::string& name);
void writeCircle(const cv::Mat& img, int x, int y);
void writeCoordinatesOnFile(string filename, int* xs, int* ys, int length);
    
