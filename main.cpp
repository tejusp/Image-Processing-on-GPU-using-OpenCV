#include <iostream>
#include <stdint.h>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <typeinfo>
#include <argp.h>
#include <sys/time.h>

using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

struct Arguments {
    string project;
    string input;
    string output;
    int padding;
    int frames;
    string extension;
    int width;
    int height;
    int area_min;
    int area_max;
    int search_win_size;
    int blur_radius;
    int threshold_win_size;
    float threshold_ratio;
    string log;
    bool verbose;
    
    Arguments():    input("data/"),
    output("output.txt"),
    padding(7),
    frames(40),
    extension(".jpg"),
    width(640),
    height(480),
    area_min(200),
    area_max(400),
    search_win_size(100),
    blur_radius(3),
    threshold_win_size(25),
    threshold_ratio(0.9),
    log("wormSeg.log"),
    verbose(true) {}
} cla;


int timeToUpload=0;

int findCentroidFromImage(cv::Mat, int*, int*, int*);

template <typename T> string NumberToString(T pNumber) {
    ostringstream oOStrStream;
    oOStrStream << pNumber;
    
    return oOStrStream.str();
}


string intToFileName(string fileNameFormat, int fileNumber) {
    string temp = NumberToString(fileNumber);
    
    return fileNameFormat.replace(fileNameFormat.size() - temp.size(), temp.size(), temp);
}


int wormSegmenter() {
    
    fstream outputFile;
    
    outputFile.open(cla.output.c_str(), ios::out);
    
    int x = -1, y = -1, area = -1;
    int adjustX = 0, adjustY = 0;
    
    for (int fileNumber = 0; fileNumber < cla.frames; fileNumber ++) {
        string fileName = cla.input + intToFileName("0000000", fileNumber) + cla.extension;
        cv::Mat src = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
        
        if (!src.data) {
            cout << endl << "Exited." << endl;
            exit(1);
        }
        
        if((x == -1) && (y == -1)) {
            findCentroidFromImage(src, &x, &y, &area);
            src = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
            adjustX = x - (cla.search_win_size / 2);
            adjustY = y - (cla.search_win_size / 2);
        }
        else {
            src = src(cv::Rect(x - (cla.search_win_size / 2), y - (cla.search_win_size / 2), cla.search_win_size, cla.search_win_size));
            
            findCentroidFromImage(src, &x, &y, &area);
            
            if((x > 0) && (y > 0)) {
                x += adjustX;
                y += adjustY;
                adjustX = x - (cla.search_win_size / 2);
                adjustY = y - (cla.search_win_size / 2);
            }
        }
        
        outputFile << fileNumber << ", " << x << ", " << y << ", " << area << endl;
    }
    
    outputFile.close();
    
    return 0;
}



int findCentroidFromImage(cv::Mat src, int *pX, int *pY, int *pArea) {

    //cout << "size = " << src.size()<<endl;

    
    

    //GPU Mat... Copy from CPU memory to GPU memory...
    cv::gpu::GpuMat gpu_src(src);

    //cout << "ms="<<ms1<<endl;
    //cout<<"diff="<<ms1-ms<<endl;
  

//    struct timeval tp1;
//    long int ms1=0;
//    struct timeval tp;
//
//    gettimeofday(&tp, NULL);
//    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    //Filters on GPU...
    cv::gpu::blur(gpu_src, gpu_src, Size(cla.blur_radius, cla.blur_radius));
    
    //Convert into Binary image on GPU...
    cv::gpu::threshold(gpu_src, gpu_src, int(cla.threshold_ratio * 255), 255, THRESH_BINARY_INV);
   

//    cv::gpu::Canny(gpu_src, gpu_src, 35, 200, 3);
 
//    gettimeofday(&tp1, NULL);
//    ms1 = tp1.tv_sec * 1000 + tp1.tv_usec / 1000;
//    timeToUpload = timeToUpload + ms1-ms;

//    struct timeval tp1;
//    long int ms1=0;
//    struct timeval tp;
//
//    gettimeofday(&tp, NULL);
//    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
//
//    cv::gpu::Canny(gpu_src, gpu_src, 35, 200, 3);
//
//
//    //Copy from GPU memory to CPU memory...
//  //  cv::Mat cpu_src(gpu_src);
//
//    gettimeofday(&tp1, NULL);
//    ms1 = tp1.tv_sec * 1000 + tp1.tv_usec / 1000;
//    timeToUpload = timeToUpload + ms1-ms;

    cv::Mat cpu_src(gpu_src);

    
    vector<vector <Point> > contours;
    
    vector<Vec4i> hierarchy;
    
    //findContours on CPU...
    //Need to write this method on GPU... To improve overall performance...
    //Check Sobel Algorithm...
    cv::findContours(cpu_src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    
    
   
    if (contours.size() > 0) {
        int largest_contour_index = 0;
        int largest_area = 0;
        
        for(int i = 0; i < contours.size(); i ++) {
            double a = cv::contourArea(contours[i], false);
            
            if(a > largest_area) {
                largest_area = a;
                
                largest_contour_index = i;
            }
        }
        
        cv::Rect bRect = cv::boundingRect(contours[largest_contour_index]);
        

    //    cout << "contours="<<contours[largest_contour_index]<<endl;   
         
        *pX = bRect.x + (bRect.width / 2);
        *pY = bRect.y + (bRect.height / 2);
        *pArea = largest_area;
    }
    else {
        *pX = -1;
        *pY = -1;
        *pArea = -1;
    }
    
    return 0;
}



int main(int argc, char **argv) {
    wormSegmenter();

    cout << "time="<<timeToUpload<<endl;

    return 0;
}
