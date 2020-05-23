#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void doSSR(cv::Mat& img);

int main()
{
    Mat src, dst;
//    String imageName( "Lenna.png" ); // by default
    String imageName( "Picture1.png" ); // by default
    src = imread(imageName, IMREAD_COLOR);
    if (src.empty())
        return -1;

    imshow("Source", src);
    cvtColor(src,src,CV_BGR2HSV);

    Mat channels[3];
    split(src, channels);
//    for (auto& channel : channels)
//        doSSR(channel);
    equalizeHist(channels[1],channels[1]);
    doSSR(channels[2]);
    cv::merge(channels,3,dst);
    cvtColor(dst,dst,CV_HSV2BGR);

//    cvtColor(src, dst, CV_BGR2GRAY);
//    doSSR(dst);
    imshow("Result", dst);

    waitKey(0);

//    cout << "Hello World!" << endl;
    return 0;
}

void replaceZeroes(Mat& img)
{
    Mat mask = img > 0;
    double minc[1], maxc[1];
    minMaxLoc(img, minc, maxc,0,0,mask);
    img.setTo(minc[0],img==0);
}

void doSSR(Mat& img)
{
//    CV_ASSERT(img.type()==CV_8UC1);
    Mat blurred, logI, logB, logIxB, logR;
    GaussianBlur(img, blurred,Size(5,5), 0);
    replaceZeroes(img);
    replaceZeroes(blurred);

    double minc[1], maxc[1];

    img.convertTo(img, CV_64F);
    blurred.convertTo(blurred,CV_64F);
    cv::log(img/255, logI);
    cv::log(blurred/255, logB);
    cv::multiply(logI, logB, logIxB);
    cv::subtract(logI, logIxB, logR);

    cv::normalize(logR, img, 0, 127
                  , NORM_MINMAX, CV_8U);
//    cv::normalize(logR, logR, 0, 1, NORM_MINMAX, CV_64F);
//    convertScaleAbs(logR, img);
}

