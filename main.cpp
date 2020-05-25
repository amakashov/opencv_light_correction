#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "lightcorrector.h"

using namespace std;
using namespace cv;

void doSSR(cv::Mat& img);

const String keys =
        "{@input | cameron.mp4 | input video}"
        "{method | ssr | correction method}"
        "{D | 10 | size for HPF}"
        "{size | 21 | size for retinex}"
        ;

int main(int argc, char* argv[])
{
    CommandLineParser parser( argc, argv, keys );

    Mat src, dst;
    string filename= parser.get<String>("@input");

    VideoCapture cap(filename);
    if (!cap.isOpened())
    {
        cout << "Cant open file "<< filename << endl;
        return -1;
    }
    cap >> src;
    char key = 0;

    Ptr<BaseLightCorrector>corrector;

    string method = parser.get<String>("method");
    int retSize = parser.get<int>("size");
    int hpfD = parser.get<int>("D");


    if (method == "ssr")
        corrector = new SSRCorrector(retSize);
    else if (method == "msr")
        corrector = new MSRCorrector(retSize);
    else if (method == "hpf")
        corrector = new HPFCorrector(src.size(), hpfD, 4, 210, 100);
    else
        throw(runtime_error("Unknown method "+method));

    while (!src.empty() && key != 27)
    {
        imshow("Original", src);

        dst = corrector->apply(src);

        imshow("After", dst);

        key = waitKey(40);
        cap >> src;
    }
    return 0;
}



