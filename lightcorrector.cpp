#include "lightcorrector.h"
#include <iostream>

using namespace cv;
using namespace std;

HPFCorrector::HPFCorrector(Size size, int D, int n, float highHVTb, float lowHVTb)
    :m_D(D), m_n(n), m_highHVTb(highHVTb), m_lowHVTb(lowHVTb)
{
    int padRows = getOptimalDFTSize(size.height);
    int padCols = getOptimalDFTSize(size.width);
    m_size = Size(padCols, padRows);
    MakeHPF(m_size, m_D, m_n, m_highHVTb, m_lowHVTb);

    m_cropRect = Rect(10,10,size.width-20,size.height-20);
}

void HPFCorrector::DoCorrction(Mat &image)
{
    Rect original(0,0, image.cols, image.rows);
    Mat imgOut, imgComplex;
    CalcFFT(image, imgComplex);
    fftshift(imgComplex, imgComplex);

    mulSpectrums(imgComplex,m_filter, imgComplex, 0);
    fftshift(imgComplex, imgComplex);
    CalcIFFT(imgComplex,  imgOut);
    image = imgOut(m_cropRect);
    image.convertTo(image, CV_8UC1);
    normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1);
}

Mat HPFCorrector::DoFFTCorrection(Mat input)
{
    Mat imgOut, imgComplex;
    CalcFFT(input, imgComplex);
    fftshift(imgComplex, imgComplex);
    mulSpectrums(imgComplex,m_filter, imgComplex, 0);
    fftshift(imgComplex, imgComplex);
    CalcIFFT(imgComplex,  imgOut);
    return imgOut;
}

void HPFCorrector::CalcFFT(Mat input, Mat &complex)
{
    int padRows = getOptimalDFTSize(input.rows);
    int padCols = getOptimalDFTSize(input.cols);

    Mat imgIn, padded, imgComplex;
    input.convertTo(imgIn, CV_32F);
    imgIn +=1;
    log(imgIn, imgIn);
    copyMakeBorder(imgIn, padded, 0, padRows - imgIn.rows, 0, padCols - imgIn.cols, BORDER_CONSTANT);

    Mat planes [] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planes, 2, imgComplex);
    dft(imgComplex, complex, DFT_SCALE);
}

void HPFCorrector::CalcIFFT(Mat complex, Mat &output)
{
    dft(complex, output, DFT_INVERSE | DFT_REAL_OUTPUT);
    exp(output,output);
}

void HPFCorrector::MakeHPF(Size size, int D, int n, float high_h_v_TB, float low_h_v_TB)
{
    Mat filter(size, CV_32F);
    Point centre = Point(filter.rows/2, filter.cols/2);
    double radius;
    float upper = (high_h_v_TB * 0.01);
    float lower = (low_h_v_TB * 0.01);

    for(int i = 0; i < filter.rows; i++)
    {
        for(int j = 0; j < filter.cols; j++)
        {
            radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
            filter.at<float>(i,j) =((upper - lower) * (1/(1 + pow((double) (D/radius), (double) (2*n))))) + lower;
        }
    }

    Mat planes [] = {Mat_<float>(filter), Mat::zeros(filter.size(), CV_32F)};
    merge(planes, 2, m_filter);
}

void HPFCorrector::fftshift(const Mat &inputImg, Mat &outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void HPFCorrector::processThreeChannels()
{
    Mat channels[3];
    cvtColor(m_image,m_image,CV_BGR2HSV);
    split(m_image,channels);
    DoCorrction(channels[2]);
    channels[0] = channels[0](m_cropRect);
    channels[1] = channels[1](m_cropRect);
    merge(channels, 3, m_image);
    cvtColor(m_image,m_image, CV_HSV2BGR);
}

Mat BaseLightCorrector::apply(Mat &input)
{
    input.copyTo(m_image);
    switch (input.type())
    {
    case CV_8UC1:
        processOneChannel();
        break;
    case CV_8UC3:
        processThreeChannels();
        break;
    default:
        throw(std::runtime_error("Unsupported image type, shoul be CV_8UC1 or CV_8UC3, got " + std::to_string(input.type())));
    }
    return m_image;
}

void BaseLightCorrector::processOneChannel()
{
    DoCorrction(m_image);
}

void BaseLightCorrector::processThreeChannels()
{
    Mat channels[3];
    cvtColor(m_image,m_image,CV_BGR2HSV);
    split(m_image,channels);
    DoCorrction(channels[2]);
    merge(channels, 3, m_image);
    cvtColor(m_image,m_image, CV_HSV2BGR);
}


void SSRCorrector::DoCorrction(Mat &image)
{
    image = DoScale(image, m_size);
    cv::normalize(image, image, 0, 255, NORM_MINMAX, CV_8U);
}

Mat SSRCorrector::DoScale(Mat &image, int size)
{
    Mat img;
    image.copyTo(img);
    Mat blurred, logI, logB, logIxB, logR;
    GaussianBlur(img, blurred,Size(size,size), 0);
    img +=1;
    blurred += 1;
    img.convertTo(img, CV_64F);
    blurred.convertTo(blurred,CV_64F);
    cv::log(img/255, logI);
    cv::log(blurred/255, logB);
    cv::multiply(logI, logB, logIxB);
    cv::subtract(logI, logIxB, logR);

    return logR;
}

void SSRCorrector::replaceZeroes(Mat& img)
{
    Mat mask = img > 0;
    double minc[1], maxc[1];
    minMaxLoc(img, minc, maxc,0,0,mask);
    img.setTo(minc[0],img==0);
}

void MSRCorrector::DoCorrction(Mat &image)
{
    Mat step, full = Mat::zeros(image.size(), CV_64F);
    for (int i = 0; i<m_scales; i++)
    {
        step = DoScale(image, m_size+i*2);
        full += step;
    }
    full = full / m_scales;
    cv::normalize(full, image, 0, 255, NORM_MINMAX, CV_8U);
}
