#ifndef LIGHTCORRECTOR_H
#define LIGHTCORRECTOR_H

#include "opencv2/imgproc.hpp"

using cv::Mat;
using cv::Size;
using cv::Rect;

class BaseLightCorrector
{
public:
    Mat apply(Mat& input);
protected:
    virtual void processOneChannel();
    virtual void processThreeChannels();
    virtual void DoCorrction(Mat& image) = 0;

    Mat m_image;
};

class HPFCorrector : public BaseLightCorrector
{
public:
    HPFCorrector(Size size, int D, int n, float highHVTb, float lowHVTb);


protected:
    virtual void DoCorrction(Mat &image) override;

    Mat DoFFTCorrection(Mat input);

    void CalcFFT(Mat input, Mat &complex);

    void CalcIFFT(Mat complex,  Mat &output);

    void MakeHPF(Size size, int D, int n, float highHVTb, float lowHVTb);

    void fftshift(const Mat& inputImg, Mat& outputImg);

    virtual void processThreeChannels() override;

    Size m_size;
    int m_D, m_n;
    float m_highHVTb, m_lowHVTb;
    Mat m_filter;
    Rect m_cropRect;
};

class SSRCorrector : public BaseLightCorrector
{
public:
    SSRCorrector(int size = 5) : m_size(size) {}

protected:
    virtual void DoCorrction(Mat &image) override;
    virtual Mat DoScale(Mat &image, int size);
    void replaceZeroes(Mat& img);
    int m_size;
};


class MSRCorrector : public SSRCorrector
{
public:
    MSRCorrector(int size = 5, int scales = 3) : m_size(size), m_scales(scales) {}

protected:
    virtual void DoCorrction(Mat &image) override;
    int m_size;
    int m_scales;
};

#endif // LIGHTCORRECTOR_H
