#include "net.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <vector>
#include <iostream>

using namespace std;


cv::Mat ncnn2cv(const ncnn::Mat& ncnn_image)
{
    int w = ncnn_image.w;
    int h = ncnn_image.h;
    int c = ncnn_image.c;

    if (c != 3)
    {
        std::cerr << "Error: Only 3-channel (RGB) images are supported!" << std::endl;
        return cv::Mat();
    }

    cv::Mat bgr_image(h, w, CV_8UC3);

    for (int y = 0; y < h; ++y)
    {
        const float* r = ncnn_image.channel(0).row(y);
        const float* g = ncnn_image.channel(1).row(y);
        const float* b = ncnn_image.channel(2).row(y);

        for (int x = 0; x < w; ++x)
        {
            // Convert float [-1, 1] to unsigned char [0, 255]
            unsigned char R = static_cast<unsigned char>((r[x] + 1.0f) / 2.0f * 255.0f);
            unsigned char G = static_cast<unsigned char>((g[x] + 1.0f) / 2.0f * 255.0f);
            unsigned char B = static_cast<unsigned char>((b[x] + 1.0f) / 2.0f * 255.0f);

            bgr_image.at<cv::Vec3b>(y, x) = cv::Vec3b(B, G, R);
        }
    }

    return bgr_image;
}



int main()
{

    ncnn::Net net;
    net.load_param("hayao.ncnn.param");
    net.load_model("hayao.ncnn.bin");
    
    const int target_size = 256;

    cv::Mat bgr = cv::imread("test.jpg", CV_LOAD_IMAGE_COLOR);
    if (bgr.empty())
    {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // Resize the image
    cv::Mat resized_img;
    cv::resize(bgr, resized_img, cv::Size(target_size, target_size));

    ncnn::Mat in = ncnn::Mat::from_pixels(resized_img.data, ncnn::Mat::PIXEL_BGR2RGB, target_size, target_size);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);


    cv::Mat output_img = ncnn2cv(out);
    cv::imshow("Output Image", output_img);
    cv::waitKey(0);

}
