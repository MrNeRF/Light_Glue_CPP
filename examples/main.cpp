#include "ALIKED.hpp"
#include "LightGlue.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

// Helper function to load and preprocess image
cv::Mat load_image(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + path);
    }

    // Convert BGR to RGB
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    return img_rgb;
}

// Helper function to draw matches between images
void draw_matches(cv::Mat& img1, cv::Mat& img2,
                  const torch::Tensor& kpts0, const torch::Tensor& kpts1,
                  const torch::Tensor& matches, const torch::Tensor& scores) {

    // Create output image
    int height = std::max(img1.rows, img2.rows);
    int width = img1.cols + img2.cols;
    cv::Mat output(height, width, CV_8UC3);

    // Copy images side by side
    img1.copyTo(output(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(output(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    // Draw matches
    auto kpts0_cpu = kpts0.cpu();
    auto kpts1_cpu = kpts1.cpu();
    auto matches_cpu = matches.cpu();
    auto scores_cpu = scores.cpu();

    for (int i = 0; i < matches_cpu.size(0); i++)
    {
        int idx0 = matches_cpu[i][0].item<int64_t>();
        int idx1 = matches_cpu[i][1].item<int64_t>();

        cv::Point2f pt1(kpts0_cpu[idx0][0].item<float>(), kpts0_cpu[idx0][1].item<float>());
        cv::Point2f pt2(kpts1_cpu[idx1][0].item<float>() + img1.cols, kpts1_cpu[idx1][1].item<float>());

        // Color based on score
        float score = scores_cpu[idx0].item<float>();
        cv::Scalar color(0, 255 * score, 0);

        cv::line(output, pt1, pt2, color, 1, cv::LINE_AA);
        cv::circle(output, pt1, 3, color, -1, cv::LINE_AA);
        cv::circle(output, pt2, 3, color, -1, cv::LINE_AA);
    }

    // Show result
    cv::imshow("Matches", output);
    cv::waitKey(0);

    // Save result
    cv::imwrite("matches.png", output);
}

int main(int argc, char* argv[]) {
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image1_path> <image2_path>" << std::endl;
        return 1;
    }

    try
    {
        // Device selection
        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Initialize models
        auto extractor = std::make_shared<ALIKED>("aliked-n32", device.str());
        auto matcher = std::make_shared<LightGlue>();

        // Move matcher to device
        matcher->to(device);

        // Load and process images
        cv::Mat img0 = load_image(argv[1]);
        cv::Mat img1 = load_image(argv[2]);

        // Extract features
        std::cout << "Extracting features..." << std::endl;
        auto feats0 = extractor->run(img0);
        auto feats1 = extractor->run(img1);

        // Match features
        std::cout << "Matching features..." << std::endl;
        feats0.insert("image_size", torch::tensor({img0.cols, img0.rows}, torch::kFloat32));
        feats1.insert("image_size", torch::tensor({img1.cols, img1.rows}, torch::kFloat32));
        auto matches01 = matcher->forward(feats0, feats1);

        // Get matches and scores
        const auto& kpts0 = feats0.at("keypoints");
        const auto& kpts1 = feats1.at("keypoints");
        const auto& matches = matches01.at("matches0");
        const auto& matching_scores = matches01.at("matching_scores0");

        // Print statistics
        std::cout << "Number of keypoints in image 0: " << kpts0.size(0) << std::endl;
        std::cout << "Number of keypoints in image 1: " << kpts1.size(0) << std::endl;

        // Count valid matches (where matches != -1)
        auto valid_matches = (matches != -1).sum().item<int64_t>();
        std::cout << "Number of matches: " << valid_matches << std::endl;
        std::cout << "Stopped after " << matches01.at("stop").item<int64_t>() << " layers" << std::endl;

        // Visualize matches
        draw_matches(img0, img1, kpts0, kpts1, matches, matching_scores);

    } catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}