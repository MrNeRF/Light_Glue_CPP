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
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <iostream>
#include <string>

// Helper function to generate a colormap for pruning visualization
cv::Scalar cm_prune(float value, float max_value) {
    float norm_value = (value == max_value) ? -1.0f : (value - 1.0f) / 9.0f;
    if (norm_value < 0)
    { // Blue for pruned points
        return cv::Scalar(255, 0, 0);
    }
    float green = std::min(1.0f, std::max(0.0f, norm_value * 2.0f));
    float red = 1.0f - green;
    return cv::Scalar(0, green * 255, red * 255);
}

void draw_matches_and_prune(cv::Mat& img1, cv::Mat& img2,
                            const torch::Tensor& kpts0, const torch::Tensor& kpts1,
                            const torch::Tensor& matches, const torch::Tensor& scores,
                            const torch::Tensor& prune0, const torch::Tensor& prune1,
                            int stop_layer) {
    int height = std::max(img1.rows, img2.rows);
    int width = img1.cols + img2.cols;
    cv::Mat output(height, width, CV_8UC3);
    img1.copyTo(output(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(output(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    // Print tensor sizes
    std::cout << "kpts0 size: " << kpts0.sizes() << std::endl;
    std::cout << "kpts1 size: " << kpts1.sizes() << std::endl;
    std::cout << "matches size: " << matches.sizes() << std::endl;
    std::cout << "scores size: " << scores.sizes() << std::endl;
    std::cout << "prune0 size: " << prune0.sizes() << std::endl;
    std::cout << "prune1 size: " << prune1.sizes() << std::endl;

    // Print tensor values for small tensors
    if (matches.numel() <= 10)
    {
        std::cout << "matches: " << matches << std::endl;
    }
    if (scores.numel() <= 10)
    {
        std::cout << "scores: " << scores << std::endl;
    }

    // Move tensors to CPU
    auto kpts0_cpu = kpts0.cpu();
    auto kpts1_cpu = kpts1.cpu();
    auto matches_cpu = matches.cpu();
    auto scores_cpu = scores.cpu();
    auto prune0_cpu = prune0.cpu();
    auto prune1_cpu = prune1.cpu();

    // Debug first few values
    float max_prune0 = prune0_cpu.flatten().max().item<float>();
    float max_prune1 = prune1_cpu.flatten().max().item<float>();

    // Get number of matches from the second dimension
    int num_matches = matches_cpu.size(1);

    const float min_score_threshold = 0.1f;
    // Draw matches
    for (int i = 0; i < num_matches; i++)
    {
        // Get the match index directly
        int64_t idx = matches_cpu[0][i].item<int64_t>();
        int64_t idx0 = i;
        int64_t idx1 = idx;

        // Get the score for this match
        float score = scores_cpu[0][i].item<float>();

        // Skip low confidence matches
        if (score < min_score_threshold)
        {
            continue;
        }

        // Debug keypoint indices
        if (idx0 >= kpts0_cpu.size(0) || idx1 >= kpts1_cpu.size(0))
        {
            std::cerr << "Error: Keypoint index out of range. idx0: " << idx0 << ", idx1: " << idx1 << std::endl;
            continue;
        }

        float x0 = kpts0_cpu[idx0][0].item<float>();
        float y0 = kpts0_cpu[idx0][1].item<float>();
        float x1 = kpts1_cpu[idx1][0].item<float>();
        float y1 = kpts1_cpu[idx1][1].item<float>();

        cv::Point2f pt1(x0, y0);
        cv::Point2f pt2(x1 + img1.cols, y1);

        // Draw the match only if score is above threshold
        cv::Scalar color(0, 255 * score, 0);
        cv::line(output, pt1, pt2, color, 1, cv::LINE_AA);
        cv::circle(output, pt1, 3, color, -1, cv::LINE_AA);
        cv::circle(output, pt2, 3, color, -1, cv::LINE_AA);
    }

    // Visualize pruning
    // for (int i = 0; i < kpts0_cpu.size(0); i++)
    //{
    //    float x0 = kpts0_cpu[i][0].item<float>();
    //    float y0 = kpts0_cpu[i][1].item<float>();
    //    cv::Scalar color = cm_prune(prune0_cpu[0][i].item<float>(), max_prune0);
    //    cv::circle(output, cv::Point2f(x0, y0), 5, color, -1, cv::LINE_AA);
    //}
    // for (int i = 0; i < kpts1_cpu.size(0); i++)
    //{
    //    float x1 = kpts1_cpu[i][0].item<float>() + img1.cols;
    //    float y1 = kpts1_cpu[i][1].item<float>();
    //    cv::Scalar color = cm_prune(prune1_cpu[0][i].item<float>(), max_prune1);
    //    cv::circle(output, cv::Point2f(x1, y1), 5, color, -1, cv::LINE_AA);
    //}

    // Add text annotation
    std::string text = "Stopped after " + std::to_string(stop_layer) + " layers";
    cv::putText(output, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    // Show and save the result
    cv::imshow("Matches and Pruning", output);
    cv::waitKey(0);
    cv::imwrite("matches_and_pruning.png", output);
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
        auto extractor = std::make_shared<ALIKED>("aliked-n16", device.str());
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
        feats0.insert("image_size", torch::tensor({static_cast<float>(img0.cols), static_cast<float>(img0.rows)}, torch::kFloat32).unsqueeze(0));
        feats1.insert("image_size", torch::tensor({static_cast<float>(img1.cols), static_cast<float>(img1.rows)}, torch::kFloat32).unsqueeze(0));
        auto matches01 = matcher->forward(feats0, feats1);

        // Get keypoints, matches, scores, and pruning information
        const auto& kpts0 = feats0.at("keypoints");
        const auto& kpts1 = feats1.at("keypoints");
        const auto& matches = matches01.at("matches0");
        const auto& matching_scores = matches01.at("matching_scores0");
        const auto& prune0 = matches01.at("prune0");
        const auto& prune1 = matches01.at("prune1");
        int stop_layer = matches01.at("stop").item<int64_t>();

        // Print statistics
        std::cout << "Number of keypoints in image 0: " << kpts0.size(0) << std::endl;
        std::cout << "Number of keypoints in image 1: " << kpts1.size(0) << std::endl;

        // Count valid matches (where matches != -1)
        auto valid_matches = (matches != -1).sum().item<int64_t>();
        std::cout << "Number of valid matches: " << valid_matches << std::endl;
        std::cout << "Stopped after " << stop_layer << " layers" << std::endl;

        // Visualize matches and pruning
        draw_matches_and_prune(img0, img1, kpts0, kpts1, matches, matching_scores, prune0, prune1, stop_layer);

    } catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
