#include "ALIKED.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

class ImageLoader {
public:
    explicit ImageLoader(const std::string& filepath) {
        for (const auto& entry : fs::directory_iterator(filepath))
        {
            const auto& path = entry.path();
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".png" || ext == ".jpg" || ext == ".ppm")
            {
                images_.push_back(path.string());
            }
        }
        std::sort(images_.begin(), images_.end());
        std::cout << "Loading " << images_.size() << " images" << std::endl;
    }

    cv::Mat operator[](size_t idx) const {
        return cv::imread(images_[idx]);
    }

    size_t size() const { return images_.size(); }

private:
    std::vector<std::string> images_;
};

class SimpleTracker {
public:
    SimpleTracker() : pts_prev_(),
                      desc_prev_() {}

    // Update function
    std::tuple<cv::Mat, int> update(const cv::Mat& img, const torch::Tensor& pts, const torch::Tensor& desc) {
        cv::Mat out = img.clone();
        int N_matches = 0;

        if (!pts_prev_.defined())
        {
            // First frame: Initialize points and descriptors
            pts_prev_ = pts.clone();
            desc_prev_ = desc.clone();

            // Draw keypoints
            for (int i = 0; i < pts.size(0); ++i)
            {
                cv::Point2f p1(pts[i][0].item<float>(), pts[i][1].item<float>());
                cv::circle(out, p1, 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            }
        } else
        {
            // Compute matches
            auto matches = mnn_matcher(desc_prev_, desc);
            N_matches = matches.size(0);

            // Draw matches
            for (int i = 0; i < N_matches; ++i)
            {
                int idx0 = matches[i][0].item<int>();
                int idx1 = matches[i][1].item<int>();

                cv::Point2f pt1(pts_prev_[idx0][0].item<float>(), pts_prev_[idx0][1].item<float>());
                cv::Point2f pt2(pts[idx1][0].item<float>(), pts[idx1][1].item<float>());
                cv::line(out, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                cv::circle(out, pt2, 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            }

            // Update previous points and descriptors
            pts_prev_ = pts.clone();
            desc_prev_ = desc.clone();
        }

        return {out, N_matches};
    }

private:
    // Nearest neighbor matcher
    static torch::Tensor mnn_matcher(const torch::Tensor& desc1, const torch::Tensor& desc2) {
        // Compute similarity matrix
        auto sim = torch::matmul(desc1, desc2.t());
        sim = torch::where(sim < 0.9, torch::zeros_like(sim), sim);

        // Nearest neighbors
        auto nn12 = std::get<1>(torch::max(sim, 1)); // Nearest in desc2 for each desc1
        auto nn21 = std::get<1>(torch::max(sim, 0)); // Nearest in desc1 for each desc2

        // Mask to enforce mutual nearest neighbors
        auto ids1 = torch::arange(sim.size(0), torch::TensorOptions().device(sim.device()));
        auto mask = (ids1 == nn21.index({nn12}));
        auto matches = torch::stack({ids1.masked_select(mask), nn12.masked_select(mask)}, 1);

        return matches;
    }

    torch::Tensor pts_prev_;
    torch::Tensor desc_prev_;
};

int main(int argc, char* argv[]) {
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_dir> [options]" << std::endl;
        return 1;
    }

    // Parse command line arguments
    const std::string input_dir = argv[1];
    const std::string model_name = "aliked-n32";
    const std::string device = "cuda";
    const int top_k = -1;
    const float scores_th = 0.2f;
    const int n_limit = 5000;

    // Initialize model
    std::cout << "Initializing ALIKED model..." << std::endl;
    const auto model = std::make_shared<ALIKED>(model_name, device, top_k, scores_th, n_limit);

    // Load images
    ImageLoader image_loader(input_dir);
    if (image_loader.size() < 2)
    {
        std::cerr << "Need at least 2 images in the input directory" << std::endl;
        return 1;
    }

    // Initialize tracker
    SimpleTracker tracker;

    // Display prompt
    std::cout << "Press 'space' to start. \nPress 'q' or 'ESC' to stop!" << std::endl;

    // Initialize video writer
    cv::VideoWriter video_writer;
    bool is_writer_initialized = false;

    // Iterate over the images
    for (size_t i = 0; i < image_loader.size(); i++)
    {
        cv::Mat img = image_loader[i];
        if (img.empty())
            break;

        // Convert image to RGB
        cv::Mat img_rgb;
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

        // Run model
        const auto pred = model->run(img_rgb);
        auto kpts = pred.at("keypoints").cpu();
        const auto desc = pred.at("descriptors").cpu();

        // Normalize and scale keypoints to pixel coordinates
        const int img_width = img.cols;
        const int img_height = img.rows;

        kpts = (kpts + 1.0) * 0.5;          // Normalize to [0, 1]
        kpts.select(1, 0).mul_(img_width);  // Scale x-coordinates
        kpts.select(1, 1).mul_(img_height); // Scale y-coordinates

        // Plot keypoints on the current image
        for (int j = 0; j < kpts.size(0); ++j)
        {
            const auto x = kpts[j][0].item<float>();
            const auto y = kpts[j][1].item<float>();

            // Validate coordinates
            if (x >= 0 && x < img_width && y >= 0 && y < img_height)
            {
                cv::circle(img, cv::Point2f(x, y), 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            } else
            {
                std::cerr << "Keypoint out of bounds: (" << x << ", " << y << ")" << std::endl;
            }
        }

        // Update tracker
        cv::Mat vis_img;
        int N_matches;
        std::tie(vis_img, N_matches) = tracker.update(img, kpts, desc);

        // Initialize video writer if not already initialized
        if (!is_writer_initialized)
        {
            int fps = 30; // Define the desired frame rate
            cv::Size frame_size(vis_img.cols, vis_img.rows);
            video_writer.open("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size, true);
            if (!video_writer.isOpened())
            {
                std::cerr << "Failed to open video writer." << std::endl;
                return -1;
            }
            is_writer_initialized = true;
        }

        // Write the frame to the video
        video_writer.write(vis_img);

        // Status message
        const std::string status = "matches/keypoints: " +
                                   std::to_string(N_matches) + "/" +
                                   std::to_string(kpts.size(0));

        // Overlay status and instructions
        cv::putText(vis_img, "Press 'q' or 'ESC' to stop.",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        cv::namedWindow(model_name);
        cv::setWindowTitle(model_name, model_name + ": " + status);
        cv::imshow(model_name, vis_img);

        // Handle user input
        const char c = static_cast<char>(cv::waitKey(1)); // Reduced wait time for smoother video creation
        if (c == 'q' || c == 27)
            break; // Quit on 'q' or 'ESC'
    }

    // Release video writer
    if (is_writer_initialized)
        video_writer.release();

    std::cout << "Video saved as output_video.avi" << std::endl;

    std::cout << "Finished!" << std::endl;
    std::cout << "Press any key to exit!" << std::endl;

    cv::destroyAllWindows();
    return 0;
}
