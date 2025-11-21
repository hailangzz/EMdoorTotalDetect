#include "utils.hpp"

bool GetVideoInfo(const std::string& video_path, int& width, int& height, double& fps, int& frame_count) {


    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return false;
    }

    width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps    = cap.get(cv::CAP_PROP_FPS);
    frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    return true;
    
}


bool resizeImage(const std::string& image_path,
                     std::vector<float>& output_data,
                     std::vector<int64_t>& input_shape)
{
    // 读取图像
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return false;
    }

    // 调整图像尺寸到 192x192
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(192, 192));

    // 转换为 float 并归一化到 [0,1]（根据模型要求可以调整）
    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);

    // HWC -> NHWC，flatten
    input_shape = {1, 192, 192, 3}; // batch, height, width, channels
    output_data.resize(1 * 192 * 192 * 3);

    // OpenCV 默认是 BGR，需要转换成 RGB
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    size_t idx = 0;
    for (int h = 0; h < resized_img.rows; h++) {
        for (int w = 0; w < resized_img.cols; w++) {
            cv::Vec3f pixel = resized_img.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; c++) {
                output_data[idx++] = pixel[c]; // R,G,B 顺序
            }
        }
    }

    return true;
}




// 计算两个 bbox 的 IOU
static float computeIOU(const PalmBox& a, const PalmBox& b) {
    float ax1 = a.x - a.w / 2.0f;
    float ay1 = a.y - a.h / 2.0f;
    float ax2 = a.x + a.w / 2.0f;
    float ay2 = a.y + a.h / 2.0f;

    float bx1 = b.x - b.w / 2.0f;
    float by1 = b.y - b.h / 2.0f;
    float bx2 = b.x + b.w / 2.0f;
    float by2 = b.y + b.h / 2.0f;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float a_area = (ax2 - ax1) * (ay2 - ay1);
    float b_area = (bx2 - bx1) * (by2 - by1);

    return inter_area / (a_area + b_area - inter_area + 1e-6f);
}

// NMS 实现
std::vector<PalmBox> nms(const std::vector<PalmBox>& boxes, float iou_threshold) {
    std::vector<PalmBox> result;
    if (boxes.empty()) return result;

    // 按 score 降序排序
    std::vector<PalmBox> sorted_boxes = boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end(),
              [](const PalmBox& a, const PalmBox& b) { return a.score > b.score; });

    std::vector<bool> suppressed(sorted_boxes.size(), false);

    for (size_t i = 0; i < sorted_boxes.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(sorted_boxes[i]);

        for (size_t j = i + 1; j < sorted_boxes.size(); j++) {
            if (suppressed[j]) continue;
            if (computeIOU(sorted_boxes[i], sorted_boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}



// sigmoid 函数
float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

std::vector<float> loadAnchorsBin(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    std::vector<float> anchors;

    if (!fin.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return anchors;
    }

    std::streamsize size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    anchors.resize(size / sizeof(float));
    fin.read(reinterpret_cast<char*>(anchors.data()), size);
    fin.close();

    return anchors;
}


// 绘制检测结果
// 绘制检测结果，box 坐标归一化到 [0,1]
void drawPalmBoxes(cv::Mat& image, const std::vector<PalmBox>& boxes) {
    int img_w = image.cols;
    int img_h = image.rows;

    for (const auto& box : boxes) {
        // 转换归一化坐标到像素坐标
        int x = static_cast<int>(box.x * img_w);
        int y = static_cast<int>(box.y * img_h);
        int w = static_cast<int>(box.w * img_w);
        int h = static_cast<int>(box.h * img_h);

        // 绘制 bounding box
        cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

        // 绘制关键点
        for (size_t k = 0; k < box.keypoints.size() / 2; k++) {
            int kp_x = static_cast<int>(box.keypoints[k * 2] * img_w);
            int kp_y = static_cast<int>(box.keypoints[k * 2 + 1] * img_h);
            cv::circle(image, cv::Point(kp_x, kp_y), 3, cv::Scalar(0, 0, 255), -1);
        }

        // 绘制分数
        std::string text = cv::format("%.2f", box.score);
        cv::putText(image, text, cv::Point(x, y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
}








