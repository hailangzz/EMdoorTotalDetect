#include "utils.hpp"


ConfigInfo readConfig(const std::string& filename) {
    ConfigInfo cfg_values;

    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return cfg_values;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // 跳过空行和注释

        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            config[key] = value;
        }
    }

    cfg_values.model_path  = config["model_path"].c_str(); 
    cfg_values.input_width  = std::stoi(config["input_width"]);  
    cfg_values.input_height  = std::stoi(config["input_height"]);  
    cfg_values.anchors_path  = config["anchors"];                // 模型anchors路径
    cfg_values.num_boxes = std::stoi(config["num_boxes"]);               // 你的模型输出框数量
    cfg_values.num_keypoints = std::stoi(config["num_keypoints"]);       // BlazePalm 每个手 7 个关键点
    cfg_values.resolution = std::stof(config["resolution"]);
    cfg_values.score_threshold = std::stof(config["score_threshold"]); // 输入图片尺寸
    cfg_values.max_frame_threshold = std::stoi(config["max_frame_threshold"]);
    cfg_values.debug_nv21_image_saver  = config["debug_nv21_image_saver"].c_str(); 

    return cfg_values;
}

//  RGA图像数据,硬件处理加速
bool nv21_to_rgb_resize(
    uint8_t* nv21,
    int src_w, int src_h,
    uint8_t* rgb,
    int dst_w, int dst_h
) {
    // wrap source NV21
    rga_buffer_t src = wrapbuffer_virtualaddr(
        nv21,
        src_w, src_h,
        RK_FORMAT_YCrCb_420_SP // NV21
    );

    // wrap destination RGB888
    rga_buffer_t dst = wrapbuffer_virtualaddr(
        rgb,
        dst_w, dst_h,
        RK_FORMAT_RGB_888
    );

    IM_STATUS ret = imresize(src, dst);
    if (ret != IM_STATUS_SUCCESS) {
        printf("RGA imresize failed: %s\n", imStrError(ret));
        return false;
    }

    return true;
}

void rgb_to_float(const uint8_t* rgb, float* out, int w, int h) {
    int pixel_cnt = w * h;
    for (int i = 0; i < pixel_cnt; i++) {
        out[i * 3 + 0] = rgb[i * 3 + 0] / 255.0f;
        out[i * 3 + 1] = rgb[i * 3 + 1] / 255.0f;
        out[i * 3 + 2] = rgb[i * 3 + 2] / 255.0f;
    }
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

std::vector<float> preprocess_image(const cv::Mat& img, int target_w, int target_h) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h));
    cv::Mat img_float;
    resized.convertTo(img_float, CV_32FC3, 1.0 / 255.0); // 归一化

    std::vector<float> input_data(target_w * target_h * 3);
    int idx = 0;
    for (int h = 0; h < target_h; ++h) {
        for (int w = 0; w < target_w; ++w) {
            cv::Vec3f pixel = img_float.at<cv::Vec3f>(h, w);
            input_data[idx++] = pixel[2]; // R
            input_data[idx++] = pixel[1]; // G
            input_data[idx++] = pixel[0]; // B
        }
    }
    return input_data;
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

 void plotDetectBoxs(const std::string & image_path,const std::vector<PalmBox>& boxes){

    cv::Mat image = cv::imread(image_path);
    
    // 5. 打印结果
    for (size_t i = 0; i < boxes.size(); i++) {
        const PalmBox& box = boxes[i];
        std::cout << "Box " << i 
                  << ": score=" << box.score 
                  << " x=" << box.x 
                  << " y=" << box.y 
                  << " w=" << box.w 
                  << " h=" << box.h 
                  << std::endl;
        std::cout << "  keypoints: ";
        for (float kp : box.keypoints) std::cout << kp << " ";
        std::cout << std::endl;
    }

    // 绘制检测结果
    drawPalmBoxes(image, boxes);
    // // 显示图像
    // cv::imshow("Palm Detection", image);
    // cv::waitKey(0);

    // 可选：保存到文件
    cv::imwrite("palm_detect_result.jpg", image);
    

}

cv::Mat plotDetectBoxsMat(cv::Mat & image_mat,const std::vector<PalmBox>& boxes){

    cv::Mat image = image_mat;
    
    // 5. 打印结果
    for (size_t i = 0; i < boxes.size(); i++) {
        const PalmBox& box = boxes[i];
        std::cout << "Box " << i 
                  << ": score=" << box.score 
                  << " x=" << box.x 
                  << " y=" << box.y 
                  << " w=" << box.w 
                  << " h=" << box.h 
                  << std::endl;
        std::cout << "  keypoints: ";
        for (float kp : box.keypoints) std::cout << kp << " ";
        std::cout << std::endl;
    }

    // 绘制检测结果
    drawPalmBoxes(image, boxes);
    // // 显示图像
    // cv::imshow("Palm Detection", image);
    // cv::waitKey(0);

    // 可选：保存到文件
    // cv::imwrite("palm_detect_result.jpg", image);
    return image;
}

// 读取配置文件信息函数


double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// 返回值为 true 表示成功，读取本地图片，并转为NV21格式
// 返回 AndroidNV21Input
// 注意：返回的 nv21_input 需要调用者使用 delete[] 释放
AndroidImageNV21 load_image_to_nv21(const std::string& image_path) {
    AndroidImageNV21 result;
    result.image_input_nv21 = nullptr;
    result.image_width = 0;
    result.image_height = 0;

    // 1. 读取图片
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return result;
    }

    // 2. 保证宽高为偶数
    int width = bgr.cols & ~1;    // 向下取偶数
    int height = bgr.rows & ~1;
    cv::Mat bgr_even = bgr(cv::Rect(0, 0, width, height));

    // 3. 转为 I420（YUV420p）
    cv::Mat yuv420p;
    cv::cvtColor(bgr_even, yuv420p, cv::COLOR_BGR2YUV_I420);

    // 4. 分配 NV21 内存
    size_t nv21_size = width * height * 3 / 2;
    uint8_t* nv21_buf = new uint8_t[nv21_size];

    // 5. 复制 Y 分量
    memcpy(nv21_buf, yuv420p.data, width * height);

    // 6. 交错 VU
    uint8_t* src_u = yuv420p.data + width * height;
    uint8_t* src_v = src_u + (width * height) / 4;
    uint8_t* dst_uv = nv21_buf + width * height;

    for (int i = 0; i < (width * height) / 4; i++) {
        dst_uv[i * 2] = src_v[i];     // V
        dst_uv[i * 2 + 1] = src_u[i]; // U
    }

    // 7. 填充返回结构体
    result.image_input_nv21 = nv21_buf;
    result.image_width = width;
    result.image_height = height;

    return result;
}
