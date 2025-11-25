#include "blaze_hand_detect_rknn.hpp"


namespace HandDetectRknn {

// 析构函数释放 RKNN 资源
Detector::Detector(const ConfigInfo& config) {
    
    // auto cfg = readConfig(this->config_file_,this->cfg_values_);
    this->initModelparameter(config);
    this->anchors_ = loadAnchorsBin(this->cfg_values_.anchors_path);

}

// 析构函数释放 RKNN 资源
Detector::~Detector() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

void Detector::initModelparameter(ConfigInfo config_info){

    this->cfg_values_ = config_info;
}
// 加载模型
bool Detector::loadModel(const std::string& model_path) {
    FILE* fp = fopen(model_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Failed to open RKNN model file: " << model_path << std::endl;
        return false;
    }

    fseek(fp, 0, SEEK_END);
    size_t model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    model_data_.resize(model_size);
    if (fread(model_data_.data(), 1, model_size, fp) != model_size) {
        std::cerr << "Failed to read RKNN model file" << std::endl;
        fclose(fp);
        return false;
    }
    fclose(fp);

    // 创建 RKNN context
    int ret = rknn_init(&ctx_, model_data_.data(), model_size, 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_init failed! ret=" << ret << std::endl;
        return false;
    }

    // 获取输入输出信息
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_query RKNN_QUERY_IN_OUT_NUM failed! ret=" << ret << std::endl;
        return false;
    }

    std::cout << "RKNN model loaded successfully. inputs=" << io_num_.n_input
              << " outputs=" << io_num_.n_output << std::endl;

    return true;
}

// opencv读取本地图片，进行推理
std::vector<PalmBox>  Detector::infer(const std::vector<float>& input) {
    std::vector<PalmBox>  results;

    if (ctx_ == 0) {
        std::cerr << "RKNN context not initialized!" << std::endl;
        return results;
    }

    if (input.empty()) {
        std::cerr << "Invalid input tensor" << std::endl;
        return results;
    }

    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = const_cast<float*>(input.data());
    inputs[0].size = input.size() * sizeof(float);
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(ctx_, io_num_.n_input, inputs);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed! ret=" << ret << std::endl;
        return results;
    }

    // 执行推理
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_run failed! ret=" << ret << std::endl;
        return results;
    }

    // 获取输出
    std::vector<rknn_output> outputs(io_num_.n_output);
    for (int i = 0; i < io_num_.n_output; i++) {
        outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed! ret=" << ret << std::endl;
        return results;
    }
    
    auto detections = this->parseRknnOutputs(
        outputs,
        anchors_,           // 你加载的 anchor
        cfg_values_.num_boxes,          // 例如 2944
        cfg_values_.num_keypoints,      // 例如 7
        cfg_values_.resolution,             // resolution
        cfg_values_.score_threshold                // score threshold
    );

    // TODO: 根据你的模型，解析 outputs[i].buf 到 Detection 结构
    // outputs[i].buf 是 float* 类型，长度是 outputs[i].size / sizeof(float)

    // 释放输出
    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

    return detections;
}

std::vector<PalmBox> Detector::infer_image_rga_zero_copy(
    const std::string& image_path  // 本地图像路径
) {
    std::vector<PalmBox> detections;

    if (ctx_ == 0) {
        std::cerr << "RKNN context not initialized!" << std::endl;
        return detections;
    }

    const int net_w = cfg_values_.resolution;
    const int net_h = cfg_values_.resolution;

    // 1️⃣ 读取图像
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return detections;
    }

    // 2️⃣ BGR → RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 3️⃣ 静态分配 RGA 临时 buffer (UINT8) 和 RKNN float32 输入 buffer
    static uint8_t* rga_buf = nullptr;
    static size_t rga_size = 0;
    if (!rga_buf || rga_size != net_w * net_h * 3) {
        if (rga_buf) free(rga_buf);
        rga_size = net_w * net_h * 3;
        posix_memalign((void**)&rga_buf, 64, rga_size);
    }

    static float* rknn_input_buf = nullptr;
    static size_t rknn_size = 0;
    if (!rknn_input_buf || rknn_size != net_w * net_h * 3 * sizeof(float)) {
        if (rknn_input_buf) free(rknn_input_buf);
        rknn_size = net_w * net_h * 3 * sizeof(float);
        posix_memalign((void**)&rknn_input_buf, 64, rknn_size);
    }

    // 4️⃣ 用 RGA 做 Resize (RGB888 -> rga_buf)
    rga_buffer_t src = wrapbuffer_virtualaddr(
        img.data,
        img.cols, img.rows,
        RK_FORMAT_RGB_888
    );

    rga_buffer_t dst = wrapbuffer_virtualaddr(
        rga_buf,
        net_w, net_h,
        RK_FORMAT_RGB_888
    );

    IM_STATUS ret = imresize(src, dst);
    if (ret != IM_STATUS_SUCCESS) {
        std::cerr << "RGA imresize failed: " << imStrError(ret) << std::endl;
        return detections;
    }

    // 5️⃣ 将 RGA 输出转换到 float32 buffer，并归一化到 [0,1]
    uint8_t* rgb_ptr = rga_buf;
    for (int i = 0; i < net_w * net_h; i++) {
        rknn_input_buf[i * 3 + 0] = rgb_ptr[i * 3 + 0] / 255.0f;
        rknn_input_buf[i * 3 + 1] = rgb_ptr[i * 3 + 1] / 255.0f;
        rknn_input_buf[i * 3 + 2] = rgb_ptr[i * 3 + 2] / 255.0f;
    }

    // 6️⃣ 设置 RKNN 输入
    rknn_input input[1];
    memset(input, 0, sizeof(input));
    input[0].index = 0;
    input[0].buf   = rknn_input_buf;  // float32 buffer
    input[0].size  = rknn_size;
    input[0].pass_through = 0;
    input[0].type  = RKNN_TENSOR_FLOAT32;
    input[0].fmt   = RKNN_TENSOR_NHWC;

    int r = rknn_inputs_set(ctx_, io_num_.n_input, input);
    if (r != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed! ret=" << r << std::endl;
        return detections;
    }

    // 7️⃣ 执行推理
    r = rknn_run(ctx_, nullptr);
    if (r != RKNN_SUCC) {
        std::cerr << "rknn_run failed! ret=" << r << std::endl;
        return detections;
    }

    // 8️⃣ 获取输出
    std::vector<rknn_output> outputs(io_num_.n_output);
    for (int i = 0; i < io_num_.n_output; i++) outputs[i].want_float = 1;

    r = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (r != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed! ret=" << r << std::endl;
        return detections;
    }

    // 9️⃣ 解析输出
    detections = this->parseRknnOutputs(
        outputs,
        anchors_,
        cfg_values_.num_boxes,
        cfg_values_.num_keypoints,
        cfg_values_.resolution,
        cfg_values_.score_threshold
    );

    // 🔟 释放 RKNN 输出
    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

    return detections;
}

std::vector<PalmBox> Detector::infer_nv21(
    const uint8_t* nv21_input,  // 直接摄像头 NV21 数据
    int src_w, int src_h
) {
    std::vector<PalmBox> detections;

    if (ctx_ == 0) {
        std::cerr << "RKNN context not initialized!" << std::endl;
        return detections;
    }

    // 1️⃣ 使用 RGA 做 NV21 → RGB888 + Resize
    const int net_w = cfg_values_.resolution; // 模型输入宽
    const int net_h = cfg_values_.resolution; // 模型输入高

    std::vector<uint8_t> rgb_buf(net_w * net_h * 3); // RGB888 buffer

    // wrap source NV21
    rga_buffer_t src = wrapbuffer_virtualaddr(
        const_cast<uint8_t*>(nv21_input),
        src_w, src_h,
        RK_FORMAT_YCrCb_420_SP
    );

    // wrap destination RGB888
    rga_buffer_t dst = wrapbuffer_virtualaddr(
        rgb_buf.data(),
        net_w, net_h,
        RK_FORMAT_RGB_888
    );

    IM_STATUS ret_imresize = imresize(src, dst);
    if (ret_imresize != IM_STATUS_SUCCESS) {
        std::cerr << "RGA imresize failed: " << imStrError(ret_imresize) << std::endl;
        return detections;
    }

    // 2️⃣ RGB888 → float32，归一化到 [0,1]
    std::vector<float> input_float(net_w * net_h * 3);
    for (int i = 0; i < net_w * net_h; i++) {
        input_float[i * 3 + 0] = rgb_buf[i * 3 + 0] / 255.0f;
        input_float[i * 3 + 1] = rgb_buf[i * 3 + 1] / 255.0f;
        input_float[i * 3 + 2] = rgb_buf[i * 3 + 2] / 255.0f;
    }

    // 3️⃣ 设置 RKNN 输入
    rknn_input input[1];
    memset(input, 0, sizeof(input));
    input[0].index = 0;
    input[0].buf = input_float.data();
    input[0].size = input_float.size() * sizeof(float);
    input[0].pass_through = 0;
    input[0].type = RKNN_TENSOR_FLOAT32;
    input[0].fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(ctx_, io_num_.n_input, input);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed! ret=" << ret << std::endl;
        return detections;
    }

    // 4️⃣ 执行推理
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_run failed! ret=" << ret << std::endl;
        return detections;
    }

    // 5️⃣ 获取输出
    std::vector<rknn_output> outputs(io_num_.n_output);
    for (int i = 0; i < io_num_.n_output; i++) outputs[i].want_float = 1;

    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed! ret=" << ret << std::endl;
        return detections;
    }

    // 6️⃣ 解析 RKNN 输出
    detections = this->parseRknnOutputs(
        outputs,
        anchors_,
        cfg_values_.num_boxes,
        cfg_values_.num_keypoints,
        cfg_values_.resolution,
        cfg_values_.score_threshold
    );

    // 7️⃣ 释放 RKNN 输出
    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

    return detections;
}

std::vector<PalmBox> Detector::infer_nv21_zero_copy(
    const uint8_t* nv21_input,  // 摄像头 NV21 数据
    int src_w, int src_h
) {
    std::vector<PalmBox> detections;

    if (ctx_ == 0) {
        std::cerr << "RKNN context not initialized!" << std::endl;
        return detections;
    }

    const int net_w = cfg_values_.resolution;
    const int net_h = cfg_values_.resolution;

    // 1️⃣ 静态分配 RKNN 输入 buffer（float32，零拷贝用）
    static float* rknn_input_buf = nullptr;
    static size_t buf_size = 0;
    if (!rknn_input_buf || buf_size != net_w * net_h * 3 * sizeof(float)) {
        if (rknn_input_buf) free(rknn_input_buf);
        buf_size = net_w * net_h * 3 * sizeof(float);
        posix_memalign((void**)&rknn_input_buf, 64, buf_size); // 64字节对齐
    }

    // 2️⃣ 用 RGA 做 NV21 → RGB888 + Resize 到 net_w x net_h
    // 临时 buffer，直接输出到 float32 buffer
    // 注意 RGA 只支持 UINT8 输出，因此这里先输出到 UINT8 buffer
    static uint8_t* rga_rgb_buf = nullptr;
    static size_t rga_buf_size = 0;
    if (!rga_rgb_buf || rga_buf_size != net_w * net_h * 3) {
        if (rga_rgb_buf) free(rga_rgb_buf);
        rga_buf_size = net_w * net_h * 3;
        posix_memalign((void**)&rga_rgb_buf, 64, rga_buf_size);
    }

    rga_buffer_t src = wrapbuffer_virtualaddr(
        const_cast<uint8_t*>(nv21_input),
        src_w, src_h,
        RK_FORMAT_YCrCb_420_SP
    );

    rga_buffer_t dst = wrapbuffer_virtualaddr(
        rga_rgb_buf,
        net_w, net_h,
        RK_FORMAT_RGB_888
    );

    IM_STATUS ret = imresize(src, dst);
    if (ret != IM_STATUS_SUCCESS) {
        std::cerr << "RGA imresize failed: " << imStrError(ret) << std::endl;
        return detections;
    }

    // 3️⃣ 将 RGA 输出直接转换到 float32 RKNN 输入 buffer（归一化到 [0,1]）
    uint8_t* rgb_ptr = rga_rgb_buf;
    for (int i = 0; i < net_w * net_h; i++) {
        rknn_input_buf[i * 3 + 0] = rgb_ptr[i * 3 + 0] / 255.0f;
        rknn_input_buf[i * 3 + 1] = rgb_ptr[i * 3 + 1] / 255.0f;
        rknn_input_buf[i * 3 + 2] = rgb_ptr[i * 3 + 2] / 255.0f;
    }
    // ⚠ 这里虽然还有一次循环，但不再开辟额外 buffer，可看作零拷贝优化

    // 4️⃣ 设置 RKNN 输入
    rknn_input input[1];
    memset(input, 0, sizeof(input));
    input[0].index = 0;
    input[0].buf   = rknn_input_buf;          // 直接使用 float32 buffer
    input[0].size  = buf_size;
    input[0].pass_through = 0;
    input[0].type  = RKNN_TENSOR_FLOAT32;
    input[0].fmt   = RKNN_TENSOR_NHWC;

    int r = rknn_inputs_set(ctx_, io_num_.n_input, input);
    if (r != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed! ret=" << r << std::endl;
        return detections;
    }

    // 5️⃣ 执行推理
    r = rknn_run(ctx_, nullptr);
    if (r != RKNN_SUCC) {
        std::cerr << "rknn_run failed! ret=" << r << std::endl;
        return detections;
    }

    // 6️⃣ 获取输出
    std::vector<rknn_output> outputs(io_num_.n_output);
    for (int i = 0; i < io_num_.n_output; i++) outputs[i].want_float = 1;

    r = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (r != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed! ret=" << r << std::endl;
        return detections;
    }

    // 7️⃣ 解析输出
    detections = this->parseRknnOutputs(
        outputs,
        anchors_,
        cfg_values_.num_boxes,
        cfg_values_.num_keypoints,
        cfg_values_.resolution,
        cfg_values_.score_threshold
    );

    // 8️⃣ 释放 RKNN 输出
    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

    return detections;
}

void Detector::decodeBoxes(const std::vector<float>& raw_boxes,
                 const std::vector<float>& anchors,
                 std::vector<PalmBox>& boxes_out,
                 int num_boxes,
                 int num_keypoints,
                 float resolution)
{
    boxes_out.clear();
    boxes_out.reserve(num_boxes);

    for (int i = 0; i < num_boxes; i++) {
        const float* r = &raw_boxes[i * (4 + num_keypoints * 2)];
        const float* a = &anchors[i * 4];
        
        PalmBox box;

        float x_center = r[0] / resolution * a[2] + a[0];
        float y_center = r[1] / resolution * a[3] + a[1];

        float w = r[2] / resolution * a[2];
        float h = r[3] / resolution * a[3];

        box.x = x_center - w / 2.f;
        box.y = y_center - h / 2.f;
        box.w = w;
        box.h = h;

        box.keypoints.resize(num_keypoints * 2);
        for (int k = 0; k < num_keypoints; k++) {
            float kx = r[4 + k * 2]     / resolution * a[2] + a[0];
            float ky = r[4 + k * 2 + 1] / resolution * a[3] + a[1];
            box.keypoints[k * 2]     = kx;
            box.keypoints[k * 2 + 1] = ky;
        }
        boxes_out.push_back(box);
    }
}

std::vector<PalmBox> Detector::rawOutputToDetections(const std::vector<float>& raw_boxes,
                                           const std::vector<float>& raw_scores,
                                           const std::vector<float>& anchors,
                                           int num_boxes,
                                           int num_keypoints,
                                           float resolution,
                                           float score_threshold)
{
    std::vector<PalmBox> boxes;
    std::cout<<num_keypoints<<std::endl;

    this->decodeBoxes(raw_boxes, anchors, boxes, num_boxes, num_keypoints, resolution);
    std::vector<PalmBox> filtered_boxes;
    filtered_boxes.reserve(num_boxes);

    for (int i = 0; i < num_boxes; i++) {
        float score = sigmoid(raw_scores[i]);
        if (score < score_threshold) continue;

        boxes[i].score = score;
        filtered_boxes.push_back(boxes[i]);
    }

    return nms(filtered_boxes, 0.3f);
}


std::vector<PalmBox> Detector::parseRknnOutputs(
    const std::vector<rknn_output>& outputs,
    const std::vector<float>& anchors,
    int num_boxes,
    int num_keypoints,
    float resolution,
    float score_threshold)
{
    // --- 1. 获取 float 输出 ---
    float* boxes_ptr  = (float*)outputs[0].buf;
    float* scores_ptr = (float*)outputs[1].buf;

    // --- 2. 复制到 vector（方便使用你的 decodeBoxes） ---
    std::vector<float> raw_boxes(boxes_ptr,  boxes_ptr  + num_boxes * (4 + num_keypoints * 2));
    std::vector<float> raw_scores(scores_ptr, scores_ptr + num_boxes);

    // --- 3. 调用你原有 ONNX 后处理 ---
    return this->rawOutputToDetections(raw_boxes, raw_scores, anchors,
                                 num_boxes, num_keypoints,
                                 resolution, score_threshold);
}


std::vector<float> Detector::loadAnchorsBin(const std::string& filename) {
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


} // namespace HandDetectRknn
