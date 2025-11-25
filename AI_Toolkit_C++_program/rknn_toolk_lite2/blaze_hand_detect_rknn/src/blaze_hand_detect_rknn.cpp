#include "blaze_hand_detect_rknn.hpp"


namespace HandDetectRknn {

// 析构函数释放 RKNN 资源
Detector::Detector() {
    
    auto cfg = readConfig(this->config_file_,this->cfg_values_);

    this->anchors_ = loadAnchorsBin(this->cfg_values_.anchors_path);

}

// 析构函数释放 RKNN 资源
Detector::~Detector() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

ConfigInfo Detector::getModelparameter(){
    return this->cfg_values_;
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

// 推理
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

std::unordered_map<std::string, std::string> Detector::readConfig(const std::string& filename,ConfigInfo &cfg_values) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return config;
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

    return config;
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
