#include "yolov8_detect.h"

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
    
    cfg_values.score_threshold = std::stof(config["score_threshold"]); // 输入图片尺寸
    cfg_values.max_frame_threshold = std::stoi(config["max_frame_threshold"]);

    return cfg_values;
}

void Detector::dump_tensor_attr(rknn_tensor_attr *attr)
{
    char dims[128] = {0};
    for (int i = 0; i < attr->n_dims; ++i) {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride = %d, "
           "fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, attr->w_stride, attr->size_with_stride,
           get_format_string(attr->fmt), get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp,
           attr->scale);
}

int NC1HWC2_i8_to_NCHW_i8(const int8_t *src, int8_t *dst, int *dims, int channel, int h, int w, int zp, float scale) {
    int batch  = dims[0];
    int C1     = dims[1];
    int C2     = dims[4];
    int hw_src = dims[2] * dims[3];
    int hw_dst = h * w;
    for (int i = 0; i < batch; i++) {
        const int8_t *src_b = src + i * C1 * hw_src * C2;
        int8_t        *dst_b = dst + i * channel * hw_dst;
        for (int c = 0; c < channel; ++c) {
            int           plane  = c / C2;
            const int8_t *src_bc = plane * hw_src * C2 + src_b;
            int           offset = c % C2;
            for (int cur_h = 0; cur_h < h; ++cur_h)
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int cur_hw                 = cur_h * w + cur_w;
                    dst_b[c * hw_dst + cur_hw] = src_bc[C2 * cur_hw + offset] ; // int8-->int8
                }
        }
    }

    return 0;
}

Detector::Detector(const ConfigInfo& config) {
    memset(&rknn_app_ctx_, 0, sizeof(rknn_app_context_t));
    init_post_process();
    init_yolov8_model(config.model_path.c_str());
}

// 析构函数释放 RKNN 资源
Detector::~Detector() {
    if (ctx_ != 0) {
        ctx_ = release_yolov8_model();
        if (ctx_ != 0)
        {
            printf("release_yolov8_model fail! ret=%d\n", ctx_);
        }
    }
    deinit_post_process();

    
}


int Detector::init_yolov8_model(const char *model_path)
{
    int ret;
    int model_len = 0;
    char *model;
    
    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx_, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_native_attrs[io_num.n_input];
    memset(input_native_attrs, 0, sizeof(input_native_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_native_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_native_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_native_attrs[i]));
    }

    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    input_native_attrs[0].type = RKNN_TENSOR_UINT8;
    rknn_app_ctx_.input_mems[0] = rknn_create_mem(ctx_, input_native_attrs[0].size_with_stride);

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx_, rknn_app_ctx_.input_mems[0], &input_native_attrs[0]);
    if (ret < 0) {
        printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_native_attrs[io_num.n_output];
    memset(output_native_attrs, 0, sizeof(output_native_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_native_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_native_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_native_attrs[i]));
    }

    // Set output tensor memory
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        rknn_app_ctx_.output_mems[i] = rknn_create_mem(ctx_, output_native_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx_, rknn_app_ctx_.output_mems[i], &output_native_attrs[i]);
        if (ret < 0) {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    // Set to context
    rknn_app_ctx_.rknn_ctx = ctx_;

    // TODO
    if (output_native_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_native_attrs[0].type == RKNN_TENSOR_INT8) {
        rknn_app_ctx_.is_quant = true;
    } else {
        rknn_app_ctx_.is_quant = false;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
    }

    rknn_app_ctx_.io_num = io_num;
    rknn_app_ctx_.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx_.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx_.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx_.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    rknn_app_ctx_.input_native_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx_.input_native_attrs, input_native_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx_.output_native_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx_.output_native_attrs, output_native_attrs, io_num.n_output * sizeof(rknn_tensor_attr));


    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        rknn_app_ctx_.model_channel = input_attrs[0].dims[1];
        rknn_app_ctx_.model_height = input_attrs[0].dims[2];
        rknn_app_ctx_.model_width = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        rknn_app_ctx_.model_height = input_attrs[0].dims[1];
        rknn_app_ctx_.model_width = input_attrs[0].dims[2];
        rknn_app_ctx_.model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           rknn_app_ctx_.model_height, rknn_app_ctx_.model_width, rknn_app_ctx_.model_channel);

    return 0;
}




int Detector::release_yolov8_model() {
    int ret;
    if (rknn_app_ctx_.input_attrs != NULL) {
        free(rknn_app_ctx_.input_attrs);
        rknn_app_ctx_.input_attrs = NULL;
    }
    if (rknn_app_ctx_.output_attrs != NULL) {
        free(rknn_app_ctx_.output_attrs);
        rknn_app_ctx_.output_attrs = NULL;
    }
    if (rknn_app_ctx_.input_native_attrs != NULL) {
        free(rknn_app_ctx_.input_native_attrs);
        rknn_app_ctx_.input_native_attrs = NULL;
    }
    if (rknn_app_ctx_.output_native_attrs != NULL) {
        free(rknn_app_ctx_.output_native_attrs);
        rknn_app_ctx_.output_native_attrs = NULL;
    }

    for (int i = 0; i < rknn_app_ctx_.io_num.n_input; i++) {
        if (rknn_app_ctx_.input_mems[i] != NULL) {
            ret = rknn_destroy_mem(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.input_mems[i]);
            if (ret != RKNN_SUCC) {
                printf("rknn_destroy_mem fail! ret=%d\n", ret);
                return -1;
            }
        }
    }
    for (int i = 0; i < rknn_app_ctx_.io_num.n_output; i++) {
        if (rknn_app_ctx_.output_mems[i] != NULL) {
            ret = rknn_destroy_mem(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.output_mems[i]);
            if (ret != RKNN_SUCC) {
                printf("rknn_destroy_mem fail! ret=%d\n", ret);
                return -1;
            }
        }
    }
    if (rknn_app_ctx_.rknn_ctx != 0) {
        ret = rknn_destroy(rknn_app_ctx_.rknn_ctx);
        if (ret != RKNN_SUCC) {
            printf("rknn_destroy fail! ret=%d\n", ret);
            return -1;
        }
        rknn_app_ctx_.rknn_ctx = 0;

    }
    return 0;
}

int Detector::inference_yolov8_model(image_buffer_t *img, object_detect_result_list *od_results) {
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    int bg_color = 114;

    if ((!&rknn_app_ctx_) || !(img) || (!od_results)) {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));

    // Pre Process
    dst_img.width = rknn_app_ctx_.model_width;
    dst_img.height = rknn_app_ctx_.model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.fd = rknn_app_ctx_.input_mems[0]->fd;
    dst_img.virt_addr = (unsigned char*)rknn_app_ctx_.input_mems[0]->virt_addr;

    if (dst_img.virt_addr == NULL && dst_img.fd == 0) {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0) {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(rknn_app_ctx_.rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    //NC1HWC2 to NCHW
    rknn_output outputs[rknn_app_ctx_.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < rknn_app_ctx_.io_num.n_output; i++) {
        int   channel = rknn_app_ctx_.output_attrs[i].dims[1];
        int   h       = rknn_app_ctx_.output_attrs[i].n_dims > 2 ? rknn_app_ctx_.output_attrs[i].dims[2] : 1;
        int   w       = rknn_app_ctx_.output_attrs[i].n_dims > 3 ? rknn_app_ctx_.output_attrs[i].dims[3] : 1;
        int   hw      = h * w;
        int   zp      = rknn_app_ctx_.output_native_attrs[i].zp;
        float scale   = rknn_app_ctx_.output_native_attrs[i].scale;
        if (rknn_app_ctx_.is_quant) {
            outputs[i].size = rknn_app_ctx_.output_native_attrs[i].n_elems * sizeof(int8_t);
            outputs[i].buf = (int8_t *)malloc(outputs[i].size);
            if (rknn_app_ctx_.output_native_attrs[i].fmt == RKNN_TENSOR_NC1HWC2) {
                NC1HWC2_i8_to_NCHW_i8((int8_t *)rknn_app_ctx_.output_mems[i]->virt_addr, (int8_t *)outputs[i].buf,
                                      (int *)rknn_app_ctx_.output_native_attrs[i].dims, channel, h, w, zp, scale);
            } else {
                memcpy(outputs[i].buf, rknn_app_ctx_.output_mems[i]->virt_addr, outputs[i].size);
            }
        } else {
            printf("Currently zero copy does not support fp16!\n");
            goto out;
        }
    }

    // Post Process
    post_process(&rknn_app_ctx_, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    for (int i = 0; i < rknn_app_ctx_.io_num.n_output; i++) {
        free(outputs[i].buf);
    }

out:
    return ret;
}