#include "carpet_detect_interface.h"
#include "detect_context.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "image_drawing.h"

// carpet_detect_interface.cpp
static DetectContext g_ctx;


bool carpet_model_init(const char* config_path)
{
    if (g_ctx.initialized) {
        printf("carpet model already initialized");
        return true;
    }
    std::string config_file_path(config_path);
    g_ctx.config = readConfig(config_file_path);
    init_post_process();

    g_ctx.detector = new Detector(g_ctx.config);
    g_ctx.camera_params = new CameraParameters(g_ctx.config);

    g_ctx.initialized = true;

    printf("carpet_model_init success");
    return true;
}

// ================= 保存结果图像 =================
static int save_index = 0;   // 静态自增计数器

const char* save_dir = "./result_images";


bool carpet_detect_infer(const cv::Mat& img, ObjectCameraDetectResult* results, int max_results)
{
    if (!g_ctx.initialized || img.empty()) {
        printf("model not initialized or empty image");
        return false;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    // cv::Mat → image_buffer_t
    src_image.virt_addr = (unsigned char*)malloc(img.total() * img.elemSize());
    memcpy(src_image.virt_addr, img.data, img.total() * img.elemSize());

    src_image.width  = img.cols;
    src_image.height = img.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size   = img.total() * img.elemSize();

    object_detect_result_list od_results;

    int ret = g_ctx.detector->inference_yolov8_model(&src_image, &od_results);
    if (ret != 0) {
        printf("yolov8 inference failed");
        free(src_image.virt_addr);
        return false;
    }

    // // 可选：画框 / 坐标转换
    // for (int i = 0; i < od_results.count; i++) {
    //     auto& det = od_results.results[i];
    //     g_ctx.camera_params->ObjectboxToCameraXYZ(
    //         det.box, det.camera_coordinates);

    //     printf("det.camera_coordinates.left_bottom:%s",det.camera_coordinates.left_bottom);
    // }

      // 可选：画框 / 坐标转换
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result* det = &od_results.results[i];
        g_ctx.camera_params->ObjectboxToCameraXYZ(det->box, det->camera_coordinates);

        results[i].prop = det->prop;
        results[i].cls_id = det->cls_id;
        results[i].coords[0].X = det->camera_coordinates.left_top.X;
        results[i].coords[0].Y = det->camera_coordinates.left_top.Y;
        results[i].coords[0].Z = det->camera_coordinates.left_top.Z;

        results[i].coords[1].X = det->camera_coordinates.right_top.X;
        results[i].coords[1].Y = det->camera_coordinates.right_top.Y;
        results[i].coords[1].Z = det->camera_coordinates.right_top.Z;

        results[i].coords[2].X = det->camera_coordinates.right_bottom.X;
        results[i].coords[2].Y = det->camera_coordinates.right_bottom.Y;
        results[i].coords[2].Z = det->camera_coordinates.right_bottom.Z;

        results[i].coords[3].X = det->camera_coordinates.left_bottom.X;
        results[i].coords[3].Y = det->camera_coordinates.left_bottom.Y;
        results[i].coords[3].Z = det->camera_coordinates.left_bottom.Z;

    }


    // 画框和概率
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
        
        printf("finish draw_rectangle!!!!! \n");

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    // write_image("out.png", &src_image);

    // 如果目录不存在就创建
    if (access(save_dir, F_OK) != 0) {
        mkdir(save_dir, 0755);
    }

    char save_path[256];
    snprintf(save_path, sizeof(save_path),
            "%s/out_%06d.png", save_dir, save_index++);
    write_image(save_path, &src_image);
    printf("save result image: %s\n", save_path);


    free(src_image.virt_addr);
    return true;
}


// bool carpet_detect_infer(const cv::Mat& img)
// {
//     if (!g_ctx.initialized || img.empty()) {
//         printf("model not initialized or empty image\n");
//         return false;
//     }

//     // 确保图像连续存储
//     cv::Mat input_img = img.isContinuous() ? img : img.clone();

//     // OpenCV 默认 BGR，YOLOv8 model 需要 RGB
//     cv::Mat img_rgb;
//     cv::cvtColor(input_img, img_rgb, cv::COLOR_BGR2RGB);

//     // 准备 image_buffer_t
//     image_buffer_t src_image;
//     memset(&src_image, 0, sizeof(image_buffer_t));
//     size_t img_size = img_rgb.total() * img_rgb.elemSize();
//     src_image.virt_addr = (unsigned char*)malloc(img_size);
//     if (!src_image.virt_addr) {
//         printf("malloc failed\n");
//         return false;
//     }
//     memcpy(src_image.virt_addr, img_rgb.data, img_size);

//     src_image.width  = img_rgb.cols;
//     src_image.height = img_rgb.rows;
//     src_image.format = IMAGE_FORMAT_RGB888;
//     src_image.size   = img_size;

//     // YOLOv8 推理
//     object_detect_result_list od_results;
//     int ret = g_ctx.detector->inference_yolov8_model(&src_image, &od_results);
//     if (ret != 0) {
//         printf("yolov8 inference failed\n");
//         free(src_image.virt_addr);
//         return false;
//     }

//     // 可选：坐标转换
//     for (int i = 0; i < od_results.count; i++) {
//         auto& det = od_results.results[i];
//         g_ctx.camera_params->ObjectboxToCameraXYZ(det.box, det.camera_coordinates);
//     }

//     // 绘制检测框和文字
//     char text[256];
//     for (int i = 0; i < od_results.count; i++) {
//         object_detect_result *det_result = &(od_results.results[i]);
//         printf("%s @ (%d %d %d %d) %.3f\n",
//                coco_cls_to_name(det_result->cls_id),
//                det_result->box.left, det_result->box.top,
//                det_result->box.right, det_result->box.bottom,
//                det_result->prop);

//         int x1 = det_result->box.left;
//         int y1 = det_result->box.top;
//         int x2 = det_result->box.right;
//         int y2 = det_result->box.bottom;

//         draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

//         sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
//         draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
//     }

//     // 保存结果图像
//     if (access(save_dir, F_OK) != 0) {
//         mkdir(save_dir, 0755);
//     }

//     char save_path[256];
//     snprintf(save_path, sizeof(save_path), "%s/out_%06d.png", save_dir, save_index++);
//     write_image(save_path, &src_image);
//     printf("save result image: %s\n", save_path);

//     free(src_image.virt_addr);
//     return true;
// }


void carpet_model_release()
{
    if (!g_ctx.initialized)
        return;

    deinit_post_process();

    delete g_ctx.detector;
    delete g_ctx.camera_params;

    g_ctx.detector = nullptr;
    g_ctx.camera_params = nullptr;
    g_ctx.initialized = false;

    printf("carpet_model_release finished");
}
