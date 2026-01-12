import cv2

# 读取图像
img = cv2.imread("/home/chenkejing/Desktop/image_49.jpg")
if img is None:
    raise ValueError("图像读取失败，请检查路径")

# 指定目标尺寸
width, height = 1280, 720  # 可以修改为你需要的大小
resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

# 显示结果
cv2.imshow("原图", img)
cv2.imshow("缩放后", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite("./image_49_resized.jpg", resized_img)
