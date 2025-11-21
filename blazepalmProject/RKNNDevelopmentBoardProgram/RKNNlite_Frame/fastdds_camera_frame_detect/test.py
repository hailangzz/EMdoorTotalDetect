import numpy as np

anchor_path = r"/home/chenkejing/PycharmProjects/EMdoorTotalDetect/blazepalmProject/RKNNDevelopmentBoardProgram/RKNNlite_Frame/fastdds_camera_frame_detect/anchors_192.npy"
anchors = np.load(anchor_path).astype("float32")
print(anchors.shape)