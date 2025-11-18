import time

class HandStateController:

    def __init__(self, threshold=7,time_thresh=1):
        self.threshold = threshold  # 连续帧阈值
        self.identification_number = 4
        self.counter = 0            # 当前累计帧数
        self.event_state = False          # 当前状态输出

        self.event_time_limit = time_thresh
        self.event_infer_send_time = time.time()
        self.event_info_send_trigger = False  # 发送手部检测时间状态标志

    def update(self, detected: bool):
        """
        detected: bool, 当前帧是否检测到手
        返回: bool, 当前平滑后的状态
        """

        if detected:
            self.counter = min(self.counter + 1, self.threshold)  # 限制最大不超过 threshold
        else:
            self.counter = max(self.counter - 1, 0)  # 限制最小不低于 0

        if self.counter >= self.identification_number:
            self.event_state = True
        else:
            self.event_state = False

        if self.event_state:
            current_time = time.time()
            delta_t = current_time-self.event_infer_send_time
            # print(delta_t)
            if delta_t > self.event_time_limit:

                self.event_info_send_trigger = True
                self.event_infer_send_time = current_time
            else:
                self.event_info_send_trigger = False
        else:
            self.event_info_send_trigger = False

        return self.event_state , self.event_info_send_trigger


controller = HandStateController()

# 模拟帧检测结果流（True表示检测到手）
frames = [False, True, True, True, True, True, True, True, False, False,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True, False, False,False,False,False, False, False,False,False,False,True,True,True,True,True,True,True,True,True,True,True]

for i, detected in enumerate(frames):
    state, trigger = controller.update(detected)
    print(f"Frame {i+1:02d}: detected={detected}, state={state}, trigger={trigger}")
    time.sleep(0.2)



