import cv2
import mediapipe as mp
 
with mp.solutions.hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:
    
    cap = cv2.VideoCapture('/home/chenkejing/Videos/test_videos/蜜加原创手指舞《采薇》.mp4')  # 打开默认摄像头
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
            
        image.flags.writeable = False  # 禁用写入提高性能
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, 
                    mp.solutions.hands_connections.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style())
        
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # ESC键退出
            break
        
cap.release()