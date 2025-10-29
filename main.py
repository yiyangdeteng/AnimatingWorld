import sys
import os
import cv2
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from core.camera.camera_manager import CameraManager
from core.animation.anime_stylizer import AnimeStylizer
from core.gesture.gesture_recognizer import GestureRecognizer
from core.filters.gesture_filter import GestureFilter

def main():
    # 初始化动画化模型
    stylizer = AnimeStylizer()
    # 初始化摄像头
    cam = CameraManager(fps=60)
    gesture = GestureRecognizer()
    gesture_filter = GestureFilter()
    cam.initialize()
    try:
        cam.set_parameter(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cam.set_parameter(cv2.CAP_PROP_EXPOSURE, -5)
        cam.set_parameter(cv2.CAP_PROP_GAIN, 5)
        print("已尝试设置曝光为-5，增益为5。如需微调请修改此值。")
    except Exception as e:
        print(f"设置曝光参数时出错: {e}")
    cam.start_capture()
    print("摄像头已启动，按q退出。")
    try:
        while True:
            frame = cam.latest_frame
            if frame is not None:
                # 先在原始帧上做手势识别
                gesture_type, center = gesture.detect_gesture(frame)
                anime_frame = stylizer.stylize(frame)
                # 坐标缩放到动画帧（256x256）
                if gesture_type and center is not None:
                    scale_x = 256 / frame.shape[1]
                    scale_y = 256 / frame.shape[0]
                    center_anim = (int(center[0] * scale_x), int(center[1] * scale_y))
                    anime_frame = gesture_filter.apply(anime_frame, center_anim, gesture_type)
                display_anime = cv2.resize(anime_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
                display_orig = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
                concat = np.hstack((display_orig, display_anime))
                cv2.imshow('Original | Anime', concat)
            else:
                print("未获取到摄像头帧")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop_capture()
        cv2.destroyAllWindows()
        print("已退出")

if __name__ == "__main__":
    main()
