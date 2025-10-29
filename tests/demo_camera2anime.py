import sys
import os
import time
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.camera.camera_manager import CameraManager
from core.animation.anime_stylizer import AnimeStylizer

if __name__ == "__main__":
    stylizer = AnimeStylizer()  # 使用动画化模块
    cam = CameraManager(fps=60)
    cam.initialize()
    try:
        cam.set_parameter(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cam.set_parameter(cv2.CAP_PROP_EXPOSURE, -5)
        cam.set_parameter(cv2.CAP_PROP_GAIN, 5)
        print("已尝试设置曝光为-5,增益为5。如需微调请修改此值。")
    except Exception as e:
        print(f"设置曝光参数时出错: {e}")
    cam.start_capture()
    print("摄像头已启动,按q退出。")
    try:
        while True:
            frame = cam.latest_frame
            if frame is not None:
                anime_frame = stylizer.stylize(frame)  # 动画化
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
