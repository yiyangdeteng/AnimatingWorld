import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.camera.camera_manager import CameraManager
import cv2


if __name__ == "__main__":
    cam = CameraManager()
    cam.initialize()
    cam.start_capture()
    print("CameraManager 初始化并开始采集。")
    frame_count = 0
    for i in range(100):
        frame = cam.latest_frame
        if frame is not None:
            print(f"第{i+1}帧: shape={frame.shape}, dtype={frame.dtype}")
            cv2.imshow('Camera', frame)
            frame_count += 1
        else:
            print(f"第{i+1}帧: 未采集到有效帧")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.03)
    cam.stop_capture()
    cv2.destroyAllWindows()
    print(f"采集结束，共采集到有效帧数: {frame_count}")
