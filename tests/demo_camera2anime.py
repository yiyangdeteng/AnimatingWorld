import sys
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.camera.camera_manager import CameraManager

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/Shinkai_53.onnx'))

# 推理函数，输入为BGR格式ndarray，输出为BGR格式ndarray
def infer_frame_with_shinkai(frame, session, input_name):
    # 预处理：BGR->RGB, resize, float32, 归一化到[-1,1]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)  # (1,256,256,3)
    # 推理
    output = session.run(None, {input_name: img})[0]
    out = output[0]
    out = (out + 1.0) * 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out

if __name__ == "__main__":
    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在: {MODEL_PATH}")
        exit(1)
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    # 启动摄像头
    cam = CameraManager(fps=60)  # 提高摄像头采集帧率
    cam.initialize()
    # 尝试自动调整曝光（部分摄像头支持）
    try:
        # 关闭自动曝光（部分摄像头参数为0或1，具体取决于驱动）
        cam.set_parameter(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        # 设置曝光值（数值范围依摄像头而定，通常为负数或小正数）
        cam.set_parameter(cv2.CAP_PROP_EXPOSURE, -5)  # 曝光值调低，降低亮度
        cam.set_parameter(cv2.CAP_PROP_GAIN, 5)      # 增益调低，降低亮度
        print("已尝试设置曝光为-5，增益为5。如需微调请修改此值。")
    except Exception as e:
        print(f"设置曝光参数时出错: {e}")
    cam.start_capture()
    print("摄像头已启动，按q退出。")
    try:
        while True:
            frame = cam.latest_frame
            if frame is not None:
                anime_frame = infer_frame_with_shinkai(frame, session, input_name)
                # 放大显示窗口
                display_anime = cv2.resize(anime_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
                display_orig = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
                # 拼接原图和动画图
                concat = np.hstack((display_orig, display_anime))
                cv2.imshow('Original | Anime', concat)
            else:
                print("未获取到摄像头帧")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # 去除 sleep，最大化推理与显示速度
    finally:
        cam.stop_capture()
        cv2.destroyAllWindows()
        print("已退出")
