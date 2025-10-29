import cv2
import threading
import time
from collections import deque

class CameraManager:
    def __init__(self, camera_index=0, width=640, height=480, fps=30, buffer_size=2):
        self.camera_index = camera_index #摄像头索引
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size #缓冲区大小
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None



    def initialize(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头索引 {self.camera_index}")
        print("摄像头已初始化")


    def start_capture(self):
        """启动摄像头捕获线程"""
        if self.running:
            print("捕获线程已在运行")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()
        print("捕获线程已启动")

    
    def _capture_loop(self):
        """摄像头捕获循环"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧")
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame_buffer.append(frame)
            time.sleep(1 / self.fps)

    @property
    def latest_frame(self):
        """获取最新的摄像头帧（属性方式）"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
            else:
                return None
    
    def stop_capture(self):
        """停止摄像头捕获线程"""
        if not self.running:
            print("捕获线程未在运行")
            return
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        print("捕获线程已停止，摄像头已释放")

    def is_running(self):
        """检查捕获线程是否在运行"""
        return self.running
    
    def set_parameter(self, prop_id, value):
        """设置摄像头参数"""
        if self.cap:
            self.cap.set(prop_id, value)

    def get_parameter(self, prop_id):
        """获取摄像头参数"""
        if self.cap:
            return self.cap.get(prop_id)
        return None
    
    def __del__(self):
        self.stop_capture()
        if self.cap:
            self.cap.release()
        print("CameraManager 已销毁")

