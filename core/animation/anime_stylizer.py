import os
import cv2
import numpy as np
import onnxruntime as ort

class AnimeStylizer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/Shinkai_53.onnx'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def stylize(self, frame: np.ndarray) -> np.ndarray:
        """
        输入BGR格式ndarray,输出BGR格式ndarray动画风格化
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.expand_dims(img, axis=0)  # (1,256,256,3)
        output = self.session.run(None, {self.input_name: img})[0]
        out = output[0]
        out = (out + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out
