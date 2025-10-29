import onnxruntime as ort
import numpy as np
import cv2
import os

def infer_image_with_shinkai(img_path, model_path=None, out_path='output.jpg'):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '../models/AnimeGANv3_Hayao_36.onnx')
        model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return False
    if not os.path.exists(img_path):
        print(f"图片文件不存在: {img_path}")
        return False
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        # 读取并预处理图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return False
        print(f"读取 {img_path} 作为输入，原始 shape: {img.shape}, dtype: {img.dtype}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        print(f"img after norm dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
        img = np.expand_dims(img, axis=0)  # (1, 256, 256, 3)
        print(f"img final shape: {img.shape}, dtype: {img.dtype}")
        # 推理
        output = session.run(None, {input_name: img})[0]
        out = output[0]
        out = (out + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out)
        print(f'推理完成，结果已保存为 {out_path}')
        return True
    except Exception as e:
        print(f"推理失败: {e}")
        return False

# 用法示例
if __name__ == "__main__":
    infer_image_with_shinkai('input3.jpg')