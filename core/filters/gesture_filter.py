import cv2
import numpy as np
import os
import random

class GestureFilter:
    """
    多手势滤镜：支持双手比心、单手比心、点赞、OK等
    """
    def __init__(self, heart_img_path=None):
        self.heart_img = None
        if heart_img_path and os.path.exists(heart_img_path):
            self.heart_img = cv2.imread(heart_img_path, cv2.IMREAD_UNCHANGED)

    def draw_heart(self, frame, center, size=80):
        overlay = frame.copy()
        x, y = int(center[0]), int(center[1])
        color = (0, 0, 255)  # 红色
        thickness = -1
        cv2.circle(overlay, (x-int(size*0.25), y-int(size*0.25)), int(size*0.3), color, thickness)
        cv2.circle(overlay, (x+int(size*0.25), y-int(size*0.25)), int(size*0.3), color, thickness)
        pts = np.array([[x-int(size*0.5), y-int(size*0.1)], [x+int(size*0.5), y-int(size*0.1)], [x, y+int(size*0.5)]], np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame

    def draw_thumbsup(self, frame, center, size=80):
        overlay = frame.copy()
        x, y = int(center[0]), int(center[1])
        color = (0, 200, 255)
        # 画一个简单的竖起大拇指
        cv2.rectangle(overlay, (x-int(size*0.15), y), (x+int(size*0.15), y+int(size*0.4)), color, -1)
        cv2.rectangle(overlay, (x-int(size*0.3), y+int(size*0.3)), (x+int(size*0.3), y+int(size*0.5)), color, -1)
        cv2.circle(overlay, (x, y), int(size*0.18), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame

    def draw_stars(self, frame, center, size=80, num_stars=5):
        overlay = frame.copy()
        x, y = int(center[0]), int(center[1])
        color = (0, 255, 255)  # 黄色
        for i in range(num_stars):
            # 随机位置和大小
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(size*0.3, size*0.7)
            sx = int(x + np.cos(angle) * dist)
            sy = int(y + np.sin(angle) * dist)
            r = random.randint(int(size*0.10), int(size*0.22))
            self._draw_star(overlay, (sx, sy), r, color)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame

    def _draw_star(self, img, center, radius, color, thickness=-1):
        # 五角星
        pts = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            x = center[0] + int(np.cos(angle) * radius)
            y = center[1] + int(np.sin(angle) * radius)
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        for i in range(5):
            cv2.line(img, pts[i], pts[(i+2)%5], color, thickness if thickness>0 else 2)

    def draw_ok(self, frame, center, size=80):
        # 画五颗位置和大小随机的星星
        return self.draw_stars(frame, center, size, num_stars=5)

    def apply(self, frame, center, gesture_type='heart', size=80):
        if gesture_type == 'heart':
            return self.draw_heart(frame, center, size)
        elif gesture_type == 'thumbsup':
            return self.draw_thumbsup(frame, center, size)
        elif gesture_type == 'ok':
            return self.draw_ok(frame, center, size)
        else:
            return frame
