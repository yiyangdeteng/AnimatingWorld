import cv2
import numpy as np
import mediapipe as mp
import math

class GestureRecognizer:
    """
    使用MediaPipe Hands检测多种手势：heart（比心）、thumbsup（点赞）、ok（OK）
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _angle(self, a, b, c):
        # 计算角度，a/b/c为(x, y)
        ab = np.array([b[0]-a[0], b[1]-a[1]])
        cb = np.array([b[0]-c[0], b[1]-c[1]])
        dot = ab.dot(cb)
        norm = np.linalg.norm(ab) * np.linalg.norm(cb)
        if norm == 0:
            return 0
        angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
        return np.degrees(angle)

    def detect_gesture(self, frame):
        # 返回 (gesture_type, center)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None, None
        # 1. 检测双手比心
        if len(results.multi_hand_landmarks) >= 2:
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]
            def get_tip_xy(hand):
                return np.array([
                    [hand.landmark[4].x, hand.landmark[4].y],
                    [hand.landmark[8].x, hand.landmark[8].y]
                ])
            tips1 = get_tip_xy(hand1)
            tips2 = get_tip_xy(hand2)
            d1 = np.linalg.norm(tips1[0] - tips2[0])
            d2 = np.linalg.norm(tips1[1] - tips2[1])
            if d1 < 0.08 and d2 < 0.08:
                cx = int((tips1[0,0] + tips2[0,0]) / 2 * frame.shape[1])
                cy = int((tips1[0,1] + tips2[0,1]) / 2 * frame.shape[0])
                return 'heart', (cx, cy)
        # 2. 检测单手点赞（大拇指竖起，其余手指收拢，拇指与食指夹角大）
        for hand in results.multi_hand_landmarks:
            lm = hand.landmark
            thumb_tip = lm[4]
            thumb_ip = lm[3]
            thumb_mcp = lm[2]
            index_mcp = lm[5]
            index_tip = lm[8]
            middle_tip = lm[12]
            ring_tip = lm[16]
            pinky_tip = lm[20]
            wrist = lm[0]
            # 竖大拇指：拇指指尖高于掌心，且拇指与食指夹角>40度，其余手指收拢
            thumb_angle = self._angle(
                (thumb_mcp.x, thumb_mcp.y),
                (thumb_ip.x, thumb_ip.y),
                (thumb_tip.x, thumb_tip.y)
            )
            thumb_index_angle = self._angle(
                (thumb_tip.x, thumb_tip.y),
                (thumb_ip.x, thumb_ip.y),
                (index_mcp.x, index_mcp.y)
            )
            # 拇指竖起且与食指夹角大
            if (thumb_tip.y < wrist.y and
                thumb_index_angle > 40 and
                index_tip.y > wrist.y and
                middle_tip.y > wrist.y and
                ring_tip.y > wrist.y and
                pinky_tip.y > wrist.y):
                cx = int(thumb_tip.x * frame.shape[1])
                cy = int(thumb_tip.y * frame.shape[0])
                return 'thumbsup', (cx, cy)
            # OK手势：拇指指尖和食指指尖距离很近，且中指、无名指、小指伸展
            d_ok = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if d_ok < 0.05 and \
                middle_tip.y < wrist.y and ring_tip.y < wrist.y and pinky_tip.y < wrist.y:
                cx = int((thumb_tip.x + index_tip.x) / 2 * frame.shape[1])
                cy = int((thumb_tip.y + index_tip.y) / 2 * frame.shape[0])
                return 'ok', (cx, cy)
        return None, None
