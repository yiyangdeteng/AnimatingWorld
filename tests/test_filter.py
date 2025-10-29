import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from core.filters.gesture_filter import GestureFilter
from core.gesture.gesture_recognizer import GestureRecognizer

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    gesture_filter = GestureFilter()
    gesture = GestureRecognizer()
    print("按q退出，做比心、点赞、OK等手势试试！")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 检测多种手势
        gesture_type, center = gesture.detect_gesture(frame)
        if gesture_type and center is not None:
            frame = gesture_filter.apply(frame.copy(), center, gesture_type)
            cv2.putText(frame, f"{gesture_type} gesture detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Gesture & Filter Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
