import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2) 
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
px_right, py_right = 0, 0
px_left, py_left = 0, 0

while True:
    success, frame = cap.read()
    
    if not success:
        print("Failed to capture image")
        break
    
    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_label = results.multi_handedness[idx].classification[0].label
            
            h, w, c = frame.shape
            
            x1 = int(hand_landmarks.landmark[8].x * w)
            y1 = int(hand_landmarks.landmark[8].y * h)
            y1_joint = int(hand_landmarks.landmark[6].y * h)
            
            x2 = int(hand_landmarks.landmark[12].x * w)
            y2 = int(hand_landmarks.landmark[12].y * h)
            y2_joint = int(hand_landmarks.landmark[10].y * h)

            sign_on = y1 < y1_joint 
            mid_on = y2 < y2_joint
            
            if hand_label == "Right":
                if sign_on and mid_on:
                    cv2.circle(frame, (x1, y1), 15, (0,255,0), cv2.FILLED)
                    px_right, py_right = x1, y1
                elif sign_on and not mid_on:
                    cv2.circle(frame, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                    if px_right == 0 and py_right == 0:
                        px_right, py_right = x1, y1
                    cv2.line(canvas, (px_right,py_right), (x1, y1), (0, 0, 255), 5)
                    px_right, py_right = x1, y1
                else:
                    px_right, py_right = 0, 0
                    
            elif hand_label == "Left":
                if sign_on and not mid_on:
                    cv2.circle(frame, (x1, y1), 40, (255, 255, 255), cv2.FILLED)
                    
                    if px_left == 0 and py_left == 0:
                        px_left, py_left = x1, y1
                    
                    cv2.line(canvas, (px_left, py_left), (x1, y1), (0, 0, 0), 50)
                    px_left, py_left = x1, y1
                else:
                    px_left, py_left = 0, 0

    frame = cv2.add(frame, canvas)
    cv2.imshow("Virtual Pen", frame)
    
    if cv2.waitKey(1)  & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()



