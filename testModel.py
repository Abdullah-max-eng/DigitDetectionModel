import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:



            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            #     mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            # )




    cv2.imshow('fram',frame)
    cv2.waitKey(25)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows.
cap.release()
cv2.destroyAllWindows()
