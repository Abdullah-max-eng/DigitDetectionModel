import cv2
import mediapipe as mp
import pickle
import numpy as np

module_dict = pickle.load(open('./model.p', 'rb'))

model = module_dict['model']



cap = cv2.VideoCapture(0)



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0:'No Hand', 1:'1', 2:'2', 3:'3'}
while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2)
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)

            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)


        prediciton = model.predict([np.asarray(data_aux)])
        predictedDegit = labels_dict[int(prediciton[0])]

        cv2.rectangle(frame, (x1,y1), (x2,y2),(0,0,0), 4)
        cv2.putText(frame,predictedDegit, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)







    cv2.imshow('fram',frame)
    cv2.waitKey(1)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows.
cap.release()
cv2.destroyAllWindows()
