import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):

    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Check if it's a directory
        for img_filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_filename)
            if os.path.isfile(img_path):  # Check if it's a file
                print(f"Processing image: {img_filename}")
                img = cv2.imread(img_path)
                if img is not None:  # Check if the file is a valid image
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    if results.multi_hand_landmarks:

                        for hand_landmarks in results.multi_hand_landmarks:
                            data_aux = []  # Initialize data auxiliary list for each hand
                            for landmark in hand_landmarks.landmark:


                                x = landmark.x
                                y = landmark.y
                                # Append x and y coordinates to data auxiliary list
                                data_aux.append(x)
                                data_aux.append(y)

                            data.append(data_aux)  # Append hand data to main data list
                            labels.append(dir_)  # Append label for hand (directory name) to labels list
                            time.sleep(1)

                            mp_drawing.draw_landmarks(
                                img_rgb,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                # Set your desired style here
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                # Set your desired style here
                            )

                    plt.figure()
                    plt.imshow(img_rgb)
                    plt.show()
                else:
                    print(f"Failed to load image: {img_filename}")
            else:
                print(f"Not a file: {img_path}")
    else:
        print(f"Not a directory: {dir_}")

    # Introduce a delay of 3 seconds between iterations


# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
