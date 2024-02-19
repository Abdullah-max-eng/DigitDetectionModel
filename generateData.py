import cv2
import os
import time

# Create a directory to store captured images
output_dir = 'data/3'
numberOfDataPoint = 10


if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Variable to track whether to capture images or not
capture_images = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Camera', frame)

    # Check for key press
    key = cv2.waitKey(1)

    # If 't' is pressed, start capturing images
    if key == ord('t'):
        capture_images = True
        starting_time = time.time()
        frame_counter = 0
        print("Starting image capture...")

    # If 'q' is pressed, quit the program
    elif key == ord('q'):
        break

    # Capture images when 't' is pressed
    if capture_images:
        if frame_counter == numberOfDataPoint - 1 :
            print("Capture finished.")
            capture_images = False
        elif time.time() - starting_time > frame_counter:
            img_name = os.path.join(output_dir, f"img{frame_counter}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Image saved as {img_name}")
            frame_counter += 1
            time.sleep(0.5)  # Capture one frame per second

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
