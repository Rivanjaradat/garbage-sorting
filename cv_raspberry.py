from teachable_machine_lite import TeachableMachineLite  # Import Teachable Machine Lite library
import cv2 as cv  # Import OpenCV library
import YB_Pcb_Car  # Import Yahboom car library
import time

cap = cv.VideoCapture(0)  # Initialize webcam capture

car = YB_Pcb_Car.YB_Pcb_Car()  # Initialize Yahboom car object

model_path = 'model.tflite'  # Path to the trained model
image_file_name = "frame.jpg"  # File name to save captured frame
labels_path = "labels.txt"  # Path to the label file
i = 0  # Counter for capturing frames

tm_model = TeachableMachineLite(model_path=model_path, labels_file_path=labels_path)  # Load the Teachable Machine model

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    time.sleep(0.1)  # Delay for smooth execution

    cv.imshow('Cam', frame)  # Display the frame in a window

    if i < 4:
        i += 1
    else:
        cv.imwrite(image_file_name, frame)  # Save the frame as an image file

        results = tm_model.classify_frame(image_file_name)  # Classify the captured frame using the model
        print(results["label"])  # Print the predicted label

        if results["label"] == "Plastic":
            car.Car_Run(50, 50)  # Control the car to run forward
            time.sleep(2)  # Delay for 2 seconds
            car.Car_Stop()  # Stop the car
        elif results["label"] == "Metal":
            car.Car_Back(40, 40)  # Control the car to move backward
            time.sleep(2)  # Delay for 2 seconds
            car.Car_Stop()  # Stop the car
        elif results["label"] == "Cartoon":
            car.Car_Right(50, 0)  # Control the car to turn right
            time.sleep(2)  # Delay for 2 seconds
            car.Car_Stop()  # Stop the car
        else:
            print("Error: Unknown label")  # Print an error message for unknown labels
            time.sleep(0.2)  # Delay for 0.2 seconds
            car.Car_Stop()  # Stop the car

        i = 0  # Reset the counter

    k = cv.waitKey(1)
    if k % 255 == 27:
        # Press ESC to close the camera view
        break
