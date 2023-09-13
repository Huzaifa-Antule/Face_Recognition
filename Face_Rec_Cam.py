import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the local image for comparison
local_image_path = "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Image_07.jpg"
local_image = cv2.imread(local_image_path)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Perform face recognition on the webcam frame
    result = DeepFace.verify(frame, local_image_path, model_name='Facenet', enforce_detection=False)

    if result["verified"]:
        print("Face matched!")
    else:
        print("Face not matched!")

    # Check if face detection was successful
    if "region" in result:
        # Extract the bounding box coordinates of the detected face
        x1, y1, x2, y2 = result["region"]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
