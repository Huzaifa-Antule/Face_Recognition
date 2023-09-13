import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the local image for comparison
local_image_path = "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Image_07.jpg"
local_image = cv2.imread(local_image_path)

# Define a sample database mapping local image to details
database = {
    "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Imae_10.jpg": {
        "name": "Huzaifa Antule",
        "number": "8999192715"
    },
    "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Imae_09.jpg": {
        "name": "Buran mistry",
        "number": "1234567890"
    }
}
while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Initialize a list to store matching results
    matching_results = []

    # Perform face recognition on the webcam frame for each image in the database
    for image_path, details in database.items():
        result = DeepFace.verify(frame, image_path, model_name='Facenet', enforce_detection=False)

        if result["verified"]:
            # Store the matching result along with the corresponding details
            matching_results.append((result, details))

    if matching_results:
        # Sort the matching results based on confidence score in descending order
        matching_results.sort(key=lambda x: x[0]["distance"], reverse=True)

        # Get the best matching result
        best_result, best_details = matching_results[0]
        name = best_details.get("name", "Unknown")
        number = best_details.get("number", "Unknown")

        # Draw the details on the frame
        cv2.putText(frame, f"Name: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Number: {number}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        
        cv2.putText(frame, "Face not matched", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Live Face Detector", frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()