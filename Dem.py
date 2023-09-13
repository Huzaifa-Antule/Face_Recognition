import cv2
import datetime
import csv
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the local images for comparison and store their paths
local_image_paths = [
    "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Imae_10.jpg",
    "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Imae_09.jpg"
]

# Define a sample database mapping local image to details and attendance
database = {
    local_image_paths[0]: {
        "name": "Huzaifa Antule",
        "roll_no": "001",
        "attendance": "Absent"
    },
    local_image_paths[1]: {
        "name": "Buran Mistry",
        "roll_no": "002",
        "attendance": "Absent"
    }
}

# Create the attendance CSV file
today_date = datetime.date.today().strftime("%Y-%m-%d")
csv_file_path = f"C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\{today_date}_Attendance.csv"

with open(csv_file_path, "w", newline="") as csv_file:
    fieldnames = ["Name", "Roll No", "Attendance"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    while True:
        # Capture frame from the webcam
        ret, frame = cap.read()

        # Initialize a list to store matching results
        matching_results = []

        # Perform face recognition on the webcam frame for each image in the database
        for image_path, details in database.items():
            result = DeepFace.verify(frame, image_path, model_name='Facenet', enforce_detection=False)

            if result["verified"]:
                # Update the attendance status and store the matching result along with the corresponding details
                details["attendance"] = "Present"
                matching_results.append((result, details))

        if matching_results:
            # Sort the matching results based on confidence score in descending order
            matching_results.sort(key=lambda x: x[0]["distance"], reverse=True)

            # Get the best matching result
            best_result, best_details = matching_results[0]
            name = best_details.get("name", "Unknown")
            roll_no = best_details.get("roll_no", "Unknown")

            # Draw the details on the frame
            cv2.putText(frame, f"Name: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll No: {roll_no}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face not matched", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Live Face Detector", frame)

        # Write attendance data to the CSV file
        for details in database.values():
            writer.writerow({"Name": details["name"], "Roll No": details["roll_no"], "Attendance": details["attendance"]})

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
