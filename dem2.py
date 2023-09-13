import cv2
import datetime
import pandas as pd
import numpy as np
from deepface import DeepFace

# Load the pre-trained face recognition model
model = DeepFace.build_model("Facenet")

# Load the database of student images and names
database = {
    "image_paths": [
        "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Image_07.jpg",
        "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Image_01.jpeg",
        # "path_to_image_3.jpg",
        # Add more image paths here
    ],
    "names": [
        "Name1",
        "Name2",
        # "Name3",
        # Add more names here
    ],
    "roll_nos": [
        "RollNo1",
        "RollNo2",
        # "RollNo3",
        # Add more roll numbers here
    ]
}

# Initialize the attendance dataframe
attendance_df = pd.DataFrame(columns=["Student_Name", "Student_RollNo", "Attendance"])

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Student Attendance System", frame)

    # Detect faces in the frame using MTCNN
    detected_faces = DeepFace.detectFace(frame, detector_backend="mtcnn")

    if len(detected_faces) > 0:
        for face in detected_faces:
            # Extract face region
            x, y, w, h = face["box"]
            face_img = frame[y:y + h, x:x + w]

            # Resize and preprocess the face image
            face_img = cv2.resize(face_img, (160, 160))
            face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
            face_img = (face_img / 255.0).astype(np.float32)  # Normalize pixel values

            # Perform face recognition on the detected face
            face_embedding = model.predict(face_img)[0]

            # Compare the face embedding with the embeddings in the database
            results = DeepFace.verify(face_embedding, database["image_paths"])

            # Get the index of the matched face (if any)
            match_index = results["verified_indexes"][0] if results["verified"] else -1

            if match_index != -1:
                # Get the name and roll number of the matched student
                name = database["names"][match_index]
                roll_no = database["roll_nos"][match_index]

                # Display the student details and mark attendance
                cv2.putText(frame, f"Name: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll No: {roll_no}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                attendance_df = attendance_df.append(
                    {"Student_Name": name, "Student_RollNo": roll_no, "Attendance": "Present"}, ignore_index=True)
            else:
                # Display "Face not matched" message
                cv2.putText(frame, "Face not matched", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the frame with the detected faces and student details
    cv2.imshow("Student Attendance System", frame)

    # Wait for 'q' key to be pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Generate the attendance CSV file with today's date
today_date = datetime.date.today().strftime("%Y-%m-%d")
attendance_csv_filename = f"{today_date}_Attendance.csv"
attendance_df.to_csv(attendance_csv_filename, index=False)

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
