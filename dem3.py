import cv2
import datetime
import pandas as pd
from deepface import DeepFace
cap = cv2.VideoCapture(0)
database = {
    "Huzaifa Antule": [
       
        {
            "image_path": "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\my1.jpg",
            "number": "8999192715"
        },
        {
            "image_path": "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\my2.jpg",
            "number": "8999192715"
        },
        {
            "image_path": "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\my3.jpg",
            "number": "8999192715"
        },
    ],
    "Buran Mistry": [
        {
            "image_path": "C:\\Users\\Huzaifa\\Desktop\\Face_Recognition Project\\Imae_09.jpg",
            "number": "1234567890"
        }
    ]
}
attendance_data = {
    "Student_Name": [],
    "Student_RollNo": [],
    "Attendance": []
}
threshold = 0.2
recognized_faces = []
# Adjust the rectangle position and size based on your requirements
rectangle_position = (180 , 90)
rectangle_size = (300, 300)
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    new_recognized_faces = []

    # Draw the stable rectangle
    cv2.rectangle(frame, rectangle_position, (rectangle_position[0] + rectangle_size[0], rectangle_position[1] + rectangle_size[1]), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        matching_results = []

        for person_name, images in database.items():
            for image_info in images:
                image_path = image_info["image_path"]
                if image_path not in recognized_faces:
                    result = DeepFace.verify(face_img, image_path, model_name='Facenet', enforce_detection=False)
                    if result["distance"] <= threshold:
                        matching_results.append((result, person_name, image_info))

        if matching_results:
            matching_results.sort(key=lambda x: x[0]["distance"])
            best_result, best_person_name, best_image_info = matching_results[0]
            name = best_person_name
            roll_no = best_image_info["number"]

            if roll_no not in attendance_data["Student_RollNo"]:
                attendance_data["Student_Name"].append(name)
                attendance_data["Student_RollNo"].append(roll_no)
                attendance_data["Attendance"].append("Present")
                new_recognized_faces.append(best_image_info["image_path"])
                cv2.putText(frame, f"Name: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Already Present", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    recognized_faces = new_recognized_faces

    cv2.imshow("Live Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
# Generate the attendance CSV file
date_str = datetime.datetime.now().strftime("%Y-%m-%d")
filename = f"Attendance_{date_str}.csv"
df = pd.DataFrame(attendance_data)
df.to_csv(filename, index=False)
