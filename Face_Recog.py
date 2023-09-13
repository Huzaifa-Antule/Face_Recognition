import tkinter as tk
from tkinter import filedialog
import cv2
from deepface import DeepFace
# create a password var , compare password wit variable 
def compare_faces(image1_path, image2_path):
    try:
        # Read the input images
        img_01 = cv2.imread(image1_path)
        img_02 = cv2.imread(image2_path)

        # Perform face verification using VGG-Face model
        result = DeepFace.verify(img_01, img_02, model_name='VGG-Face')

        # Obtain similarity score (distance) from the result dictionary
        similarity = result['distance']

        # Set threshold and check for face match
        threshold = 0.3  # Adjust this threshold value as per your requirement

        if similarity < threshold:
            return "The faces are matched."
        else:
            return "The faces are not matched."
    except Exception as e:
        return f"Error occurred: {str(e)}"

def browse_image(entry):
    filetypes = (
        ("JPEG Files", "*.jpeg"),
        ("PNG Files", "*.png"),
        ("JPG Files", "*.jpg"),
    )
    filename = filedialog.askopenfilename(filetypes=filetypes)
    entry.delete(0, tk.END)
    entry.insert(tk.END, filename)

def compare_images():
    image_path_1 = entry_1.get()
    image_path_2 = entry_2.get()

    output = compare_faces(image_path_1, image_path_2)
    output_label.config(text=f"Result :{output}")

    # Clear the output after 10 seconds
    window.after(10000, clear_output)
def clear_output():
    output_label.config(text="")
# Create the main Tkinter window
window = tk.Tk()
window.title("Face Comparison by Huzaifa Antule")

# Create the image path entry widgets
entry_1 = tk.Entry(window, width=50)
entry_1.grid(row=0, column=0, padx=10, pady=10)
browse_button_1 = tk.Button(window, text="Browse", command=lambda: browse_image(entry_1))
browse_button_1.grid(row=0, column=1, padx=5)

entry_2 = tk.Entry(window, width=50)
entry_2.grid(row=1, column=0, padx=10, pady=10)
browse_button_2 = tk.Button(window, text="Browse", command=lambda: browse_image(entry_2))
browse_button_2.grid(row=1, column=1, padx=5)

# Create the compare button
compare_button = tk.Button(window, text="Compare", command=compare_images)
compare_button.grid(row=2, column=0, padx=10, pady=10)

tk.Label(text="Alert : Process May Take few Minutes.").grid(row=3, column=0, padx=10, pady=10)

# Create the output label
output_label = tk.Label(window, text="")
output_label.grid(row=4, column=0, padx=10, pady=10)

# Start the Tkinter event loop
window.mainloop()
