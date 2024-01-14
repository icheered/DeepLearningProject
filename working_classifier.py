import tkinter as tk
from tkinter import filedialog
from tkinter import Canvas
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Load InceptionV3 model pre-trained on ImageNet
model = InceptionV3(weights='imagenet')

def classify_image(file_path):
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode and display the top-3 predicted classes
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    result = "\n".join([f"{label}: {prob:.2%}" for (imagenet_id, label, prob) in decoded_predictions])
    return result

def open_image():
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", ".png;.jpg;*.jpeg")])
    if file_path:
        result_label.config(text="Classifying...")
        result = classify_image(file_path)
        result_label.config(text=result)
        display_image(file_path)

def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    canvas.config(width=img.width(), height=img.height())
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img

# Create the main window
root = tk.Tk()
root.title("InceptionV3 Image Classifier")

# Create UI components
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

canvas = Canvas(root)
canvas.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Start the GUI
root.mainloop()