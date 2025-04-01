import tensorflow.lite as tflite
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="currency_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess an image
img_path = "C:\\Users\\asus\\OneDrive\\Desktop\\currency\\Indian currency dataset v1\\test\\Background__311.jpg"  # Change this to your test image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

# Run inference
interpreter.invoke()

# Get prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)
print(f"Predicted Class: {predicted_class}")
