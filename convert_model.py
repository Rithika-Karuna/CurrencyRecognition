import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("currency_model.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("currency_model.tflite", "wb") as f:
    f.write(tflite_model)
