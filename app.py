import gradio as gr
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('mnist_model.h5')

# Function to make predictions
def predict_image(img):
    # Preprocess the input image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])

    return f"Predicted Digit: {predicted_class}"

# Define the input component
input_component = gr.Image(preprocessing_fn=preprocess_input, shape=(28, 28, 1))

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=input_component,
    outputs="text",
    live=True,
    theme="light",
    capture_session=True,
    interpretation="default"
)

# Launch the Gradio interface
iface.launch()
