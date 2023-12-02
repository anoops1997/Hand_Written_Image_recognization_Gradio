import gradio as gr
import tensorflow as tf
import numpy as np

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('mnist_model.h5')

# Function to make predictions
def predict_image(img):
    # Preprocess the input image
    img = img.reshape((28, 28, 1)).astype('float32') / 255.0

    # Make predictions
    predictions = model.predict(np.expand_dims(img, axis=0))

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])

    return f"Predicted Digit: {predicted_class}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs="image",
    outputs="text",
)

# Launch the Gradio interface
iface.launch()
