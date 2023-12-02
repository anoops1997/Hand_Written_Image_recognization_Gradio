import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained MNIST model
model = tf.keras.models.load_model('mnist_model.h5')

# Function to make predictions
def predict_image(drawing):
    try:
        # Get the drawing as a NumPy array
        img_array = drawing.image

        # Resize the input image to (28, 28)
        pil_image = Image.fromarray((img_array * 255).astype('uint8'))
        pil_image = pil_image.resize((28, 28))

        # Convert the image to grayscale
        pil_image = pil_image.convert("L")

        # Convert the image to a numpy array
        img_array = np.array(pil_image)

        # Normalize the pixel values to the range [0, 1]
        img_array = img_array.astype('float32') / 255.0

        # Ensure the array has the correct shape
        if img_array.shape == (28, 28):
            img_array = np.expand_dims(img_array, axis=-1)

            # Make predictions
            predictions = model.predict(np.expand_dims(img_array, axis=0))

            # Get the predicted class
            predicted_class = np.argmax(predictions[0])

            return f"Predicted Digit: {predicted_class}"
        else:
            return "Error: Invalid image size after resizing"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Create the Gradio interface with a drawing input
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Drawing(allow_clear=True),
    outputs="text",
)

# Launch the Gradio interface
iface.launch()
