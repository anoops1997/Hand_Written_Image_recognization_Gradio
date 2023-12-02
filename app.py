import gradio as gr
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('mnist_model.h5')

# Define the prediction function
def predict_digit(image):
    # Convert Gradio Image to NumPy array
    img_array = np.array(image)
    
    # Convert to grayscale and resize to 28x28 (MNIST model input size)
    img = Image.fromarray(img_array).convert('L').resize((28, 28))
    
    # Convert to NumPy array and normalize
    img = np.array(img).astype('float32') / 255.0
    
    # Reshape to match the model input shape
    img = img.reshape((1, 28, 28, 1))
    
    # Make the prediction
    prediction = model.predict(img)
    
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)

    return str(predicted_digit)

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", height=280, width=280, label="Draw a digit"),
    outputs="text"
)
iface.launch()
