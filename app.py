import gradio as gr
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("best_finetuned.keras")

# Define class names (from Notebook 1)
class_names = ['airplane', 'automobile', 'ship']

def predict(image):
    # Preprocess image to match training (128x128)
    img = tf.image.resize(image, (128, 128))
    img = np.expand_dims(img, axis=0)
    
    # Model includes rescaling layer, so just predict
    prediction = model.predict(img)[0]
    
    # Return a dictionary of labels and confidence scores
    return {class_names[i]: float(prediction[i]) for i in range(3)}

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Airplane-Car-Ship Classifier"
)

interface.launch()