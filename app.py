import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model("best_finetuned.keras")
class_names = ['airplane', 'automobile', 'ship']

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by "how important" they are for the prediction
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def predict(image):
    # Preprocess to match training size (128x128)
    img_array = tf.image.resize(image, (128, 128))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    
    # FIX: Treat the entire MobileNetV2 backbone as the "last conv layer"
    # because its output is the 4x4 feature map. 
    last_conv_layer_name = 'mobilenetv2_1.00_128'
    
    # FIX: Pass the MAIN model to Grad-CAM, not the backbone
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    
    # Superimpose heatmap on original image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Combine heatmap with original image (40% heatmap, 60% original)
    superimposed_img = heatmap * 0.4 + image
    
    return {class_names[i]: float(preds[0][i]) for i in range(3)}, np.uint8(superimposed_img)

# Interface setup
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=[gr.Label(num_top_classes=3), gr.Image(label="Why the model chose this")],
    title="Airplane-Car-Ship Classifier with Explainability"
)

interface.launch()