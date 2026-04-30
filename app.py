import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model("best_finetuned.keras")
class_names = ['airplane', 'automobile', 'ship']

def make_gradcam_heatmap(img_array, model, backbone_name):
    # Get the backbone layer and its index in the sequence
    backbone_layer = model.get_layer(backbone_name)
    backbone_idx = model.layers.index(backbone_layer)
    
    with tf.GradientTape() as tape:
        x = img_array
        
        # 1. Eagerly execute layers up to the backbone
        for layer in model.layers[:backbone_idx + 1]:
            # Skip InputLayer as it cannot be "called" with a tensor in eager mode
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
            
        features = x
        # Tell the gradient tape to track this feature map
        tape.watch(features)
        
        # 2. Eagerly execute the rest of the classification head
        for layer in model.layers[backbone_idx + 1:]:
            x = layer(x)
            
        preds = x
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Compute gradients of the predicted class with respect to the feature map
    grads = tape.gradient(class_channel, features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    features = features[0]
    
    # 4. Multiply each channel by its gradient importance
    heatmap = features @ tf.expand_dims(pooled_grads, axis=-1)
    heatmap = tf.squeeze(heatmap)
    
    # 5. Normalize the heatmap safely
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10 # Prevent division by zero
    heatmap = heatmap / max_heat
    
    return heatmap.numpy()


def predict(image):
    # Preprocess to match training size (128x128)
    img_array = tf.image.resize(image, (128, 128))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    
    # Use our Eager-Execution Grad-CAM
    backbone_name = 'mobilenetv2_1.00_128'
    heatmap = make_gradcam_heatmap(img_array, model, backbone_name)
    
    # Superimpose heatmap on original image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert OpenCV's BGR heatmap to RGB <---
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Balance the weights and clip the values to prevent overflow <---
    superimposed_img = np.clip(heatmap * 0.4 + image * 0.6, 0, 255)
    
    return {class_names[i]: float(preds[0][i]) for i in range(3)}, np.uint8(superimposed_img)


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=[gr.Label(num_top_classes=3), gr.Image(label="Why the model chose this")],
    title="Airplane-Car-Ship Classifier with Explainability"
)

interface.launch()