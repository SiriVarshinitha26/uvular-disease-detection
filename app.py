from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# ---------------- IMAGE FORMAT SUPPORT ----------------

from PIL import Image
import pillow_avif
from pillow_heif import register_heif_opener

register_heif_opener()

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model(
    "uvular_model.keras",
    compile=False
)

# ---------------- UPLOAD FOLDER ----------------

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- CLASSES ----------------

classes = [
    'No Disease',
    'Disease Detected'
]

# ---------------- GRAD-CAM FUNCTION ----------------

def generate_gradcam(img_path, model):

    img = image.load_img(
        img_path,
        target_size=(160,160)
    )

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    base_model = model.layers[0]

    for layer in reversed(base_model.layers):

        if 'conv' in layer.name:

            last_conv_layer = layer

            break

    # Rebuild Model

    x = base_model.output

    x = model.layers[1](x)

    x = model.layers[2](x)

    x = model.layers[3](x)

    output = model.layers[4](x)

    grad_model = tf.keras.models.Model(

        inputs=base_model.input,

        outputs=[
            last_conv_layer.output,
            output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(
        heatmap,
        0
    )

    if np.max(heatmap) != 0:

        heatmap /= np.max(heatmap)

    return heatmap

# ---------------- HOME PAGE ----------------

@app.route('/')

def home():

    return render_template('index.html')

# ---------------- PREDICTION ----------------

@app.route('/predict', methods=['POST'])

def predict():

    file = request.files['file']

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # ---------------- CONVERT IMAGE TO JPG ----------------

    try:

        img_pil = Image.open(filepath)

        # Correct image orientation
        try:
            from PIL import ImageOps
            img_pil = ImageOps.exif_transpose(img_pil)
        except:
            pass

        # Convert all image types to RGB
        img_pil = img_pil.convert("RGB")

        # Create converted JPG filename
        jpg_filepath = os.path.splitext(filepath)[0] + "_converted.jpg"

        # Save as JPG
        img_pil.save(
            jpg_filepath,
            "JPEG",
            quality=95
        )

        # Use converted JPG from now on
        filepath = jpg_filepath

    except Exception as e:

        print("IMAGE CONVERSION ERROR:", e)

        return (
            "Unable to read this image. "
            "Please upload a valid image file."
        )

    # ---------------- IMAGE PREPROCESSING ----------------

    img = image.load_img(
        filepath,
        target_size=(160,160)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    # ---------------- PREDICTION ----------------

    prediction = model.predict(img_array)

    predicted_index = 1 if prediction[0][0] > 0.5 else 0

    result = classes[predicted_index]

    # ---------------- CONFIDENCE SCORE ----------------

    if predicted_index == 1:

        confidence = round(
            float(prediction[0][0]) * 100,
            2
        )

    else:

        confidence = round(
            (1 - float(prediction[0][0])) * 100,
            2
        )

    # ---------------- GENERATE GRAD-CAM ----------------

    heatmap = generate_gradcam(
        filepath,
        model
    )

    img_cv = cv2.imread(filepath)

    img_cv = cv2.resize(
        img_cv,
        (160,160)
    )

    heatmap = cv2.resize(
        heatmap,
        (160,160)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        img_cv,
        0.6,
        heatmap,
        0.4,
        0
    )

    gradcam_filename = "gradcam_" + os.path.splitext(file.filename)[0]+".jpg"

    gradcam_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        gradcam_filename
    )

    cv2.imwrite(
        gradcam_path,
        superimposed_img
    )

    # ---------------- RESULT ----------------

    return render_template(

        'result.html',

        prediction=result,

        confidence=confidence,

        image_path=filepath,

        gradcam_path=gradcam_path

    )

# ---------------- MAIN ----------------

if __name__ == '__main__':

    app.run(debug=True)