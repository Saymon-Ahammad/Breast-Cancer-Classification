from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = tf.keras.models.load_model("hybrid_breast_cancer.h5", compile=False)

CLASS_NAMES = ["Cancer", "Non-Cancer"]
IMG_SIZE = 224  

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    probabilities = [0,0]
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            image_path = '/' + save_path.replace("\\","/") 

            img = preprocess_image(save_path)
            preds = model.predict(img)
            class_id = int(np.argmax(preds))
            prediction = CLASS_NAMES[class_id]
            confidence = round(float(preds[0][class_id]) * 100, 2)
            probabilities = [float(p)*100 for p in preds[0]]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        image_path=image_path,
        class_names=CLASS_NAMES
    )

if __name__ == "__main__":
    app.run(debug=True)
