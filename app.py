""" import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('model/plant_model.h5')
classes = ['Healthy', 'Powdery', 'Rust']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(150,150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

        prediction = model.predict(img_tensor)[0]
        class_index = np.argmax(prediction)
        disease = classes[class_index]
        confidence = prediction[class_index]

        # Remedies
        remedies = {
            "Healthy": "No disease detected. Keep maintaining good conditions.",
            "Powdery": "Use fungicides. Remove infected leaves. Ensure good airflow.",
            "Rust": "Apply sulfur-based fungicides. Remove affected foliage."
        }

        result = {
            'label': disease,
            'accuracy': f"{confidence*100:.2f}%",
            'remedy': remedies[disease],
            'image': filepath
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
"""
'''import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = load_model('model/plant_model.h5')
classes = ['Healthy', 'Powdery', 'Rust']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']

        # Create uploads folder if not exists
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)

        # Save file
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

        # Predict
        prediction = model.predict(img_tensor)[0]
        class_index = np.argmax(prediction)
        disease = classes[class_index]
        confidence = prediction[class_index]

        # Remedies
        remedies = {
            "Healthy": "No disease detected. Keep maintaining good conditions.",
            "Powdery": "Use fungicides. Remove infected leaves. Ensure good airflow.",
            "Rust": "Apply sulfur-based fungicides. Remove affected foliage."
        }

        # Only send the relative path inside static folder to template
        relative_path = os.path.join('uploads', file.filename)  # e.g. 'uploads/myimage.jpg'

        result = {
            'label': disease,
            'accuracy': f"{confidence*100:.2f}%",
            'remedy': remedies[disease],
            'image': relative_path  # pass this directly
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid

app = Flask(__name__)
model = load_model('model/plant_model.h5')
classes = ['Healthy', 'Powdery', 'Rust']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)

        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(upload_folder, unique_filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(150,150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

        prediction = model.predict(img_tensor)[0]
        class_index = np.argmax(prediction)
        disease = classes[class_index]
        confidence = prediction[class_index]

        remedies = {
            "Healthy": "No disease detected. Keep maintaining good conditions.",
            "Powdery": "Use fungicides. Remove infected leaves. Ensure good airflow.",
            "Rust": "Apply sulfur-based fungicides. Remove affected foliage."
        }

        result = {
            'label': disease,
            'accuracy': f"{confidence*100:.2f}%",
            'remedy': remedies[disease],
            'image': filepath
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

