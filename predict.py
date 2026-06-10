import joblib
import numpy as np
from PIL import Image

MODEL_PATH = "model/image_classifier.pkl"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
label_mapping = saved_data["label_mapping"]
image_size = saved_data["image_size"]


def predict_image(image_path):

    img = Image.open(image_path)

    img = img.convert("RGB")

    img = img.resize(image_size)

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = img_array.flatten()

    img_array = img_array.reshape(1, -1)

    prediction = model.predict(
        img_array
    )[0]

    return label_mapping[prediction]