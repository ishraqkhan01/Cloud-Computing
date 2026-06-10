from flask import Flask
from flask import render_template
from flask import request

from predict import predict_image

import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return "No file uploaded"

    image = request.files["image"]

    if image.filename == "":

        return "No file selected"

    image_path = os.path.join(
        UPLOAD_FOLDER,
        image.filename
    )

    image.save(
        image_path
    )

    result = predict_image(
        image_path
    )

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )