"""Flask web app for digit image recognition."""
import logging

from flask import Flask, render_template, request

from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


@app.route("/")
def main():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Preprocess the uploaded image, predict the digit, and render the result."""
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except Exception:  # pylint: disable=broad-exception-caught
        logging.exception("Prediction failed")
        return render_template("result.html", err="File cannot be processed.")


if __name__ == "__main__":
    app.run(port=9000, debug=True)