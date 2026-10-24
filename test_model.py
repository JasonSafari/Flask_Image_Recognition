import os
import pytest
import numpy as np
from keras.models import load_model
from model import preprocess_img, predict_result  # Adjust based on your structure
from io import BytesIO
from unittest.mock import patch
 
from PIL import Image


# Load the model before tests run
@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    model = load_model("digit_model.h5")  # Adjust path as needed
    return model

# Basic Tests

def test_preprocess_img():
    """Test the preprocess_img function."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)

    # Check that the output shape is as expected
    assert processed_img.shape == (1, 224, 224, 3), "Processed image shape should be (1, 224, 224, 3)"

    # Check that values are normalized (between 0 and 1)
    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1, "Image pixel values should be normalized between 0 and 1"


def test_predict_result(model):
    """Test the predict_result function."""
    img_path = "test_images/4/Sign 4 (92).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)

    # Print the prediction for debugging
    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    # Check that the prediction is an integer (convert if necessary)
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer class index"


# Advanced Tests

def test_invalid_image_path():
    """Test preprocess_img with an invalid image path."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")

def test_image_shape_on_prediction(model):
    """Test the prediction output shape."""
    img_path = "test_images/5/Sign 5 (86).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Ensure that the prediction output is an integer
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "The prediction should be an integer"

def test_model_predictions_consistency(model):
    """Test that predictions for the same input are consistent."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)

    # Make multiple predictions
    predictions = [predict_result(processed_img) for _ in range(5)]

    # Check that all predictions are the same
    assert all(p == predictions[0] for p in predictions), "Predictions for the same input should be consistent"

def make_png_bytes(size=(32, 32), mode="RGB", color="white"):
    """Build an in-memory PNG image and return it as a BytesIO object."""
    buffer = BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer
 
 
def post_file(client, file_obj, filename):
    """POST a file to /prediction using the form field name the app expects."""
    return client.post(
        "/prediction",
        data={"file": (file_obj, filename)},
        content_type="multipart/form-data",
    )
 
 
# ---------- Home page ----------
 
def test_home_page_loads(client):
    """The home page can be accessed successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.data  # page is not empty
 
 
def test_unknown_route_returns_404(client):
    """An unknown URL returns 404."""
    assert client.get("/does-not-exist").status_code == 404
 
 
# ---------- Valid request ----------
 
def test_valid_upload_returns_prediction(client):
    """A valid image upload shows the predicted digit (model is mocked)."""
    with patch("app.preprocess_img", return_value="fake-array") as mock_pre, \
         patch("app.predict_result", return_value=7) as mock_predict:
        response = post_file(client, make_png_bytes(), "digit.png")
 
    assert response.status_code == 200
    assert b"7" in response.data
    assert b"File cannot be processed" not in response.data
    mock_pre.assert_called_once()
    mock_predict.assert_called_once_with("fake-array")
 
 
def test_prediction_is_converted_to_string(client):
    """Edge case: digit 0 is falsy but must still be displayed."""
    with patch("app.preprocess_img", return_value="fake-array"), \
         patch("app.predict_result", return_value=0):
        response = post_file(client, make_png_bytes(), "zero.png")
 
    assert response.status_code == 200
    assert b"File cannot be processed" not in response.data
    assert b"0" in response.data
 
 
# ---------- Missing file ----------
 
def test_missing_file_is_handled(client):
    """A POST with no uploaded file shows the error message instead of crashing."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
 
 
def test_wrong_form_field_name_is_handled(client):
    """Edge case: a file sent under the wrong field name counts as missing."""
    response = client.post(
        "/prediction",
        data={"image": (make_png_bytes(), "digit.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
 
 
def test_empty_filename_is_handled(client):
    """Edge case: submitting the form without choosing a file."""
    response = post_file(client, BytesIO(b""), "")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
 
 
# ---------- Invalid or unexpected input ----------
 
def test_non_image_file_is_handled(client):
    """A text file renamed as an image is rejected by the real preprocessing."""
    response = post_file(client, BytesIO(b"this is not an image"), "fake.png")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
 
 
def test_empty_file_is_handled(client):
    """Edge case: a zero-byte upload is rejected cleanly."""
    response = post_file(client, BytesIO(b""), "empty.png")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
 
 
def test_grayscale_image_is_handled(client):
    """Edge case: a 1-channel image does not match the model's RGB input."""
    response = post_file(client, make_png_bytes(mode="L", color=128), "gray.png")
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
 
 
def test_model_failure_is_handled(client):
    """If the model itself raises, the user sees the error message."""
    with patch("app.preprocess_img", return_value="fake-array"), \
         patch("app.predict_result", side_effect=RuntimeError("model exploded")):
        response = post_file(client, make_png_bytes(), "digit.png")
 
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
 
 
def test_get_on_prediction_route_not_allowed(client):
    """The prediction route only accepts POST."""
    assert client.get("/prediction").status_code == 405


