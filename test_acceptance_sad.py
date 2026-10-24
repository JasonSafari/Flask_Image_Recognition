"""Acceptance tests for error handling when no file is uploaded."""


def test_acceptance_missing_file(client):
    """
    Test Case: No File Uploaded
    - Purpose: Validate the application's behavior when no file is provided in the upload request.
    - Scenario:
        - Simulate a POST request to the `/prediction` route with no file data.
        - Assert the response status code is 200 (to indicate a valid request was processed).
        - Verify that the response includes an appropriate error message.
    """
    # Simulate a POST request with no file data
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    # Assertions:
    # 1. Ensure the response status code is 200, indicating the request was processed.
    assert response.status_code == 200
    # 2. Check for a meaningful error message in the response data.
    #    Modify the message check if your application uses a different error response text.
    assert b"File cannot be processed" in response.data  # Expected error message