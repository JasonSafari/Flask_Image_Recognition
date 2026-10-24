"""Shared pytest fixtures for the Flask app."""
import pytest

from app import app as flask_app


@pytest.fixture
def client():
    """Provide a Flask test client."""
    with flask_app.test_client() as test_client:
        yield test_client