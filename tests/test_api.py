from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "alcohol": 14.23,
            "color_intensity": 5.64,
            "proline": 1065
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [1, 2, 3]
    assert result["class_name"] in [
        "Wine Class 1",
        "Wine Class 2",
        "Wine Class 3"
    ]


def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "alcohol": -1,
            "color_intensity": 5.64,
            "proline": 1065
        }
    )

    assert response.status_code == 422