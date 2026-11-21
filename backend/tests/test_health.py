from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_reports_image_runtime():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["runtime"] == "ONNX Runtime / CPU"
    assert "model_ready" in response.json()


def test_video_endpoint_is_retired():
    response = client.post("/api/v1/predict/video")
    assert response.status_code == 410


def test_rejects_non_image_upload():
    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("clip.mp4", b"not an image", "video/mp4")},
    )
    assert response.status_code == 415


def test_rejects_invalid_image_bytes():
    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")},
    )
    assert response.status_code == 400



def test_valid_image_upload_returns_model_unavailable_without_checkpoint(
    monkeypatch, tmp_path
):
    from io import BytesIO
    from PIL import Image

    from backend.app.main import predictor
    from backend.app.inference.predictor import Predictor

    # Create a predictor pointing to a nonexistent checkpoint.
    unavailable_predictor = Predictor(
        model_path=str(tmp_path / "missing_model.onnx"),
        threshold=0.60,
        image_size=224,
    )

    # Replace the application's predictor for this test.
    monkeypatch.setattr(
        "backend.app.main.predictor",
        unavailable_predictor,
    )

    buffer = BytesIO()
    Image.new("RGB", (64, 64), color="white").save(
        buffer, format="JPEG"
    )

    response = client.post(
        "/api/v1/predict/image",
        files={
            "file": (
                "sign.jpg",
                buffer.getvalue(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "model_unavailable"
    assert "message" in response.json()

