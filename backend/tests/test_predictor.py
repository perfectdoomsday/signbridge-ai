import numpy as np
from PIL import Image

from backend.app.inference.predictor import Predictor


def test_preprocess_returns_normalized_nchw_tensor():
    image = Image.new("RGB", (640, 480), color=(120, 80, 40))
    result = Predictor.preprocess(image, 224)
    assert result.shape == (1, 3, 224, 224)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_missing_model_is_explicitly_unavailable(tmp_path):
    predictor = Predictor(str(tmp_path / "missing.onnx"))
    assert not predictor.ready
    result = predictor.predict(Image.new("RGB", (224, 224)))
    assert result["status"] == "model_unavailable"
