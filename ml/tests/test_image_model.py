import torch

from ml.models.image_classifier import build_image_model


def test_mobilenet_v3_small_output_shape():
    model = build_image_model(num_classes=26, pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 26)
