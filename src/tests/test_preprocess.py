from preprocess_img import preprocess
import numpy as np


def test_preprocess_shapes_and_range():
    img = np.random.randint(0, 256, size=(600, 400, 3), dtype=np.uint8)
    out = preprocess(img)
    assert out.shape == (1, 512, 512, 1)
    assert np.issubdtype(out.dtype, np.floating)
    assert 0.0 <= float(out.min()) <= 1.0
    assert 0.0 <= float(out.max()) <= 1.0
