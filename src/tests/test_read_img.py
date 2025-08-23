from read_img import read_jpg_file
from PIL import Image
import numpy as np
import cv2


def test_read_jpg_file_roundtrip(tmp_path):
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), img)

    img2, img2show = read_jpg_file(str(p))
    assert img2.shape == (100, 100, 3)
    assert img2.dtype == np.uint8
    assert isinstance(img2show, Image.Image)
