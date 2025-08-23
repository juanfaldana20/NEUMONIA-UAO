import numpy as np
import integrator


class DummyModel:
    def predict(self, x):
        # 3 clases: bacteriana, normal, viral -> forzamos "normal"
        return np.array([[0.1, 0.8, 0.1]])


def dummy_grad_cam(array):
    return np.zeros((512, 512, 3), dtype=np.uint8)


def test_predict_with_monkeypatch(monkeypatch):
    
    monkeypatch.setattr(integrator, "model_fun", lambda: DummyModel())
    monkeypatch.setattr(integrator, "grad_cam", dummy_grad_cam)

    array = np.zeros((512, 512, 3), dtype=np.uint8)
    label, proba, heatmap = integrator.predict(array)

    assert label == "normal"
    assert 0.0 < proba <= 100.0
    assert heatmap.shape == (512, 512, 3)
