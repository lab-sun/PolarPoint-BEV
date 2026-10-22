import pytest
import torch

from PolarPointBEV.config import GlobalConfig
from PolarPointBEV.metric import Cal_IoU, Gragh_Metric
from PolarPointBEV.model import PIDController
from PolarPointBEV.resnet import resnet34 as resnet34_impl
from PolarPointBEV.resnet_original import resnet34 as resnet34_original_impl


def _forbid_download(*args, **kwargs):
    raise AssertionError("pretrained weight download was attempted")


def _unpretrained(factory):
    def _fn(*args, **kwargs):
        return factory(pretrained=False)

    return _fn


def _train_batch(batch_size):
    # train.py feeds front_img from the 256x900 RGB camera and
    # state = cat(speed, target_point, one-hot command).
    speed = torch.rand(batch_size, 1) / 12.0
    target_point = torch.randn(batch_size, 2)
    command = torch.zeros(batch_size, 6)
    command[:, 3] = 1
    state = torch.cat([speed, target_point, command], 1)
    front_img = torch.randn(batch_size, 3, 256, 900)
    return front_img, state, target_point


def _assert_planner_outputs(pred, batch_size, pred_len, with_graph):
    expected = {
        "pred_speed": (batch_size, 1),
        "pred_value_traj": (batch_size, 1),
        "pred_features_traj": (batch_size, 256),
        "pred_wp": (batch_size, pred_len, 2),
        "pred_value_ctrl": (batch_size, 1),
        "pred_features_ctrl": (batch_size, 256),
        "mu_branches": (batch_size, 2),
        "sigma_branches": (batch_size, 2),
    }
    if with_graph:
        expected["graph"] = (batch_size, 3, 16, 27)
    assert set(pred) == set(expected) | {"future_feature", "future_mu", "future_sigma"}
    for key, shape in expected.items():
        assert tuple(pred[key].shape) == shape
        assert torch.isfinite(pred[key]).all()
    assert (pred["mu_branches"] >= 0).all()
    assert (pred["sigma_branches"] >= 0).all()
    for key in ("future_feature", "future_mu", "future_sigma"):
        assert len(pred[key]) == pred_len
    for step in range(pred_len):
        assert tuple(pred["future_feature"][step].shape) == (batch_size, 256)
        assert tuple(pred["future_mu"][step].shape) == (batch_size, 2)
        assert tuple(pred["future_sigma"][step].shape) == (batch_size, 2)
        assert torch.isfinite(pred["future_feature"][step]).all()
        assert (pred["future_mu"][step] >= 0).all()
        assert (pred["future_sigma"][step] >= 0).all()


def test_xplan_forward_matches_train_inputs(monkeypatch):
    import PolarPointBEV.model as model_mod
    import PolarPointBEV.resnet as resnet_mod

    monkeypatch.setattr(resnet_mod, "load_state_dict_from_url", _forbid_download)
    monkeypatch.setattr(model_mod, "resnet34", _unpretrained(resnet34_impl))

    config = GlobalConfig()
    model = model_mod.XPlan(config).eval()
    batch_size = 2
    front_img, state, target_point = _train_batch(batch_size)
    assert tuple(state.shape) == (batch_size, 9)

    with torch.no_grad():
        pred = model(front_img, state, target_point)

    _assert_planner_outputs(pred, batch_size, config.pred_len, with_graph=True)


def test_pretrain_tcp_forward_matches_train_inputs(monkeypatch):
    import PolarPointBEV.model_pretrain as pretrain_mod
    import PolarPointBEV.resnet_original as resnet_mod

    monkeypatch.setattr(resnet_mod, "load_state_dict_from_url", _forbid_download)
    monkeypatch.setattr(pretrain_mod, "resnet34", _unpretrained(resnet34_original_impl))

    config = GlobalConfig()
    model = pretrain_mod.TCP(config).eval()
    batch_size = 2
    front_img, state, target_point = _train_batch(batch_size)

    with torch.no_grad():
        pred = model(front_img, state, target_point)

    _assert_planner_outputs(pred, batch_size, config.pred_len, with_graph=False)
    assert "graph" not in pred


def test_pid_controller_step():
    kp, ki, kd, n = 0.75, 0.75, 0.3, 40
    pid = PIDController(K_P=kp, K_I=ki, K_D=kd, n=n)

    err1 = 1.0
    # The window is created full of zeros, so the first step drops one zero.
    expected1 = kp * err1 + ki * (err1 / n) + kd * (err1 - 0.0)
    assert pid.step(err1) == pytest.approx(expected1)

    err2 = -0.25
    expected2 = kp * err2 + ki * ((err1 + err2) / n) + kd * (err2 - err1)
    assert pid.step(err2) == pytest.approx(expected2)


def _logits_from_labels(labels):
    logits = torch.full((labels.shape[0], 3, labels.shape[1], labels.shape[2]), -5.0)
    for batch in range(labels.shape[0]):
        for row in range(labels.shape[1]):
            for col in range(labels.shape[2]):
                logits[batch, int(labels[batch, row, col]), row, col] = 5.0
    return logits


def test_graph_metric_and_iou_small_inputs():
    gt_a = torch.tensor(
        [
            [[0, 1, 2], [2, 1, 0]],
            [[1, 1, 0], [2, 2, 0]],
        ],
        dtype=torch.long,
    )
    gt_b = torch.tensor(
        [
            [[2, 0, 1], [0, 2, 1]],
        ],
        dtype=torch.long,
    )
    pred_a = _logits_from_labels(gt_a)
    pred_b = _logits_from_labels(gt_b)
    # One wrong pixel in the second batch.
    pred_b[0, :, 0, 0] = -5.0
    pred_b[0, 0, 0, 0] = 5.0

    f1_overall, report, f1_mean = Gragh_Metric([gt_a, gt_b], [pred_a, pred_b])
    pixels = gt_a.numel() + gt_b.numel()
    assert f1_overall == pytest.approx((pixels - 1) / pixels)
    assert 0.0 < f1_mean < 1.0
    assert isinstance(report, str)
    assert "f1-score" in report

    matrix = Cal_IoU([gt_a, gt_b], [pred_a, pred_b])
    assert tuple(matrix.mat.shape) == (3, 3)
    assert int(matrix.mat.sum()) == pixels
    assert int(torch.diag(matrix.mat).sum()) == pixels - 1
    assert "IoU" in str(matrix)

    f1_one, report_one, f1_mean_one = Gragh_Metric([gt_a], [pred_a])
    assert f1_one == pytest.approx(1.0)
    assert f1_mean_one == pytest.approx(1.0)
    assert isinstance(report_one, str)
    matrix_one = Cal_IoU([gt_a], [pred_a])
    assert int(torch.diag(matrix_one.mat).sum()) == gt_a.numel()
