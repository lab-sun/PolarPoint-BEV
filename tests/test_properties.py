"""Property tests for graph F1, IoU, and PIDController.

CI and the default profile draw 200 examples. HYPOTHESIS_PROFILE=stress draws 2000.
"""

import importlib.util
import math
import os
import warnings
from collections import deque
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import example, given, settings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, f1_score

from PolarPointBEV.metric import Cal_IoU, ConfusionMatrix, Gragh_Metric, List2List
from PolarPointBEV.model import PIDController
from PolarPointBEV.model_pretrain import PIDController as PretrainPIDController

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
pytestmark = [
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:overflow encountered in scalar multiply:RuntimeWarning"),
]

settings.register_profile("default", max_examples=200, deadline=None, database=None)
settings.register_profile("ci", max_examples=200, deadline=None, database=None)
settings.register_profile("stress", max_examples=2000, deadline=None, database=None)
_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "default")
if _PROFILE not in ("default", "ci", "stress"):
    _PROFILE = "default"
settings.load_profile(_PROFILE)

# stress is settings(max_examples=2000, deadline=None); CI/default stay at 200.
if os.environ.get("HYPOTHESIS_PROFILE") == "stress":
    PROP = settings(max_examples=2000, deadline=None, database=None)
else:
    PROP = settings(max_examples=200, deadline=None, database=None)

_NUM_CLASSES = 3
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_leaderboard_pid():
    path = os.path.join(_REPO_ROOT, "leaderboard", "team_code", "pid_controller.py")
    spec = importlib.util.spec_from_file_location("leaderboard_pid_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PIDController


LeaderboardPIDController = _load_leaderboard_pid()


def _same_number(left, right):
    left = float(left)
    right = float(right)
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _close_number(left, right):
    # ConfusionMatrix.compute casts counts to float32. Match that rounding.
    left = float(left)
    right = float(right)
    if math.isnan(left) and math.isnan(right):
        return True
    if math.isinf(left) or math.isinf(right):
        return left == right
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)


def _same_sequence(left, right):
    assert len(left) == len(right)
    for left_value, right_value in zip(left, right):
        assert _same_number(left_value, right_value)


def _flatten_reference(arrays):
    flat = []
    for array in arrays:
        flat.extend(np.asarray(array).reshape(-1).tolist())
    return flat


def _torch_batches(gts, logits):
    gt_tensors = [torch.as_tensor(np.asarray(gt), dtype=torch.long) for gt in gts]
    logit_tensors = [torch.as_tensor(np.asarray(logit), dtype=torch.float64) for logit in logits]
    return gt_tensors, logit_tensors


def _predicted_labels(logits):
    return np.argmax(np.asarray(logits), axis=1)


def _f1_reference(gts, logits):
    y_true = []
    y_pred = []
    for gt, logit in zip(gts, logits):
        y_true.extend(np.asarray(gt).reshape(-1).tolist())
        y_pred.extend(_predicted_labels(logit).reshape(-1).tolist())
    if len(y_true) == 0:
        return float("nan"), "empty", float("nan")
    micro = f1_score(y_true, y_pred, average="micro")
    macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred)
    return micro, report, macro


def _confusion_reference(gts, logits, num_classes=_NUM_CLASSES):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, logit in zip(gts, logits):
        labels = np.asarray(gt).reshape(-1)
        predicted = _predicted_labels(logit).reshape(-1)
        for label, pred in zip(labels.tolist(), predicted.tolist()):
            if 0 <= label < num_classes:
                matrix[label, pred] += 1
    return matrix


def _iou_reference(matrix):
    counts = np.asarray(matrix, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc_global = np.diag(counts).sum() / counts.sum()
        acc = np.diag(counts) / counts.sum(axis=1)
        iou = np.diag(counts) / (counts.sum(axis=1) + counts.sum(axis=0) - np.diag(counts))
    return acc_global, acc, iou


def _values_in_unit_interval_or_nan(values):
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        if math.isnan(float(value)):
            continue
        assert 0.0 <= float(value) <= 1.0


def _swap_labels_and_predictions(gts, logits):
    swapped_gts = []
    swapped_logits = []
    for gt, logit in zip(gts, logits):
        logit = np.asarray(logit)
        predicted = _predicted_labels(logit)
        swapped_gts.append(np.asarray(predicted, dtype=np.int64))
        fresh = np.full(logit.shape, -1.0, dtype=np.float64)
        gt_array = np.asarray(gt)
        for cls in range(fresh.shape[1]):
            fresh[:, cls][gt_array == cls] = 1.0
        swapped_logits.append(fresh)
    return swapped_gts, swapped_logits


def _perfect_logits(gt):
    logit = np.full((gt.shape[0], _NUM_CLASSES) + tuple(gt.shape[1:]), -2.0, dtype=np.float64)
    for cls in range(_NUM_CLASSES):
        logit[:, cls][gt == cls] = 2.0
    return logit


def _pid_reference(errors, kp, ki, kd, n):
    # List window, not deque: append, then drop the oldest sample past n.
    window = [0] * n
    outputs = []
    for error in errors:
        window.append(error)
        if len(window) > n:
            del window[: len(window) - n]
        if len(window) >= 2:
            integral = np.mean(window)
            derivative = window[-1] - window[-2]
        else:
            integral = 0.0
            derivative = 0.0
        outputs.append(kp * error + ki * integral + kd * derivative)
    return outputs


def _run_pid(controller_cls, errors, kp, ki, kd, n):
    controller = controller_cls(K_P=kp, K_I=ki, K_D=kd, n=n)
    return [controller.step(error) for error in errors]


@st.composite
def _array_lists(draw):
    count = draw(st.integers(min_value=0, max_value=4))
    arrays = []
    for _ in range(count):
        rank = draw(st.integers(min_value=1, max_value=3))
        shape = tuple(draw(st.integers(min_value=0, max_value=4)) for _ in range(rank))
        size = 1
        for dim in shape:
            size *= dim
        raw = draw(
            st.lists(
                st.integers(min_value=-30, max_value=30),
                min_size=size,
                max_size=size,
            )
        )
        arrays.append(np.asarray(raw, dtype=np.int64).reshape(shape))
    return arrays


@st.composite
def _metric_cases(draw):
    mode = draw(
        st.sampled_from(
            [
                "random",
                "zeros",
                "single",
                "perfect",
                "anti",
                "nan",
                "posinf",
                "neginf",
                "huge",
                "outofrange",
                "mixed_batch",
            ]
        )
    )
    height = draw(st.integers(min_value=1, max_value=3))
    width = draw(st.integers(min_value=1, max_value=3))
    if mode == "mixed_batch":
        shapes = [(2, height, width), (2, height, width), (1, height, width)]
    else:
        shapes = []
        for _ in range(draw(st.integers(min_value=1, max_value=3))):
            shapes.append((draw(st.integers(min_value=1, max_value=2)), height, width))
    label = draw(st.integers(min_value=0, max_value=_NUM_CLASSES - 1))
    gts = []
    logits = []
    for batch, gt_height, gt_width in shapes:
        pixels = batch * gt_height * gt_width
        if mode in ("zeros", "anti"):
            gt = np.zeros((batch, gt_height, gt_width), dtype=np.int64)
        elif mode == "single":
            gt = np.full((batch, gt_height, gt_width), label, dtype=np.int64)
        elif mode == "outofrange":
            raw = draw(st.lists(st.integers(min_value=-1, max_value=3), min_size=pixels, max_size=pixels))
            gt = np.asarray(raw, dtype=np.int64).reshape(batch, gt_height, gt_width)
        else:
            raw = draw(
                st.lists(
                    st.integers(min_value=0, max_value=_NUM_CLASSES - 1),
                    min_size=pixels,
                    max_size=pixels,
                )
            )
            gt = np.asarray(raw, dtype=np.int64).reshape(batch, gt_height, gt_width)
        logit = np.zeros((batch, _NUM_CLASSES, gt_height, gt_width), dtype=np.float64)
        if mode == "random":
            raw_logits = draw(
                st.lists(st.integers(min_value=-3, max_value=3), min_size=logit.size, max_size=logit.size)
            )
            logit = np.asarray(raw_logits, dtype=np.float64).reshape(logit.shape)
        elif mode in ("perfect", "single"):
            logit = _perfect_logits(gt)
        elif mode == "anti":
            logit[:, 1] = 4.0
        elif mode == "nan":
            logit[:, 0] = np.nan
            logit[:, 1] = 1.0
        elif mode == "posinf":
            logit[:, 2] = np.inf
        elif mode == "neginf":
            logit[:, :] = -np.inf
            logit[:, 2] = 0.0
        elif mode == "huge":
            logit[:, 0] = 1e308
            logit[:, 1] = -1e308
        elif mode == "outofrange":
            raw_logits = draw(
                st.lists(st.integers(min_value=-2, max_value=2), min_size=logit.size, max_size=logit.size)
            )
            logit = np.asarray(raw_logits, dtype=np.float64).reshape(logit.shape)
        gts.append(gt)
        logits.append(logit)
    if draw(st.booleans()):
        gts.append(np.zeros((0, height, width), dtype=np.int64))
        logits.append(np.zeros((0, _NUM_CLASSES, height, width), dtype=np.float64))
    return gts, logits


@st.composite
def _pid_cases(draw):
    n = draw(st.integers(min_value=0, max_value=6))
    bounded = {"allow_nan": False, "allow_infinity": False, "allow_subnormal": False}
    kp = draw(st.floats(min_value=-4, max_value=4, **bounded))
    ki = draw(st.floats(min_value=-4, max_value=4, **bounded))
    kd = draw(st.floats(min_value=-4, max_value=4, **bounded))
    count = draw(st.integers(min_value=0, max_value=5))
    errors = draw(
        st.lists(
            st.floats(min_value=-20, max_value=20, **bounded),
            min_size=count,
            max_size=count,
        )
    )
    return kp, ki, kd, n, errors


def _zeros_case():
    return (
        [np.zeros((1, 2, 2), dtype=np.int64)],
        [np.zeros((1, _NUM_CLASSES, 2, 2), dtype=np.float64)],
    )


def _single_class_case():
    gt = np.ones((2, 2, 3), dtype=np.int64)
    return [gt], [_perfect_logits(gt)]


def _nan_case():
    gt = np.array([[[0, 1], [2, 0]]], dtype=np.int64)
    logit = np.zeros((1, _NUM_CLASSES, 2, 2), dtype=np.float64)
    logit[:, 0] = np.nan
    logit[0, 1, 0, 1] = 5.0
    return [gt], [logit]


def _inf_case():
    gt = np.zeros((1, 1, 2), dtype=np.int64)
    logit = np.zeros((1, _NUM_CLASSES, 1, 2), dtype=np.float64)
    logit[:, 2] = np.inf
    logit[0, 0, 0, 1] = -np.inf
    return [gt], [logit]


def _huge_case():
    gt = np.array([[[2, 0]]], dtype=np.int64)
    logit = np.zeros((1, _NUM_CLASSES, 1, 2), dtype=np.float64)
    logit[:, 0] = 1e308
    logit[:, 1] = -1e308
    return [gt], [logit]


def _outofrange_case():
    gt = np.array([[[0, -1], [3, 1]]], dtype=np.int64)
    logit = np.zeros((1, _NUM_CLASSES, 2, 2), dtype=np.float64)
    logit[0, 0, 0, 0] = 1.0
    logit[0, 1, 1, 1] = 1.0
    return [gt], [logit]


def test_hypothesis_profile_example_cap():
    if os.environ.get("HYPOTHESIS_PROFILE") == "stress":
        assert PROP.max_examples >= 2000
    else:
        assert PROP.max_examples == 200
    assert PROP.deadline is None


def test_list2list_keeps_large_integers_distinct():
    labels = np.array([2**53, 2**53 + 1, -2**60], dtype=np.int64).reshape(1, 1, 3)
    flat = List2List([labels])
    assert flat == [2**53, 2**53 + 1, -2**60]
    assert [type(value) is int for value in flat] == [True, True, True]


def test_list2list_variable_batch_sizes_and_zero_axes():
    arrays = [
        np.arange(12, dtype=np.int64).reshape(2, 2, 3),
        np.arange(6, dtype=np.int64).reshape(1, 2, 3),
        np.arange(18, dtype=np.int64).reshape(3, 2, 3),
        np.zeros((0, 2, 3), dtype=np.int64),
        np.zeros((2, 0, 3), dtype=np.int64),
        np.zeros((2, 3, 0), dtype=np.int64),
    ]
    assert List2List(arrays) == _flatten_reference(arrays)
    assert List2List([]) == []


def test_graph_metric_empty_pixels_are_nan():
    cases = [
        ([], []),
        ([torch.zeros((0, 2, 2), dtype=torch.long)], [torch.zeros((0, 3, 2, 2))]),
        ([torch.zeros((1, 0, 2), dtype=torch.long)], [torch.zeros((1, 3, 0, 2))]),
        ([torch.zeros((2, 2, 0), dtype=torch.long)], [torch.zeros((2, 3, 2, 0))]),
    ]
    for gts, logits in cases:
        micro, report, macro = Gragh_Metric(gts, logits)
        assert math.isnan(micro)
        assert math.isnan(macro)
        assert report == "empty"


def test_empty_batch_does_not_change_scores():
    gt = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    logits = torch.as_tensor(_perfect_logits(gt.numpy()))
    empty_gt = torch.zeros((0, 2, 2), dtype=torch.long)
    empty_logits = torch.zeros((0, 3, 2, 2))
    full = Gragh_Metric([gt], [logits])
    mixed = Gragh_Metric([empty_gt, gt], [empty_logits, logits])
    assert full[0] == mixed[0]
    assert full[1] == mixed[1]
    assert full[2] == mixed[2]
    full_iou = Cal_IoU([gt], [logits]).mat
    mixed_iou = Cal_IoU([empty_gt, gt], [empty_logits, logits]).mat
    assert torch.equal(full_iou, mixed_iou)


def test_cal_iou_empty_input_is_zero_matrix():
    matrix = Cal_IoU([], [])
    acc_global, acc, iou = matrix.compute()
    assert matrix.mat is not None
    assert tuple(matrix.mat.shape) == (3, 3)
    assert int(matrix.mat.sum()) == 0
    assert torch.isnan(acc_global)
    assert torch.isnan(acc).all()
    assert torch.isnan(iou).all()
    assert "IoU" in str(matrix)


def test_confusion_string_for_four_correct_class_zero():
    gt = torch.zeros((1, 2, 2), dtype=torch.long)
    logits = torch.zeros((1, 3, 2, 2), dtype=torch.float64)
    logits[:, 0] = 1
    text = str(Cal_IoU([gt], [logits]))
    assert text == (
        "global correct: 100.0\n"
        "average row correct: ['100.0', 'nan', 'nan']\n"
        "IoU: ['100.0', 'nan', 'nan']\n"
        "mean IoU: nan"
    )


def test_confusion_string_mean_iou_uses_one_decimal():
    gt = torch.tensor([[[0, 1], [2, 0]]])
    logits = torch.full((1, 3, 2, 2), -1.0)
    logits[0, 0, 0, 0] = 1.0
    logits[0, 1, 0, 1] = 1.0
    logits[0, 2, 1, 0] = 1.0
    logits[0, 0, 1, 1] = 1.0
    text = str(Cal_IoU([gt], [logits]))
    assert text == (
        "global correct: 100.0\n"
        "average row correct: ['100.0', '100.0', '100.0']\n"
        "IoU: ['100.0', '100.0', '100.0']\n"
        "mean IoU: 100.0"
    )


def test_confusion_string_exact_for_known_matrix():
    # Counts are not multiples of ten, so one-decimal text, two-decimal text,
    # and a mean IoU scaled by 10 or 1/100 cannot print the same line.
    matrix = ConfusionMatrix(3)
    matrix.mat = torch.tensor([[2, 1, 0], [0, 1, 1], [1, 0, 3]], dtype=torch.int64)
    assert str(matrix) == (
        "global correct: 66.7\n"
        "average row correct: ['66.7', '50.0', '75.0']\n"
        "IoU: ['50.0', '33.3', '60.0']\n"
        "mean IoU: 47.8"
    )


def test_confusion_counts_are_int64_and_metrics_match_float32():
    labels = torch.tensor([0, 0, 0, 1, 2])
    predicted = torch.tensor([0, 0, 1, 1, 2])
    matrix = ConfusionMatrix(3)
    matrix.update(labels, predicted)
    assert matrix.mat.dtype == torch.int64
    acc_global, acc, iou = matrix.compute()
    assert acc_global.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert iou.dtype == torch.float32
    counts = matrix.mat.to(dtype=torch.float32)
    ref_global = torch.diag(counts).sum() / counts.sum()
    ref_acc = torch.diag(counts) / counts.sum(1)
    ref_iou = torch.diag(counts) / (counts.sum(1) + counts.sum(0) - torch.diag(counts))
    assert torch.equal(acc_global, ref_global)
    assert torch.equal(acc, ref_acc)
    assert torch.equal(iou, ref_iou)
    # 2/3 is not the same float32 and float64 value.
    assert not math.isclose(float(iou[0]), 2.0 / 3.0, rel_tol=0.0, abs_tol=0.0)

    empty = ConfusionMatrix(3)
    empty_global, _, _ = empty.compute()
    assert empty.mat.dtype == torch.int64
    assert empty_global.dtype == torch.float32


def test_confusion_reduce_all_reduces_when_initialized():
    matrix = ConfusionMatrix(3)
    matrix.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    before = matrix.mat.clone()

    def add_one(tensor, *args, **kwargs):
        tensor.add_(1)

    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.barrier") as barrier, patch(
        "torch.distributed.all_reduce", side_effect=add_one
    ) as all_reduce:
        assert matrix.reduce_from_all_processes() is None
    assert barrier.call_count == 1
    assert all_reduce.call_count == 1
    assert torch.equal(matrix.mat, before + 1)


def test_confusion_update_is_additive_and_reset_clears():
    labels = torch.tensor([0, 1, 1])
    predicted = torch.tensor([0, 1, 0])
    once = ConfusionMatrix(3)
    once.update(labels, predicted)
    twice = ConfusionMatrix(3)
    twice.update(labels, predicted)
    twice.update(labels, predicted)
    assert torch.equal(twice.mat, once.mat * 2)
    twice.reset()
    assert torch.equal(twice.mat, torch.zeros((3, 3), dtype=torch.int64))


def test_confusion_ignores_labels_outside_class_range():
    labels = torch.tensor([0, -1, 3, 1, 2])
    predicted = torch.tensor([0, 1, 2, 1, 0])
    matrix = ConfusionMatrix(3)
    matrix.update(labels, predicted)
    expected = _confusion_reference(
        [labels.numpy().reshape(1, 1, 5)],
        [np.eye(3, dtype=np.float64)[predicted.numpy()].reshape(1, 3, 1, 5)],
    )
    assert torch.equal(matrix.mat, torch.as_tensor(expected))


def test_confusion_reduce_is_noop_without_process_group():
    matrix = ConfusionMatrix(3)
    matrix.update(torch.tensor([0, 1]), torch.tensor([2, 1]))
    before = matrix.mat.clone()
    assert matrix.reduce_from_all_processes() is None
    assert torch.equal(matrix.mat, before)
    assert torch.distributed.is_initialized() is False


def test_confusion_reduce_without_process_group_leaves_mat_unchanged():
    matrix = ConfusionMatrix(3)
    matrix.update(torch.tensor([0, 1]), torch.tensor([2, 1]))
    before = matrix.mat.clone()

    # Unavailable distributed package returns before any process-group check.
    with patch("torch.distributed.is_available", return_value=False), patch(
        "torch.distributed.is_initialized", return_value=False
    ) as is_initialized, patch("torch.distributed.barrier") as barrier, patch(
        "torch.distributed.all_reduce"
    ) as all_reduce:
        assert matrix.reduce_from_all_processes() is None
    assert torch.equal(matrix.mat, before)
    assert is_initialized.call_count == 0
    assert barrier.call_count == 0
    assert all_reduce.call_count == 0

    # Available, but no process group is initialized: still no collective.
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=False
    ) as is_initialized, patch("torch.distributed.barrier") as barrier, patch(
        "torch.distributed.all_reduce"
    ) as all_reduce:
        assert matrix.reduce_from_all_processes() is None
    assert torch.equal(matrix.mat, before)
    assert is_initialized.call_count == 1
    assert barrier.call_count == 0
    assert all_reduce.call_count == 0


def test_pid_zero_errors_stay_zero():
    controller = PIDController(K_P=0.75, K_I=0.75, K_D=0.3, n=40)
    assert [controller.step(0.0) for _ in range(5)] == [0.0] * 5


def test_pid_negative_errors_track_absolute_bounds():
    controller = PIDController(K_P=1.0, K_I=0.0, K_D=0.0, n=4)
    controller.step(-4.0)
    assert controller._max == 4.0
    assert controller._min == -4.0
    controller.step(-1.5)
    assert controller._max == 4.0
    assert controller._min == -4.0
    controller.step(-9.0)
    assert controller._max == 9.0
    assert controller._min == -9.0


def test_pid_window_of_one_is_proportional():
    controller = PIDController(K_P=1.5, K_I=4.0, K_D=-3.0, n=1)
    assert controller.step(2.0) == pytest.approx(3.0)
    assert controller.step(-1.0) == pytest.approx(-1.5)


def test_pid_constant_error_reaches_kp_plus_ki():
    n = 5
    error = 1.25
    kp, ki, kd = 0.5, 0.25, 0.1
    controller = PIDController(K_P=kp, K_I=ki, K_D=kd, n=n)
    values = [controller.step(error) for _ in range(n + 2)]
    for value in values[n - 1 :]:
        assert value == pytest.approx((kp + ki) * error)


@PROP
@given(arrays=_array_lists())
@example(arrays=[])
@example(arrays=[np.zeros((2, 3, 4), dtype=np.int64)])
@example(arrays=[np.zeros((0, 2, 2), dtype=np.int64), np.zeros((1, 0, 2), dtype=np.int64)])
@example(arrays=[np.array([2**53, 2**53 + 1], dtype=np.int64)])
def test_list2list_matches_independent_flatten(arrays):
    flat = List2List(arrays)
    assert flat == _flatten_reference(arrays)
    assert len(flat) == sum(int(np.asarray(array).size) for array in arrays)
    if flat and all(np.asarray(array).dtype.kind == "i" for array in arrays):
        assert all(type(value) is int for value in flat)


@PROP
@given(case=_metric_cases())
@example(case=_zeros_case())
@example(case=_single_class_case())
@example(case=_nan_case())
@example(case=_inf_case())
@example(case=_huge_case())
@example(case=_outofrange_case())
def test_graph_metrics_match_references(case):
    gts, logits = case
    gt_tensors, logit_tensors = _torch_batches(gts, logits)
    micro, report, macro = Gragh_Metric(gt_tensors, logit_tensors)
    expected_micro, expected_report, expected_macro = _f1_reference(gts, logits)
    assert _same_number(micro, expected_micro)
    assert _same_number(macro, expected_macro)
    assert report == expected_report
    assert isinstance(report, str)
    _values_in_unit_interval_or_nan([micro, macro])

    matrix = Cal_IoU(gt_tensors, logit_tensors)
    expected_matrix = _confusion_reference(gts, logits)
    assert torch.equal(matrix.mat, torch.as_tensor(expected_matrix))
    in_range_pixels = 0
    for gt in gts:
        gt_array = np.asarray(gt)
        in_range_pixels += int(np.logical_and(gt_array >= 0, gt_array < _NUM_CLASSES).sum())
    assert int(matrix.mat.sum()) == in_range_pixels

    acc_global, acc, iou = matrix.compute()
    expected_global, expected_acc, expected_iou = _iou_reference(expected_matrix)
    assert _close_number(acc_global, expected_global)
    assert all(_close_number(left, right) for left, right in zip(acc.tolist(), expected_acc.tolist()))
    assert all(_close_number(left, right) for left, right in zip(iou.tolist(), expected_iou.tolist()))
    _values_in_unit_interval_or_nan(acc.tolist())
    _values_in_unit_interval_or_nan(iou.tolist())
    if in_range_pixels == int(sum(np.asarray(gt).size for gt in gts)) and in_range_pixels > 0:
        accuracy = float(torch.diag(matrix.mat).sum()) / float(matrix.mat.sum())
        assert float(micro) == pytest.approx(accuracy)

    labels_in_range = all(int(np.asarray(gt).min()) >= 0 and int(np.asarray(gt).max()) < _NUM_CLASSES for gt in gts if np.asarray(gt).size)
    if labels_in_range:
        swapped_gts, swapped_logits = _swap_labels_and_predictions(gts, logits)
        swapped = Cal_IoU(*_torch_batches(swapped_gts, swapped_logits)).mat
        assert torch.equal(swapped, matrix.mat.t())


@PROP
@given(case=_pid_cases())
@example(case=(0.75, 0.75, 0.3, 40, [0.0, 0.0, 0.0]))
@example(case=(0.75, 0.75, 0.3, 40, [float("nan")]))
@example(case=(0.75, 0.75, 0.3, 40, [float("inf")]))
@example(case=(0.75, 0.75, 0.3, 40, [float("-inf")]))
@example(case=(0.75, 0.75, 0.3, 40, [1e308]))
@example(case=(0.75, 0.75, 0.3, 40, [-1e308]))
@example(case=(1.0, 0.0, 0.0, 1, [1.0, -2.0, 3.5]))
@example(case=(0.5, 0.25, 0.1, 0, [1.0, 2.0]))
@example(case=(0.0, 0.0, 0.0, 4, [1.0, -1.0, 5.0]))
@example(case=(0.75, 0.75, 0.3, 4, [1.0, 1.0, 1.0, 1.0, 1.0]))
@example(case=(0.75, 0.75, 0.3, 2, []))
def test_pid_matches_reference_and_other_copies(case):
    kp, ki, kd, n, errors = case
    actual = _run_pid(PIDController, errors, kp, ki, kd, n)
    expected = _pid_reference(errors, kp, ki, kd, n)
    _same_sequence(actual, expected)
    _same_sequence(actual, _run_pid(PretrainPIDController, errors, kp, ki, kd, n))
    _same_sequence(actual, _run_pid(LeaderboardPIDController, errors, kp, ki, kd, n))
    def _moderate(values):
        # Subnormals and overflows make a linear gain fail by an ulp or become inf.
        return all(math.isfinite(float(value)) and (float(value) == 0.0 or 1e-8 <= abs(float(value)) <= 1e8) for value in values)

    if _moderate((kp, ki, kd)) and _moderate(errors) and _moderate(actual):
        negated = _run_pid(PIDController, [-value for value in errors], kp, ki, kd, n)
        _same_sequence(negated, [-value for value in actual])
        scaled_inputs = [2.0 * value for value in errors]
        if _moderate(scaled_inputs):
            scaled = _run_pid(PIDController, scaled_inputs, kp, ki, kd, n)
            _same_sequence(scaled, [2.0 * value for value in actual])
    if n > 0:
        controller = PIDController(K_P=kp, K_I=ki, K_D=kd, n=n)
        for error in errors:
            controller.step(error)
        assert len(controller._window) == n
        assert isinstance(controller._window, deque)
