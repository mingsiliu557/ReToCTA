#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from medpy import metric
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score


def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return 100. #float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return 100.#float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)


def data_process(pred, label, threshold=0.5):
    '''
    pred = np.array(pred*255).astype(np.uint8)
    label = np.array(label*255).astype(np.uint8)

    for N in range(pred.shape[0]):
        for C in range(pred.shape[1]):
            _, pred[N][C] = cv2.threshold(pred[N][C], 0, 255, cv2.THRESH_OTSU)
            _, label[N][C] = cv2.threshold(label[N][C], 0, 255, cv2.THRESH_OTSU)
    pred, label = pred // 255, label // 255
    '''
    pred = np.array(pred)
    label = np.array(label)

    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    return pred.astype(np.uint8), label.astype(np.uint8)


def dice_compute(test, reference):
    batch_size = reference.shape[0]
    disc_dices, cup_dices = [], []

    for batch in range(batch_size):
        disc_dice, cup_dice = dice(test=test[batch][0], reference=reference[batch][0]), \
                              dice(test=test[batch][1], reference=reference[batch][1])
        disc_dices.append(disc_dice)
        cup_dices.append(cup_dice)
    return disc_dices, cup_dices


def asd_compute(test, reference):
    batch_size = reference.shape[0]
    disc_asds, cup_asds = [], []

    for batch in range(batch_size):
        disc_asd, cup_asd = avg_surface_distance(test=test[batch][0], reference=reference[batch][0]), \
                            avg_surface_distance(test=test[batch][1], reference=reference[batch][1])
        disc_asds.append(disc_asd)
        cup_asds.append(cup_asd)
    return disc_asds, cup_asds


def hd_compute(test, reference):
    batch_size = reference.shape[0]
    disc_hds, cup_hds = [], []

    for batch in range(batch_size):
        disc_hd, cup_hd = hausdorff_distance_95(test=test[batch][0], reference=reference[batch][0]), \
                          hausdorff_distance_95(test=test[batch][1], reference=reference[batch][1])
        disc_hds.append(disc_hd)
        cup_hds.append(cup_hd)
    return disc_hds, cup_hds


def dice_metric(pred, label):
    batch_size = pred.shape[0]
    dices = []
    smooth = 1e-6

    for batch in range(batch_size):
        intersection = (pred[batch] * label[batch]).sum()
        dice = (2 * intersection + smooth) / (pred[batch].sum() + label[batch].sum() + smooth)
        dices.append(dice*100.)
    return dices


def _as_2d_bool(mask):
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask after squeeze, got shape {mask.shape}.")
    return mask.astype(bool)


def cldice_metric(pred, label):
    batch_size = pred.shape[0]
    cldices = []

    for batch in range(batch_size):
        pred_mask = _as_2d_bool(pred[batch])
        label_mask = _as_2d_bool(label[batch])

        pred_skeleton = skeletonize(pred_mask)
        label_skeleton = skeletonize(label_mask)

        pred_skeleton_sum = pred_skeleton.sum()
        label_skeleton_sum = label_skeleton.sum()

        if pred_skeleton_sum == 0 and label_skeleton_sum == 0:
            cldices.append(100.0 if np.array_equal(pred_mask, label_mask) else 0.0)
            continue

        tprec = 0.0
        tsens = 0.0
        if pred_skeleton_sum > 0:
            tprec = np.logical_and(pred_skeleton, label_mask).sum() / pred_skeleton_sum
        if label_skeleton_sum > 0:
            tsens = np.logical_and(label_skeleton, pred_mask).sum() / label_skeleton_sum

        cldice = 0.0
        if tprec + tsens > 0:
            cldice = (2 * tprec * tsens) / (tprec + tsens)
        cldices.append(cldice * 100.)

    return cldices


def _betti_numbers(mask):
    mask = _as_2d_bool(mask)
    foreground_structure = np.ones((3, 3), dtype=np.uint8)
    background_structure = np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]], dtype=np.uint8)

    _, beta_0 = ndi.label(mask, structure=foreground_structure)

    background_labels, background_count = ndi.label(~mask, structure=background_structure)
    if background_count == 0:
        return int(beta_0), 0

    border_labels = np.concatenate([
        background_labels[0, :],
        background_labels[-1, :],
        background_labels[:, 0],
        background_labels[:, -1],
    ])
    border_labels = set(np.unique(border_labels))
    beta_1 = sum(1 for label_idx in range(1, background_count + 1)
                 if label_idx not in border_labels)

    return int(beta_0), int(beta_1)


def beta_error_metric(pred, label):
    batch_size = pred.shape[0]
    beta_errors = []

    for batch in range(batch_size):
        pred_beta_0, pred_beta_1 = _betti_numbers(pred[batch])
        label_beta_0, label_beta_1 = _betti_numbers(label[batch])
        beta_error = abs(pred_beta_0 - label_beta_0) + abs(pred_beta_1 - label_beta_1)
        beta_errors.append(float(beta_error))

    return beta_errors


def calculate_metrics(test, reference, include_topology=False):
    # test, reference = data_process(pred=test, label=reference, threshold=0.5)
    # disc_dice, cup_dice = dice_metric(test, reference)
    # disc_dis, cup_dis = asd_compute(test, reference)
    test, reference = data_process(pred=test, label=reference, threshold=0.5)
    dice = dice_metric(test, reference)
    if include_topology:
        cldice = cldice_metric(test, reference)
        beta_error = beta_error_metric(test, reference)
        return [dice, cldice, beta_error]
    # disc_dis, cup_dis = hd_compute(test, reference)
    return [dice]


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}
