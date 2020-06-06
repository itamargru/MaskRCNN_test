
import numpy as np
import mrcnn.utils as utils
import matplotlib.pyplot as plt

# TODO: create function in Tester that compute the inputs for this function
def compute_batch_iou(num_classes, list_pred_masks, list_gt_masks, matched_classes):
    """
    This Function takes a list of masks (preds and gt) coming from several images and computes the average IoU,
    given the IoU of each image as an array with dim = num_classes+1.
    note: the average is taken among images, not among instances.

    :param num_classes: ndarray (1,num_images): how many classes exist in each image? (background doesn't count)
    :param list_pred_masks: list of the masks which were predicted during evaluation
    :param list_gt_masks: list of the ground truth masks
    :param matched_classes: list of ndarray contain the class per mask. The indices are corresponding between
                        the prediction to the GT.
    :return IoU_mean: a numpy.ndarray of shape [num_classes+1,1], with the average IoU per class.

    :assumption: the callee needs to sort the elements in the lists according to the output of utils.compute_matches
    """
    IoU_mean = np.zeros((num_classes + 1 ,1))
    cnt_instances = np.zeros((num_classes + 1, 1))
    N = len(list_pred_masks)
    for i in np.arange(N):
        IoU_curr, existing_classes_curr = compute_mean_iou_per_image(num_classes, list_pred_masks[i], list_gt_masks[i], matched_classes[i])
        IoU_mean += IoU_curr
        cnt_instances += existing_classes_curr
    select_nonzero = cnt_instances != 0
    IoU_mean[select_nonzero] = IoU_mean[select_nonzero] / cnt_instances[select_nonzero]
    return IoU_mean

def compute_mean_iou_per_image(num_classes, pred_masks, gt_masks, matched_classes):
    '''
    This function computes the average IoU per image.
    each image can contain different amount and kind of mask's classes,
    the average is taken among each class separately.
    :param num_classes: ndarray (1,num_images): how many classes exist in each image? (background doesn't count)
    :param pred_masks: the image's predicted masks
    :param gt_masks: the image's ground truth masks
    :param matched_classes: ndarray it's indices has correspondence between the prediction to the GT.
    :return IoU_per_class: mean IoU over all instances from the same class. ndarray with shape[num_classes + 1, 1]
    :return class_exist: ndarray with shape[num_classes + 1, 1], which contain 1 at indces where the class has at least
                        one instance in the image
    '''
    # IoU: mat. IoU score of each possible pair of masks gt Vs preds
    IoU = utils.compute_overlaps_masks(pred_masks, gt_masks)
    diag_indices = np.arange(IoU.shape[0])  # ndarray. IoU of matched masks
    IoU_diag = IoU[diag_indices, diag_indices]
    IoU_per_class = np.zeros((num_classes+1, 1))
    cnt_classes = np.zeros((num_classes+1, 1))
    for i in np.arange(IoU_diag.shape[0]):
        IoU_per_class[matched_classes[i]] += IoU_diag[i]
        cnt_classes[matched_classes[i]] += 1
    select_nonzero = cnt_classes != 0
    IoU_per_class[select_nonzero] = IoU_per_class[select_nonzero] / cnt_classes[select_nonzero]
    class_exist = cnt_classes > 0
    return IoU_per_class, class_exist.astype(np.uint8)

def plot_session(model):
    # Get input and output to classifier and mask heads.
    mrcnn = model.run_graph([image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])
    pass

import itertools
def get_confusion_matrix(num_classes, gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1))

    for i, j in itertools.product(range(overlaps.shape[0]), range(overlaps.shape[1])):
        if overlaps[i, j] > threshold:
            confusion_matrix[pred_class_ids[i], gt_class_ids[j]] += 1

    thresh = threshold
    if overlaps.shape[0] != 0 and overlaps.shape[1] != 0:
        max_iou_pred = np.max(overlaps, axis=1)
        select_smaller_thresh =  max_iou_pred < thresh
        class_pred = pred_class_ids[select_smaller_thresh]
        confusion_matrix[class_pred, np.zeros_like(class_pred)] += 1

        max_iou_gt = np.max(overlaps, axis=0)
        select_smaller_thresh = max_iou_gt < thresh
        class_gt = gt_class_ids[select_smaller_thresh]
        confusion_matrix[np.zeros_like(class_gt), class_gt] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    confusion_matrix: [pred_boxes, gt_boxes] IoU confusion_matrix of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]),
                                  range(confusion_matrix.shape[1])):
        color = ("white" if confusion_matrix[i, j] > thresh
                 else "black" if confusion_matrix[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}".format(confusion_matrix[i, j]),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Confusion Matrix")
    plt.show()
    return fig

def acumulate_confussion_matrix_multiple_thresh(matrices, threshs, num_classes, gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names):
    for i, thresh in enumerate(threshs):
        matrices[i,:,:] += get_confusion_matrix(num_classes, gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=thresh)
    return matrices, threshs

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_curve
def plot_roc_curve(list_gt_masks, list_pred_masks, matched_classes):
    """

    N = len(list_gt_masks) - number of images
    Mi = number of segments in the i'th image
    M = for i in [1...N] M += Mi
    :param list_gt_masks: list of ground truth masks each of shape [Height, Width, Mi]
    :param list_pred_masks: list of predicted masks each of shape [Height, Width, Mi]
    :param matched_classes: the element i is the class of the i'th masks
    :return:
    """
    gt_masks = np.concatenate(list_gt_masks, axis=2)
    gt_masks = np.transpose(gt_masks, (2, 0, 1))
    pred_masks = np.transpose(np.concatenate(list_pred_masks, axis=3), (3, 0, 1, 2))
    matched_classes = np.concatenate(matched_classes)
    num_classes = 5#np.max(matched_classes) + 1

    array_shape = tuple(list(gt_masks.shape) + [num_classes])
    masks = np.zeros(array_shape, dtype=np.bool)

    # gt_masks make into shape [M, height, width, classes]
    for i in np.arange(num_classes):
        masks[matched_classes == i, :, :, i] = True
    gt_masks = masks * np.expand_dims(gt_masks, axis=-1)
    gt_masks = gt_masks.astype(np.bool)
    # pred_masks = masks * np.expand_dims(pred_masks, axis=3)
    # pred_masks = pred_masks.astype(np.bool)
    segs = gt_masks
    p = pred_masks

    plt.figure(figsize=(10, 10))
    for i in range(p.shape[-1]):
        fpr, tpr, _ = roc_curve(segs[:, :, :, i].ravel(), p[:, :, :, i].ravel())

        _p = np.round(p[:, :, :, i].ravel()).astype(np.int32)
        bas = balanced_accuracy_score(segs[:, :, :, i].ravel(), _p)

        plt.subplot(4, 4, i + 1)
        plt.plot(fpr, tpr)
        plt.title("Class " + str(i))
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

    plt.tight_layout()
    plt.show()


def collect_roc_data(list_gt_masks, list_pred_masks, matched_classes):
    """

    N = len(list_gt_masks) - number of images
    Mi = number of segments in the i'th image
    M = for i in [1...N] M += Mi
    :param list_gt_masks: list of ground truth masks each of shape [Height, Width, Mi]
    :param list_pred_masks: list of predicted masks each of shape [Height, Width, Mi]
    :param matched_classes: the element i is the class of the i'th masks
    :return:
    """
    gt_masks = np.concatenate(list_gt_masks, axis=2)
    gt_masks = np.transpose(gt_masks, (2, 0, 1))
    pred_masks = np.transpose(np.concatenate(list_pred_masks, axis=3), (3, 0, 1, 2))
    matched_classes = np.concatenate(matched_classes)
    num_classes = 5#np.max(matched_classes) + 1

    array_shape = tuple(list(gt_masks.shape) + [num_classes])
    masks = np.zeros(array_shape, dtype=np.bool)

    # gt_masks make into shape [M, height, width, classes]
    for i in np.arange(num_classes):
        masks[matched_classes == i, :, :, i] = True
    gt_masks = masks * np.expand_dims(gt_masks, axis=-1)
    gt_masks = gt_masks.astype(np.bool)
    # pred_masks = masks * np.expand_dims(pred_masks, axis=3)
    # pred_masks = pred_masks.astype(np.bool)
    segs = gt_masks
    p = pred_masks

    fpr_list = []
    tpr_list = []
    for i in range(p.shape[-1]):
        fpr, tpr, _ = roc_curve(segs[:, :, :, i].ravel(), p[:, :, :, i].ravel())
        fpr_list += [fpr]
        tpr_list += [tpr]

    fpr = np.stack(fpr_list, axis=-1)
    tpr = np.stack(tpr_list, axis=-1)
    return fpr, tpr

def get_IoU_from_matches(match_pred2gt, matched_classes, ovelaps):
    """
    if given an image, claculate the IoU of the segments in the image
    :param match_pred2gt: maps index of predicted segment to index of ground truth segment
    :param matched_classes: maps index of predicted segment to class number
    :param ovelaps: maps [predicted segment index, gt segment index] to the IoU value of the segments
    :return:
        1. IoUs - IoU for all segments
        2. IoUs_classes - mean IoU per class
    """
    IoUs = [ [] for _ in range(5) ]
    match_pred2gt = match_pred2gt.astype(np.int32)
    for pred, gt in enumerate(match_pred2gt):
        if gt < 0:
            continue
        IoUs[matched_classes[pred]].append(ovelaps[pred, gt])

    # mean segments's IoU according to classes
    IoUs_classes = np.zeros((5, 1))
    for class_idx, lst in enumerate(IoUs):
        if not lst:
            continue
        arr = np.array(lst)
        IoUs_classes[class_idx] = (np.mean(arr))

    return IoUs, IoUs_classes
