
import numpy as np
import mrcnn.utils as utils

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
    pass
