import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from geo_ai_metrics.annotation import AnnotatedImage


def match_predictions(pred_classes: np.ndarray, true_classes: np.ndarray, iou: np.ndarray, threshold: float, agnostic=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (np.ndarray): Predicted class indices of shape(N,).
            true_classes (np.ndarray): Target class indices of shape(M,).
            iou (np.ndarray): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            threshold (float): IoU threshold

        Returns:
            (np.ndarray): Correct tensor of shape(N,) for IoU threshold.
            (np.ndarray): Mathes array of shape(K, 2) where the ith row is [m, n] that represents a match between mth target and nth prediction

        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        conformity = -1 * np.ones((pred_classes.shape[0], 1)).astype('int32')

        if not agnostic:
            # LxD matrix where L - labels (rows), D - detections (columns)
            correct_class = true_classes[:, np.newaxis] == pred_classes
            iou = iou * correct_class  # zero out the wrong classes

        matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
        matches = np.array(matches).T

        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]

                # matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                
                matches = matches[np.sort(np.unique(matches[:, 1], return_index=True)[1])]
                matches = matches[np.sort(np.unique(matches[:, 0], return_index=True)[1])]

            conformity[matches[:, 1].astype(int), 0] = matches[:, 0].astype(int)
        return matches


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.array): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard iou if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1: np.ndarray, box2: np.ndarray, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.ndarray): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (np.ndarray): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    (a1, a2), (b1, b2) = np.split(box1.reshape(-1, 1, 4), 2, axis=2), np.split(box2.reshape(1, -1, 4), 2, axis=2)
    inter = np.prod(np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, np.inf), axis=2)
    
    # box1 = torch.tensor(box1)
    # box2 = torch.tensor(box2)
    
    # (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    # inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # a1, a2 = a1.numpy(), a2.numpy()
    # b1, b2 = b1.numpy(), b2.numpy()
    # inter = inter.numpy()
    
    # IoU = inter / (area1 + area2 - inter)
    return inter / (np.prod((a2 - a1), 2) + np.prod((b2 - b1), 2) - inter + eps)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    # intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    # union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    # return intersection / (union + eps)

    intersection = np.clip(mask1 @ mask2.T, 0, np.inf)
    union = (mask1.sum(1)[:, np.newaxis] + mask2.sum(1)[np.newaxis]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def get_classes_boxes_segments(image: AnnotatedImage):
    "Get classes as array (N,), xyxy boxes as array (N, 4), list of segments with len N"

    classes = []
    boxes = []
    segments = []

    for bbox in image.annotations:
        cat = bbox.category_id
        xywh = bbox.bbox
        sgm = bbox.segmentation
        x, y, w, h = xywh
        xyxy = [x, y, x + w, y + h]
        classes.append(cat)
        boxes.append(xyxy)
        segments.append(sgm)
    
    classes = np.array(classes).astype('int32')
    boxes = np.array(boxes)

    return classes, boxes, segments


def get_segments_iou(segments1: list, segments2: list, size: tuple) -> float:
    w, h = size
    masks = [np.zeros((h, w), dtype='uint8') for i in range(2)]
    segments_list = [segments1, segments2]
    
    for i, segments in enumerate(segments_list):
        mask = masks[i]
        new_segments = []
        for s in segments:
            s = np.array(s).reshape(-1, 1, 2)
            new_segments.append(s)
        cv2.fillPoly(mask, new_segments, 1)
    
    mask1 = masks[0].reshape(1, -1)
    mask2 = masks[1].reshape(1, -1)

    iou = mask_iou(mask1, mask2)[0, 0]
    return iou
    


### -----------------------------------------------------------------------------------------------

def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]) ** 2 + (kpt1[:, None, :, 1] - kpt2[..., 1]) ** 2  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / (2 * sigma) ** 2 / (area[:, None, None] + eps) / 2  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return (torch.exp(-e) * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def smooth_BCE(eps=0.1):
    """
    Computes smoothed positive and negative Binary Cross-Entropy targets.

    This function calculates positive and negative label smoothing BCE targets based on a given epsilon value.
    For implementation details, refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.

    Args:
        eps (float, optional): The epsilon value for label smoothing. Defaults to 0.1.

    Returns:
        (tuple): A tuple containing the positive and negative label smoothing BCE targets.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.array): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task='detect'):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == 'detect' else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in (None, 0.001) else conf  # apply 0.25 if default val conf is passed
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, labels):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class).
            labels (Array[M, 5]): Ground truth bounding boxes and their associated class labels.
                                  Each row should contain (class, x1, y1, x2, y2).
        """
        if labels.size(0) == 0:  # Check if labels is empty
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1]) if self.task == 'detect' else (tp, fp)  # remove background class if task=detect

 



def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

