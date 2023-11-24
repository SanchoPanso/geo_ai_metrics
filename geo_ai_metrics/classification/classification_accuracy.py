import numpy as np
import torch
import cv2
from typing import Any, Tuple
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage
from geo_ai_metrics.utils.utils import box_iou, mask_iou, match_predictions


class ClassificationAccuracy:
    def __init__(self, mode='box'):
        self.mode = mode # 'box' or 'mask'
    
    def __call__(self, pred_annot: Annotation, gt_annot: Annotation, *args: Any, **kwds: Any) -> Any:

        tpc = 0
        fpc = 0

        for name in pred_annot.images:
            if name not in gt_annot.images:
                continue

            pred_image = pred_annot.images[name]
            gt_image = gt_annot.images[name]
            
            matched_pred_classes, matched_gt_classes = self.get_matched_classes(pred_image, gt_image)
            class_equality = matched_pred_classes == matched_gt_classes
            
            cur_tpc = class_equality.sum()
            cur_fpc = len(class_equality) - cur_tpc

            tpc += cur_tpc
            fpc += cur_fpc

        class_accuracy = tpc / (tpc + fpc)
        return class_accuracy
    
    def get_matched_classes(self, pred_image: AnnotatedImage, gt_image: AnnotatedImage, iou_thresh=0.5) -> Tuple[np.ndarray, np.ndarray]:
        pred_classes, pred_boxes, pred_segm = self.get_classes_boxes_segments(pred_image)
        gt_classes, gt_boxes, gt_segm = self.get_classes_boxes_segments(gt_image)
        width, height = pred_image.width, pred_image.height
        
        if self.mode == 'box':
            iou = box_iou(pred_boxes, gt_boxes)
            matches = match_predictions(pred_classes, gt_classes, iou, iou_thresh, agnostic=True)   
        
        elif self.mode == 'mask':
            iou = np.zeros((len(pred_segm), len(gt_segm)))
            for i, pred_s in enumerate(pred_segm):
                for j, gt_s in enumerate(gt_segm):
                    iou[i, j] = self.get_segments_iou(pred_s, gt_s, (width, height))
            matches = match_predictions(pred_classes, gt_classes, iou, iou_thresh, agnostic=True)
        else:
            raise ValueError(f"mode \'{self.mode}\' is not valid")
        
        matched_pred_classes = pred_classes[matches[:, 1]]
        matched_gt_classes = gt_classes[matches[:, 0]]
        
        return matched_pred_classes, matched_gt_classes

    def get_classes_boxes_segments(self, image: AnnotatedImage):
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
    
    def get_segments_iou(self, segments1: list, segments2: list, size: tuple) -> float:
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