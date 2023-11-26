import numpy as np
import torch
import cv2
from typing import Any, Tuple
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage
from geo_ai_metrics.utils.utils import box_iou, mask_iou, match_predictions


class Localization360Accuracy:
    def __init__(self, mode='box'):
        self.mode = mode
    
    def __call__(self, pred_annot: Annotation, gt_annot: Annotation, *args: Any, **kwds: Any) -> Any:
        
        deviations = []
        
        for name in pred_annot.images:
            if name not in gt_annot.images:
                continue

            pred_image = pred_annot.images[name]
            gt_image = gt_annot.images[name]

            cur_deviations = self.get_deviations(pred_image, gt_image)
            deviations += cur_deviations    
        
        deviations = np.array(deviations)
        mean = deviations.mean()
        std = deviations.std()
        
        return mean, std
    
    def get_deviations(self, pred_image: AnnotatedImage, gt_image: AnnotatedImage, iou_thresh=0.5) -> Tuple[np.ndarray, np.ndarray]:
        pred_classes = self.get_classes(pred_image)
        pred_boxes = self.get_boxes(pred_image)
        pred_segm = self.get_segmentations(pred_image)
        pred_pts = self.get_points(pred_image)
        
        gt_classes = self.get_classes(gt_image)
        gt_boxes = self.get_boxes(gt_image)
        gt_segm = self.get_segmentations(gt_image)
        gt_pts = self.get_points(gt_image)
    
        width, height = pred_image.width, pred_image.height
        
        if self.mode == 'box':
            iou = box_iou(pred_boxes, gt_boxes)
            matches = match_predictions(pred_classes, gt_classes, iou, iou_thresh)   
                
        elif self.mode == 'mask':
            iou = np.zeros((len(pred_segm), len(gt_segm)))
            for i, pred_s in enumerate(pred_segm):
                for j, gt_s in enumerate(gt_segm):
                    iou[i, j] = self.get_segments_iou(pred_s, gt_s, (width, height))
            matches = match_predictions(pred_classes, gt_classes, iou, iou_thresh)
            
        else:
            raise ValueError(f"mode \'{self.mode}\' is not valid")
        
        deviations = []
        
        for m in matches:
            gt_idx, pred_idx = m
            if len(pred_pts[pred_idx]) == 0 or len(gt_pts[gt_idx]) == 0:
                continue
            pred_center = np.array(pred_pts[pred_idx]).mean(axis=0)
            gt_center = np.array(gt_pts[gt_idx]).mean(axis=0)
            deviation = np.sqrt(((pred_center - gt_center) ** 2).sum())
            deviations.append(deviation)
                
        return deviations
    
    def get_segments_center(self, segments: list, size: tuple):
        w, h = size
        mask = np.zeros((h, w), dtype='uint8')
        
        new_segments = []
        for s in segments:
            s = np.array(s).reshape(-1, 1, 2)
            new_segments.append(s)
        cv2.fillPoly(mask, new_segments, 1)
        
        xs, ys = np.where(mask)
        x_center = xs.sum() / xs.shape[0]
        y_center = ys.sum() / ys.shape[0]
        
        return x_center, y_center    

    def get_classes(self, image: AnnotatedImage):
        """Get classes as array (N,)"""
        classes = []
        
        for bbox in image.annotations:
            cat = bbox.category_id
            classes.append(cat)
        
        classes = np.array(classes).astype('int32')
        return classes
        
    def get_boxes(self, image: AnnotatedImage):
        """Get xyxy boxes as array (N, 4)"""
        boxes = []
        
        for bbox in image.annotations:
            xywh = bbox.bbox
            x, y, w, h = xywh
            xyxy = [x, y, x + w, y + h]
            boxes.append(xyxy)
        
        boxes = np.array(boxes)

        return boxes
    
    def get_segmentations(self, image: AnnotatedImage):
        """Get list of segments with len N"""
        segments = []

        for bbox in image.annotations:
            sgm = bbox.segmentation
            segments.append(sgm)

        return segments
    
    def get_points(self, image: AnnotatedImage):
        points = []

        for bbox in image.annotations:
            pts = bbox.points
            points.append(pts)
            
        return points
    
    def get_annotations_info(self, 
                             image: AnnotatedImage, 
                             get_classses=True,
                             get_boxes=True):
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


