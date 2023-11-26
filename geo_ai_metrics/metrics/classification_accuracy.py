import os
import numpy as np
import glob
import torch
import cv2
from typing import Any, Tuple
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage
from geo_ai_metrics.utils.utils import box_iou, mask_iou, match_predictions
from geo_ai_metrics.utils.utils import get_classes_boxes_segments, get_segments_iou
from geo_ai_metrics.utils.visualization import draw_segments


class ClassificationAccuracy:
    def __init__(self, mode='box'):
        self.mode = mode # 'box' or 'mask'
    
    def __call__(
        self, 
        pred_annot: Annotation, 
        gt_annot: Annotation,
        save: bool = True,
        save_dir: str = './outs',
        images_dir: str = None,  
        *args: Any, 
        **kwds: Any) -> Any:

        tpc = 0
        fpc = 0

        for name in pred_annot.images:
            if name not in gt_annot.images:
                continue

            pred_image = pred_annot.images[name]
            gt_image = gt_annot.images[name]
            
            matched_pred_classes, matched_gt_classes, matches = self.get_matched_classes(pred_image, gt_image)
            class_equality = matched_pred_classes == matched_gt_classes
            
            if save:
                self.visualize(gt_image, pred_image, matches, class_equality, images_dir, name, save_dir)
            
            cur_tpc = class_equality.sum()
            cur_fpc = len(class_equality) - cur_tpc

            tpc += cur_tpc
            fpc += cur_fpc

        class_accuracy = tpc / (tpc + fpc)
        return class_accuracy
    
    def get_matched_classes(
        self, 
        pred_image: AnnotatedImage, 
        gt_image: AnnotatedImage,
        iou_thresh=0.5) -> Tuple[np.ndarray, np.ndarray]:
        
        pred_classes, pred_boxes, pred_segm = get_classes_boxes_segments(pred_image)
        gt_classes, gt_boxes, gt_segm = get_classes_boxes_segments(gt_image)
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
        
        return matched_pred_classes, matched_gt_classes, matches

    def visualize(
        self, 
        gt_image: AnnotatedImage, 
        pred_image: AnnotatedImage, 
        matches: np.ndarray,
        class_equality: np.ndarray,
        images_dir: str, 
        name: str, 
        save_dir: str):
        
        pred_classes, pred_boxes, pred_segm = get_classes_boxes_segments(pred_image)
        gt_classes, gt_boxes, gt_segm = get_classes_boxes_segments(gt_image)
        width, height = pred_image.width, pred_image.height
        
        img = self.get_img_for_vis(images_dir, name, (width, height))
        os.makedirs(save_dir, exist_ok=True)
        
        vis_img = img.copy()
        for i, m in enumerate(matches):
            gt_idx, pred_idx = m
            gt_box = gt_boxes[gt_idx].astype('int32')
            pred_box = pred_boxes[pred_idx].astype('int32')
            cls_is_equal = class_equality[i]
            
            main_color = int(100 + 155 * i / len(matches))
            if cls_is_equal:
                gt_color = (0, main_color, 0)
                pred_color = (40, main_color, 0)
            else:
                gt_color = (0, 0, main_color)
                pred_color = (40, 0, main_color)
            
            # cv2.rectangle(vis_img, gt_box[:2], gt_box[2:], gt_color, 2)
            # cv2.rectangle(vis_img, pred_box[:2], pred_box[2:], pred_color, 2)
            
            vis_img = draw_segments(vis_img, gt_segm[gt_idx], 7, gt_color)
            vis_img = draw_segments(vis_img, pred_segm[pred_idx], 4, pred_color)
            
        cv2.imwrite(os.path.join(save_dir, f"{name}.jpg"), vis_img)
        
    
    def get_img_for_vis(self, images_dir: str, name: str, imgsz: tuple) -> np.ndarray:
        img_paths = glob.glob(os.path.join(images_dir, name + '*'))
        if len(img_paths) == 0:
            return np.zeros((imgsz[1], imgsz[0], 3), dtype='uint8')
        
        img = cv2.imread(img_paths[0])
        if img is None:
            return np.zeros((imgsz[1], imgsz[0], 3), dtype='uint8')
        
        return img
    

    