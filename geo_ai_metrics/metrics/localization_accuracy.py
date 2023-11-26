import numpy as np
import torch
import cv2
import os
import glob
from typing import Any, Tuple
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage
from geo_ai_metrics.utils.utils import box_iou, mask_iou, match_predictions
from geo_ai_metrics.utils.utils import get_classes_boxes_segments, get_segments_iou
from geo_ai_metrics.utils.visualization import draw_segments


class LocalizationAccuracy:
    def __init__(self, mode='box'):
        self.mode = mode
    
    def __call__(self, 
        pred_annot: Annotation, 
        gt_annot: Annotation,
        scale: float = 1,
        save: bool = False,
        save_dir: str = './outs',
        images_dir: str = None,  
        *args: Any, 
        **kwds: Any) -> Any:
        
        deviations = []
        
        for name in pred_annot.images:
            if name not in gt_annot.images:
                continue

            pred_image = pred_annot.images[name]
            gt_image = gt_annot.images[name]

            cur_deviations = self.get_deviations(pred_image, gt_image, name, save, images_dir, save_dir)
            deviations += cur_deviations    
        
        deviations = np.array(deviations)
        mean = deviations.mean() * scale
        std = deviations.std() * scale
        
        return mean, std
    
    def get_deviations(self, pred_image: AnnotatedImage, gt_image: AnnotatedImage, name, save, images_dir, save_dir, iou_thresh=0.5) -> Tuple[np.ndarray, np.ndarray]:
        pred_classes, pred_boxes, pred_segm = get_classes_boxes_segments(pred_image)
        gt_classes, gt_boxes, gt_segm = get_classes_boxes_segments(gt_image)
        width, height = pred_image.width, pred_image.height
        
        if self.mode == 'box':
            iou = box_iou(pred_boxes, gt_boxes)
            matches = match_predictions(pred_classes, gt_classes, iou, iou_thresh)   
            matched_pred_boxes = pred_boxes[matches[:, 1]]
            matched_gt_boxes = gt_boxes[matches[:, 0]]
            
            pred_centers_x = (matched_pred_boxes[:, 0] + matched_pred_boxes[:, 2]) / 2
            pred_centers_y = (matched_pred_boxes[:, 1] + matched_pred_boxes[:, 3]) / 2
            gt_centers_x = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2
            gt_centers_y = (matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
            
            deviations = np.sqrt((pred_centers_x - gt_centers_x) ** 2 + 
                                 (pred_centers_y - gt_centers_y) ** 2)
            deviations = deviations.tolist()
        
        elif self.mode == 'mask':
            iou = np.zeros((len(pred_segm), len(gt_segm)))
            for i, pred_s in enumerate(pred_segm):
                for j, gt_s in enumerate(gt_segm):
                    iou[i, j] = get_segments_iou(pred_s, gt_s, (width, height))
            matches = match_predictions(pred_classes, gt_classes, iou, iou_thresh)
            
            deviations = []
            for m in matches:
                gt_idx, pred_idx = m
                pred_xy = self.get_segments_center(pred_segm[pred_idx], (width, height))
                gt_xy = self.get_segments_center(gt_segm[gt_idx], (width, height))
                deviation = np.sqrt((pred_xy[0] - gt_xy[0]) ** 2 + (pred_xy[1] - gt_xy[1]) ** 2)
                deviations.append(deviation)
        
        else:
            raise ValueError(f"mode \'{self.mode}\' is not valid")
        
        if save:
            self.visualize(gt_image, pred_image, matches, deviations, images_dir, name, save_dir)
        
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
    
    def visualize(
        self, 
        gt_image: AnnotatedImage, 
        pred_image: AnnotatedImage, 
        matches: np.ndarray,
        deviations,
        images_dir: str, 
        name: str, 
        save_dir: str):
        
        pred_classes, pred_boxes, pred_segm = self.get_classes_boxes_segments(pred_image)
        gt_classes, gt_boxes, gt_segm = self.get_classes_boxes_segments(gt_image)
        width, height = pred_image.width, pred_image.height
        
        img = self.get_img_for_vis(images_dir, name, (width, height))
        os.makedirs(save_dir, exist_ok=True)
        
        vis_img = img.copy()
        for i, m in enumerate(matches):
            gt_idx, pred_idx = m
            gt_box = gt_boxes[gt_idx].astype('int32')
            pred_box = pred_boxes[pred_idx].astype('int32')
            
            main_color = int(100 + 155 * i / len(matches))
            gt_color = (main_color, 0, 0)
            pred_color = (main_color, 0, 40)
            
            # cv2.rectangle(vis_img, gt_box[:2], gt_box[2:], gt_color, 2)
            # cv2.rectangle(vis_img, pred_box[:2], pred_box[2:], pred_color, 2)
            
            vis_img = draw_segments(vis_img, gt_segm[gt_idx], 7, gt_color)
            vis_img = draw_segments(vis_img, pred_segm[pred_idx], 4, pred_color)
            
            label = f'{deviations[i]:.2f}'
            xc, yc = (gt_box[0] + gt_box[2]) // 2, (gt_box[1] + gt_box[3]) // 2

            #((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(vis_img, text=label, org=(xc, yc),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0),
                        lineType=cv2.LINE_AA)
            
        cv2.imwrite(os.path.join(save_dir, f"{name}.jpg"), vis_img)    
    
    def get_img_for_vis(self, images_dir: str, name: str, imgsz: tuple) -> np.ndarray:
        img_paths = glob.glob(os.path.join(images_dir, name + '*'))
        if len(img_paths) == 0:
            return np.zeros((imgsz[1], imgsz[0], 3), dtype='uint8')
        
        img = cv2.imread(img_paths[0])
        if img is None:
            return np.zeros((imgsz[1], imgsz[0], 3), dtype='uint8')
        
        return img
    
    
    