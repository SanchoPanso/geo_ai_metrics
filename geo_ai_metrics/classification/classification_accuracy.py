import numpy as np
import torch
from typing import Any
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage
from geo_ai_metrics.utils.utils import box_iou, match_predictions


class ClassificationAccuracy:
    def __init__(self):
        pass
    
    def __call__(self, pred_annot: Annotation, gt_annot: Annotation, *args: Any, **kwds: Any) -> Any:

        tpc = 0
        fpc = 0

        for name in pred_annot.images:
            if name not in gt_annot.images:
                continue

            pred_image = pred_annot.images[name]
            gt_image = gt_annot.images[name]

            pred_classes, pred_boxes = self.get_classes_and_boxes(pred_image)
            gt_classes, gt_boxes = self.get_classes_and_boxes(gt_image)

            iou = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()
            matches = match_predictions(pred_classes, gt_classes, iou, 0.5, agnostic=True)

            matched_pred_classes = pred_classes[matches[:, 1]]
            matched_gt_classes = gt_classes[matches[:, 0]]

            class_equality = matched_pred_classes == matched_gt_classes
            cur_tpc = class_equality.sum()
            cur_fpc = len(class_equality) - cur_tpc

            tpc += cur_tpc
            fpc += cur_fpc

        class_accuracy = tpc / (tpc + fpc)
        return class_accuracy
    

    def get_classes_and_boxes(self, image: AnnotatedImage):
        "Get classes as array (N,) and xyxy boxes as array (N, 4)"

        classes = []
        boxes = []

        for bbox in image.annotations:
            cat = bbox.category_id
            xywh = bbox.bbox
            x, y, w, h = xywh
            xyxy = [x, y, x + w, y + h]
            classes.append(cat)
            boxes.append(xyxy)
        
        classes = np.array(classes).astype('int32')
        boxes = np.array(boxes)

        return classes, boxes


