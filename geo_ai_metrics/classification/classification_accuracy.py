from typing import Any
from geo_ai_metrics.annotation.annotation import Annotation
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

            # pred_boxes = get_boxes(pred_image)
            # gt_boxes = get_boxes(gt_image)

            # iou = box_iou(pred_boxes, gt_boxes)





