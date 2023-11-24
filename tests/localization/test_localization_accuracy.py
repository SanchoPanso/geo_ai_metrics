import pytest
import sys
import numpy as np
from geo_ai_metrics.localization.localization_accuracy import LocalizationAccuracy 
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage, AnnotatedObject


def test_localization_accuracy():
    metric = LocalizationAccuracy()
    gt_bboxes = [
        AnnotatedObject(category_id=0, bbox=[0, 0, 10, 10]),
        AnnotatedObject(category_id=0, bbox=[1, 1, 10, 10]),
    ]

    pred_bboxes = [
        AnnotatedObject(category_id=0, bbox=[1, 0, 10, 10]),
        AnnotatedObject(category_id=0, bbox=[2, 1, 10, 10]),
    ]

    gt_image = AnnotatedImage(width=100, height=100, annotations=gt_bboxes)
    pred_image = AnnotatedImage(width=100, height=100, annotations=pred_bboxes)

    gt_annot = Annotation(categories=['0', '1'], images={'0': gt_image})
    pred_annot = Annotation(categories=['0', '1'], images={'0': pred_image}) 

    mean, std = metric(pred_annot, gt_annot)
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(0.0)


def test_localization_accuracy_mask():
    metric = LocalizationAccuracy(mode='mask')
    gt_bboxes = [
        AnnotatedObject(category_id=0, bbox=[0, 0, 10, 10],
                        segmentation=[[0, 0, 0, 10, 10, 10, 10, 0]]),
        AnnotatedObject(category_id=0, bbox=[1, 1, 10, 10],
                        segmentation=[[1, 1, 1, 11, 11, 11, 11, 1]]),
    ]

    pred_bboxes = [
        AnnotatedObject(category_id=0, bbox=[1, 0, 10, 10],
                        segmentation=[[1, 0, 1, 10, 11, 10, 11, 0]]),
        AnnotatedObject(category_id=0, bbox=[2, 1, 10, 10],
                        segmentation=[[2, 1, 2, 11, 12, 11, 12, 1]]),
    ]

    gt_image = AnnotatedImage(width=100, height=100, annotations=gt_bboxes)
    pred_image = AnnotatedImage(width=100, height=100, annotations=pred_bboxes)

    gt_annot = Annotation(categories=['0', '1'], images={'0': gt_image})
    pred_annot = Annotation(categories=['0', '1'], images={'0': pred_image}) 

    mean, std = metric(pred_annot, gt_annot)
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(0.0)
    

if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
