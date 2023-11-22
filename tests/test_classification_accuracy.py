import pytest
import sys
import numpy as np
from geo_ai_metrics.classification.classification_accuracy import ClassificationAccuracy
from geo_ai_metrics.annotation.annotation import Annotation, AnnotatedImage, AnnotatedObject


def test_classification_accuracy():
    metric = ClassificationAccuracy()
    gt_bboxes = [
        AnnotatedObject(category_id=0, bbox=[0, 0, 1, 1]),
        AnnotatedObject(category_id=0, bbox=[1, 1, 1, 1]),
        AnnotatedObject(category_id=0, bbox=[2, 2, 1, 1]),
    ]

    pred_bboxes = [
        AnnotatedObject(category_id=0, bbox=[0, 0, 1, 1]),
        AnnotatedObject(category_id=1, bbox=[1, 1, 1, 1]),
        AnnotatedObject(category_id=0, bbox=[3, 3, 1, 1]),
    ]

    gt_image = AnnotatedImage(width=100, height=100, annotations=gt_bboxes)
    pred_image = AnnotatedImage(width=100, height=100, annotations=pred_bboxes)

    gt_annot = Annotation(categories=['0', '1'], images={'0': gt_image})
    pred_annot = Annotation(categories=['0', '1'], images={'0': pred_image}) 

    acc = metric(pred_annot, gt_annot)

    assert acc == pytest.approx(0.5)
    

if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
