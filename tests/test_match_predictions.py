import pytest
import sys
import numpy as np
from geo_ai_metrics.utils import match_predictions


def test_match_predictions():
    threshold = 0.5
    pred_classes = np.array([0, 0, 0, 0])
    true_classes = np.array([0, 0, 0])
    iou = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.33, 0.55, 1],
            [0, 0.5, 0, 0],
        ]
    )

    expected_matches = np.array([[1, 3], [0, 0], [2, 1]], dtype='int32')
    got_matches = match_predictions(pred_classes, true_classes, iou, threshold)
    print(expected_matches)
    print(got_matches)
    assert (expected_matches == got_matches).min() == True


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
