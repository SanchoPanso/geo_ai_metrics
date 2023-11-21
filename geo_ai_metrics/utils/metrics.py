import torch
import numpy as np


def match_predictions(pred_classes: np.ndarray, true_classes: np.ndarray, iou: np.ndarray, threshold: float):
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
        conformity = -1 * np.ones((pred_classes.shape[0],)).astype('int32')

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

            conformity[matches[:, 1].astype(int)] = matches[:, 0].astype(int)
            print(matches)
        return matches



if __name__ == '__main__':
     pred_classes = np.array([0, 0, 0, 0])
     true_classes = np.array([0, 0, 0])
     iou = np.array(
          [
               [1, 0, 0, 0],
               [0, 0.33, 0.55, 1],
               [0, 0.5, 0, 0],
          ]
     )
     correct = match_predictions(pred_classes, true_classes, iou, 0.5)
     print(correct)
