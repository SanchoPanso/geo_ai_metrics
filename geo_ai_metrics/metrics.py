import torch
import numpy as np


def match_predictions(iouv, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]

                    # matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    
                    matches = matches[np.sort(np.unique(matches[:, 1], return_index=True)[1])]
                    matches = matches[np.sort(np.unique(matches[:, 0], return_index=True)[1])]

                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device), matches.astype(int)



if __name__ == '__main__':
     iouv = torch.tensor([0.5])
     pred_classes = torch.tensor([0, 0, 0, 0])
     true_classes = torch.tensor([0, 0, 0])
     iou = torch.tensor(
          [
               [1, 0, 0, 0],
               [0, 0.33, 0.55, 1],
               [0, 0.5, 0, 0],
          ]
     )
     correct = match_predictions(iouv, pred_classes, true_classes, iou)
     print(correct)
