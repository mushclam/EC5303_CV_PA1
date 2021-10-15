import numpy as np


def calculate_IoU(n_label, predicted, gt):
    if n_label <= 2:
        intersection = np.logical_and(predicted, gt)
        union = np.logical_or(predicted, gt)
        iou_score = np.sum(intersection) / np.sum(union) * 100.
    elif n_label > 2:
        scores = []
        for i in range(n_label):
            t_gt = np.where(gt == i+1, 1., 0.)
            t_predicted = np.where(predicted == i+1, 1., 0.)
            intersection = np.logical_and(t_predicted, t_gt)
            union = np.logical_or(t_predicted, t_gt)
            score = np.sum(intersection) / np.sum(union) * 100.
            scores.append(score)
        iou_score = np.mean(scores)
    print(f'IoU score: {iou_score}')