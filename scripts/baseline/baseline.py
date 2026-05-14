from ultralytics import YOLO


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        box_a, box_b: [x1, y1, x2, y2]  (left edge, top edge, right edge, bottom edge)

    Returns:
        float: IoU score in [0, 1]
    """
    # Find the coordinates of the intersection rectangle
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    # If there's no intersection, width or height will be ≤ 0
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Area of each box individually
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Union = sum of areas minus the part we counted twice
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

