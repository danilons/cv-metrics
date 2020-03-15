import numpy as np
from itertools import count
from pycocotools.mask import iou as compute_iou
from object_detection.metrics import coco_tools


class MapEvaluator:

    def __init__(self, categories):
        self.images = []
        self.annotations = []
        self.categories = categories
        self.detections = []
        self.id_gen = count(0, 1)
        self.groundtruth_boxes = {}
        self.groundtruth_labels = {}
        self.detection_boxes = {}
        self.detection_labels = {}

    def add_ground_truth_image(self, image_id, boxes, labels):
        for n, bbox in enumerate(boxes):
            self.annotations.append({
                "id": next(self.id_gen),
                "image_id": image_id,
                "category_id": int(labels[n]),
                "bbox": bbox,
                "area": 0,
                "iscrowd": 0
            })
        self.images.append({'id': image_id})
        self.groundtruth_boxes[image_id] = boxes
        self.groundtruth_labels[image_id] = labels

    def add_detection(self, image_id, boxes, labels, score):
        for n, bbox in enumerate(boxes):
            self.detections.append({
                'image_id': image_id,
                'category_id': int(labels[n]),
                'bbox': bbox,
                'score': float(score[n])
            })
        self.detection_boxes[image_id] = boxes
        self.detection_labels[image_id] = labels

    @property
    def groundtruth(self):
        return {"annotations": self.annotations,
                "images": self.images,
                "categories": self.categories}

    def evaluate(self):
        groundtruth = coco_tools.COCOWrapper(self.groundtruth)
        detections = groundtruth.LoadAnnotations(self.detections)
        evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections)
        metrics, _ = evaluator.ComputeMetrics()
        return metrics

    def confusion_matrix(self, iou_threshold=.3):
        confusion_matrix = np.zeros(shape=(len(self.categories) + 1,
                                           len(self.categories) + 1))

        for image in self.images:
            image_id = image['id']
            gt_boxes = self.groundtruth_boxes[image_id]
            gt_labels = self.groundtruth_labels[image_id]
            dt_boxes = self.detection_boxes[image_id]
            dt_labels = self.detection_labels[image_id]

            matches = []
            for i in range(len(gt_boxes)):
                for j in range(len(dt_boxes)):
                    iou = compute_iou(gt_boxes[i][np.newaxis], dt_boxes[j][np.newaxis], [0])
                    if iou > iou_threshold:
                        matches.append([i, j, iou])

            matches = np.array(matches)
            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate detections from the list.
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate ground truths from the list.
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            for i in range(len(gt_boxes)):
                if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                    confusion_matrix[gt_labels[i] - 1][dt_labels[int(matches[matches[:, 0] == i, 1][0])] - 1] += 1
                else:
                    confusion_matrix[gt_labels[i] - 1][confusion_matrix.shape[1] - 1] += 1

            for i in range(len(dt_boxes)):
                if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][dt_labels[i] - 1] += 1
        return confusion_matrix

    def plot_confusion_matrix(self, classes=None):
        import matplotlib.pyplot as plt

        cm = self.confusion_matrix()
        classes = classes or ["{}".format(lbl + 1) for lbl in range(32)]
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title="Confusion matrix",
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        _ = plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, "{}".format(int(cm[i, j])),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

