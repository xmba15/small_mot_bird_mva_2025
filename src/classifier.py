from typing import Any, Dict

import cv2
import numpy as np
import timm
import torch
from mmengine.structures.instance_data import InstanceData

__all__ = ("Classifier",)


class Classifier:
    __DEFAULT_CONFIG = {
        "patch_size": 196,
        "conf_thr": 0.5,
    }

    def __init__(
        self,
        encoder_name: str,
        checkpoint_file: str,
        device: str,
        opt_config: Dict[str, Any] | None = None,
    ):
        self.config = self.__DEFAULT_CONFIG.copy()
        if opt_config is not None:
            self.config.update(opt_config)

        self.device = torch.device(device)

        self.model = timm.create_model(
            encoder_name,
            num_classes=1,
            checkpoint_path=checkpoint_file,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def filter_by_fine_classification(
        self,
        detections: InstanceData,
        image_path: str,
    ):
        bboxes = detections.bboxes
        scores = detections.scores
        labels = detections.labels

        if len(bboxes) == 0:
            return detections

        image = cv2.imread(image_path)[..., [2, 1, 0]]

        patches = []
        for idx, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = map(int, bbox)
            patch_size = self.config["patch_size"]
            patches.append(
                cv2.resize(
                    image[ymin:ymax, xmin:xmax],
                    (patch_size, patch_size),
                )
            )
        del image

        patches = np.stack(patches, axis=0)
        patches = patches.transpose(0, 3, 1, 2)
        patches = torch.from_numpy(patches).float() / 255.0

        logits = self.model(patches.to(self.device))
        probs = logits.sigmoid().cpu().numpy()
        del logits

        probs = probs.squeeze(1)

        valid_mask = probs >= self.config["conf_thr"]
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        return InstanceData(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
        )
