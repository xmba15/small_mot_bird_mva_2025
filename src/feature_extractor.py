from dataclasses import dataclass

import kornia
import numpy as np
import torch

__all__ = (
    "LocalFeatureGroup",
    "FeatureExtractor",
)


@dataclass
class LocalFeatureGroup:
    kpts: torch.Tensor
    descs: torch.Tensor
    responses: torch.Tensor

    def _rescale(self, scale: float):
        self.kpts *= scale

    def __len__(self):
        return len(self.kpts)


class FeatureExtractor:
    def __init__(
        self,
        device_str: str = "cpu",
    ):
        if "cuda" in device_str and not torch.cuda.is_available():
            device_str = "cpu"
        self._device = torch.device(device_str)
        self._disk_detector = (
            kornia.feature.DISK.from_pretrained("depth").to(self._device).eval()
        )

    def _get_model_input(self, image: np.ndarray):
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self._device)
        return image_tensor

    @torch.no_grad()
    def detect_and_compute_keypoints(
        self,
        image,
        num_features: int,
    ) -> LocalFeatureGroup:
        features_list = self._disk_detector(
            self._get_model_input(image),
            num_features,
            pad_if_not_divisible=True,
        )[0]
        return LocalFeatureGroup(
            features_list.keypoints,
            features_list.descriptors,
            features_list.detection_scores,
        )
