from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import kornia
import numpy as np
import torch

from .feature_extractor import FeatureExtractor

__all__ = (
    "ImageMatcher",
    "KeypointTransformGroup",
)


@dataclass
class KeypointTransformGroup:
    query_kpts: List[cv2.KeyPoint]
    ref_kpts: List[cv2.KeyPoint]
    transform_matrix: np.ndarray | None = None

    def __len__(self):
        return len(self.query_kpts)

    def draw_matches(
        self,
        query_image: np.ndarray,
        ref_image: np.ndarray,
    ) -> None:
        match_img = cv2.drawMatches(
            query_image,
            self.query_kpts,
            ref_image,
            self.ref_kpts,
            [
                cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0)
                for i in range(len(self.query_kpts))
            ],
            None,
            flags=2,
        )

        return match_img

    def warp(
        self,
        input_data,
        output_shape,
    ):
        output_height, output_width = output_shape[:2]
        input_dtype = input_data.dtype

        flags = cv2.INTER_NEAREST if input_dtype == bool else cv2.INTER_CUBIC
        output = cv2.warpPerspective(
            input_data.astype(np.uint8) if input_dtype == bool else input_data,
            self.transform_matrix,
            (output_width, output_height),
            flags=flags,
        )
        return output.astype(bool) if input_dtype == bool else output


def get_transformation(
    query_kpts: List[cv2.KeyPoint],
    ref_kpts: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    use_affine: bool,
    method=cv2.USAC_MAGSAC,
    config: Dict[str, Any] = {
        "reproj_threshold": 0.5,
        "max_iters": 10000,
        "confidence": 0.999,
    },
) -> KeypointTransformGroup:
    if len(query_kpts) < 4 or len(ref_kpts) < 4:
        return KeypointTransformGroup([], [], None)

    transform_estimator = cv2.estimateAffine2D if use_affine else cv2.findHomography
    homography, mask = transform_estimator(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        method=method,
        ransacReprojThreshold=config["reproj_threshold"],
        maxIters=config["max_iters"],
        confidence=config["confidence"],
    )

    if homography is None:
        return KeypointTransformGroup([], [], None)

    matches = np.array(matches)[np.all(mask > 0, axis=1)]
    matches = list(matches)

    if use_affine:
        homography = np.concatenate((homography, np.array([[0, 0, 1.0]])))
    homography = homography.astype(np.float64)

    query_kpts = [query_kpts[m.queryIdx] for m in matches]
    ref_kpts = [ref_kpts[m.queryIdx] for m in matches]

    kpt_transform_group = KeypointTransformGroup(
        query_kpts,
        ref_kpts,
        homography,
    )

    return kpt_transform_group


class ImageMatcher:
    __DEFAULT_CONFIG = {
        "to_downsample": True,
        "num_features": 2048,
        "ransac": {
            "reproj_threshold": 10.0,
            "max_iters": 100000,
            "confidence": 0.999,
        },
        "use_affine": True,
    }

    def __init__(
        self,
        device_str="cpu",
        optconfig: Dict[str, Any] | None = None,
    ):
        self.config = self.__DEFAULT_CONFIG.copy()
        if optconfig is not None:
            self.config.update(optconfig)

        if "cuda" in device_str and not torch.cuda.is_available():
            device_str = "cpu"
        self._device = torch.device(device_str)

        self._light_glue_matcher = (
            kornia.feature.LightGlueMatcher("disk").to(self._device).eval()
        )
        self._feature_extractor = FeatureExtractor(device_str)

    @torch.no_grad()
    def run(
        self,
        query_image: np.ndarray,
        ref_image: np.ndarray,
    ):
        query_lfg = self.extract_features(query_image)
        ref_lfg = self.extract_features(ref_image)

        return self.match_features(
            query_lfg,
            ref_lfg,
            query_image.shape[:2],
            ref_image.shape[:2],
        )

    @torch.no_grad()
    def match_features(
        self,
        query_lfg,
        ref_lfg,
        query_shape,
        ref_shape,
    ):
        _, idxs = self._light_glue_matcher(
            query_lfg.descs,
            ref_lfg.descs,
            self._get_lafs(query_lfg.kpts),
            self._get_lafs(ref_lfg.kpts),
            torch.tensor(query_shape, device=self._device),
            torch.tensor(ref_shape, device=self._device),
        )
        idxs = idxs.detach().cpu()
        query_indices = idxs[:, 0]
        ref_indices = idxs[:, 1]

        if len(query_indices) <= 4:
            return KeypointTransformGroup([], [], None)

        transform_group = get_transformation(
            self.get_cv2_kpts(
                query_lfg.kpts[query_indices],
                query_lfg.responses[query_indices],
            ),
            self.get_cv2_kpts(
                ref_lfg.kpts[ref_indices],
                ref_lfg.responses[ref_indices],
            ),
            matches=[
                cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
                for idx in range(len(query_indices))
            ],
            use_affine=self.config["use_affine"],
            method=cv2.USAC_MAGSAC,
            config=self.config["ransac"],
        )

        return transform_group

    def extract_features(self, image):
        if self.config["to_downsample"]:
            local_feature_group = self._feature_extractor.detect_and_compute_keypoints(
                cv2.resize(image, dsize=None, fx=0.5, fy=0.5),
                num_features=self.config["num_features"],
            )
            local_feature_group._rescale(2)
        else:
            local_feature_group = self._feature_extractor(
                image,
                num_features=self.config["num_features"],
            )

        return local_feature_group

    def _get_lafs(
        self,
        kpts: torch.Tensor,
    ) -> torch.Tensor:
        return kornia.feature.laf_from_center_scale_ori(
            kpts[None],
            96 * torch.ones(1, len(kpts), 1, 1, device=self._device),
        )

    def get_cv2_kpts(
        self,
        kpts: torch.Tensor,
        responses: torch.Tensor,
    ):
        return [
            cv2.KeyPoint(
                x=float(x),
                y=float(y),
                size=0.0,
                response=float(response),
            )
            for (x, y), response in zip(kpts.reshape(-1, 2), responses)
        ]
