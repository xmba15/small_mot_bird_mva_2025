import argparse
import functools
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
import yaml
from boxmot import create_tracker
from boxmot.trackers.ocsort.ocsort import convert_x_to_bbox
from boxmot.utils.ops import xyxy2xysr
from loguru import logger
from mmengine.structures.instance_data import InstanceData

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.classifier import Classifier
from src.detector_ensembler import DetectorEnsembler
from src.image_matcher import ImageMatcher


def human_sort(s):
    """Sort list the way humans do"""
    import re

    pattern = r"([0-9]+)"
    return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]


def transform_bbox_xyxy(bbox_xyxy, transform_matrix, score=None):
    assert len(bbox_xyxy) == 4

    dx = transform_matrix[0, 2]
    dy = transform_matrix[1, 2]

    x1, y1, x2, y2 = bbox_xyxy
    x1_new = x1 + dx
    y1_new = y1 + dy
    x2_new = x2 + dx
    y2_new = y2 + dy

    out = [x1_new, y1_new, x2_new, y2_new]
    if score is not None:
        out.append(score)

    return np.array(out, dtype=np.float32)


@dataclass
class VideoInfo:
    video_dir: str

    def __post_init__(self):
        assert os.path.isdir(self.video_dir)

        frame_paths = glob.glob(os.path.join(self.video_dir, "*.jpg"))
        assert len(frame_paths) > 0

        frame_paths = sorted(frame_paths, key=human_sort)

        self.frame_paths = frame_paths

        logger.info(f"{self.video_dir} has {len(self.frame_paths)} frames")

    @functools.cached_property
    def video_name(self):
        return os.path.basename(self.video_dir.rstrip("/"))


def get_args():
    parser = argparse.ArgumentParser("track video")
    parser.add_argument(
        "--video_dir", type=str, default="./data/smot4sb/pub_test/0001/"
    )
    parser.add_argument(
        "--config_path", type=str, default="./config/base_tracking.yaml"
    )
    parser.add_argument("--to_show", action="store_true")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=4.0)
    parser.add_argument("--debug_dir", type=str, default="./tmp")
    parser.add_argument("--out_dir", type=str, default="./predictions/pub_test/")

    return parser.parse_args()


def flatten_detections(detections: InstanceData):
    dets = []
    bboxes = detections.bboxes
    scores = detections.scores
    labels = detections.labels

    for bbox, score, label in zip(bboxes, scores, labels):
        dets.append([*bbox, score, label])

    dets = np.array(dets)

    return dets


def main():
    args = get_args()
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    video_info = VideoInfo(
        video_dir=args.video_dir,
    )

    debug_dir = os.path.join(args.debug_dir, video_info.video_name)
    os.makedirs(debug_dir, exist_ok=True)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    ensembler = DetectorEnsembler(
        hparams["detector"]["mm_config_paths"],
        hparams["detector"]["checkpoints"],
        conf_thrs=hparams["detector"]["conf_thrs"],
        device=hparams["device"],
    )

    classifier = Classifier(
        hparams["classifier"]["encoder_name"],
        hparams["classifier"]["checkpoint"],
        device=hparams["device"],
        opt_config={
            "conf_thr": hparams["classifier"]["conf_thr"],
        },
    )
    image_matcher = ImageMatcher(device_str="cuda")

    device = torch.device(hparams["device"])

    tracker = create_tracker(
        tracker_type=hparams["tracker"]["tracker_type"],
        tracker_config=hparams["tracker"]["tracker_config_path"],
        device=device,
        reid_weights=Path(hparams["tracker"]["reid_weights"]),
    )

    track_results = []
    prev_transform_matrix = np.eye(3, 3, dtype=np.float32)
    for idx, frame_path in enumerate(tqdm.tqdm(video_info.frame_paths)):
        detections = ensembler.inference(
            frame_path,
        )

        detections = classifier.filter_by_fine_classification(
            detections,
            frame_path,
        )

        flattened_dets = flatten_detections(detections)

        frame = cv2.imread(frame_path)

        if idx > 0 and hparams["tracker"]["tracker_type"] == "ocsort":
            prev_frame_path = video_info.frame_paths[idx - 1]
            prev_frame = cv2.imread(prev_frame_path)
            kpt_transform_group = image_matcher.run(
                prev_frame,
                frame,
            )

            if kpt_transform_group.transform_matrix is not None:
                transform_matrix = kpt_transform_group.transform_matrix.copy()
            else:
                transform_matrix = prev_transform_matrix.copy()

            prev_transform_matrix = transform_matrix.copy()
            transform_matrix = transform_matrix[:2, :]

            for tk_idx, active_track in enumerate(tracker.active_tracks):
                bbox_xyxy = active_track.get_state()[0]
                conf = active_track.conf
                transformed_bbox_xyxy = transform_bbox_xyxy(
                    bbox_xyxy,
                    transform_matrix,
                    conf,
                )
                m = transform_matrix[:, :2]
                t = transform_matrix[:, 2].reshape(2, 1)

                tracker.active_tracks[tk_idx].last_observation = transformed_bbox_xyxy

                for dt in range(active_track.delta_t, -1, -1):
                    if active_track.age - dt in active_track.observations:
                        ps = (
                            active_track.observations[active_track.age - dt][:4]
                            .reshape(2, 2)
                            .T
                        )
                        ps = m @ ps + t
                        tracker.active_tracks[tk_idx].observations[
                            active_track.age - dt
                        ][:4] = ps.T.reshape(-1)

                tracker.active_tracks[tk_idx].kf.apply_affine_correction(m, t)

        online_targets = tracker.update(
            flattened_dets,
            frame,
        )

        frame_id = idx + 1
        for t in online_targets:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            tid = int(t[4])

            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if vertical:
                continue

            track_results.append(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1,1,1\n"
            )

        frame = tracker.plot_results(frame, show_trajectories=True)

        if args.to_show:
            cv2.imshow(
                "BoXMOT",
                cv2.resize(frame, None, fx=0.5, fy=0.5),
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(debug_dir, frame_name)
        ensembler.add_datasample(
            detections,
            frame_path,
            min(hparams["detector"]["conf_thrs"]),
            dataset_meta={
                "classes": ("bird",),
            },
            output_path=output_path,
        )

    res_file = os.path.join(args.out_dir, f"{video_info.video_name}.txt")
    with open(res_file, "w") as _file:
        _file.writelines(track_results)

    logger.info(f"save results to {res_file}")


if __name__ == "__main__":
    main()
