import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import tqdm
import yaml
from loguru import logger
from mmdet.evaluation.functional import bbox_overlaps

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.detector_ensembler import DetectorEnsembler


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/create_bird_classification_dataset.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    return parser.parse_args()


def bbox_xywh_to_xyxy(bboxes_xywh):
    bboxes_xyxy = bboxes_xywh.copy()
    bboxes_xyxy[:, 2] += bboxes_xyxy[:, 0]  # xmax = x + w
    bboxes_xyxy[:, 3] += bboxes_xyxy[:, 1]  # ymax = y + h
    return bboxes_xyxy


def process_split(
    data_root: str,
    annotation_json_path: str,
    out_dir: str,
    ensembler: DetectorEnsembler,
):
    os.makedirs(out_dir, exist_ok=True)
    pos_sample_dir = os.path.join(out_dir, "positive")
    neg_sample_dir = os.path.join(out_dir, "negative")
    for _dir in [pos_sample_dir, neg_sample_dir]:
        os.makedirs(_dir, exist_ok=True)

    with open(annotation_json_path, "r") as _file:
        data = json.load(_file)

    annotations_by_image_id = defaultdict(list)
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        annotations_by_image_id[image_id].append(bbox)

    for image_id in annotations_by_image_id:
        annotations_by_image_id[image_id] = bbox_xywh_to_xyxy(
            np.array(annotations_by_image_id[image_id], dtype=np.float32)
        )

    image_name_by_id = {}
    for img in data["images"]:
        image_id = img["id"]
        file_name = img["file_name"]
        image_name_by_id[image_id] = file_name

    for image_id in tqdm.tqdm(image_name_by_id):
        file_name = image_name_by_id[image_id]
        file_name_base = os.path.basename(file_name).split(".")[0]

        file_path = os.path.join(data_root, file_name)
        image = cv2.imread(file_path)
        assert image is not None
        bboxes = annotations_by_image_id[image_id]

        if len(bboxes) == 0:
            logger.warning(f"{file_path} has no gt bboxes")
            continue

        detected_data_sample = ensembler.inference(file_path)
        detected_bboxes = detected_data_sample.pred_instances.bboxes.numpy()

        assert isinstance(bboxes, np.ndarray), bboxes

        ious = bbox_overlaps(detected_bboxes, bboxes)

        has_overlap = (ious > 0).any(axis=1)
        negative_bboxes = detected_bboxes[~has_overlap]

        for bbox_id, bbox in enumerate(negative_bboxes):
            xmin, ymin, xmax, ymax = map(int, bbox)
            window = image[ymin:ymax, xmin:xmax]
            window_out_path = os.path.join(
                neg_sample_dir, f"{file_name_base}_{bbox_id}.jpg"
            )
            try:
                cv2.imwrite(window_out_path, window)
            except Exception as e:
                logger.exception(f"Failed to write neg bbox {file_path} {bbox}")

        for bbox_id, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = map(int, bbox)
            window = image[ymin:ymax, xmin:xmax]
            window_out_path = os.path.join(
                pos_sample_dir, f"{file_name_base}_{bbox_id}.jpg"
            )
            cv2.imwrite(window_out_path, window)


def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    ensembler = DetectorEnsembler(
        hparams["model"]["mm_config_path"],
        hparams["model"]["checkpoints"],
        args.device,
    )

    process_split(
        **hparams["dataset"]["train"],
        ensembler=ensembler,
    )

    process_split(
        **hparams["dataset"]["val"],
        ensembler=ensembler,
    )


if __name__ == "__main__":
    main()
