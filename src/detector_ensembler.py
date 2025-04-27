from typing import Any, Dict, List

import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.models.utils import weighted_boxes_fusion
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
from mmengine.config import Config, ConfigDict
from mmengine.registry import init_default_scope
from mmengine.structures.instance_data import InstanceData

init_default_scope("mmdet")

__all__ = ("DetectorEnsembler",)


class DetectorEnsembler:
    __DEFAULT_CONFIG = {
        "wbf": {
            "iou_thr": 0.4,
            "conf_type": "max",
        },
    }

    def __init__(
        self,
        config_files: List[str],
        checkpoint_files: List[str],
        conf_thrs: List[float],
        device: str,
        opt_config: Dict[str, Any] | None = None,
    ):
        self.config = self.__DEFAULT_CONFIG.copy()
        if opt_config is not None:
            self.config.update(opt_config)

        def load_cfg(config_file):
            cfg = Config.fromfile(config_file)
            cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

            return cfg

        assert len(conf_thrs) == len(checkpoint_files)
        self.models = [
            init_detector(
                load_cfg(config_file),
                checkpoint_file,
                device=device,
                cfg_options={},
            )
            for config_file, checkpoint_file in zip(config_files, checkpoint_files)
        ]
        self.conf_thrs = conf_thrs

    @torch.no_grad()
    def inference(self, image_path: str):
        results = [
            inference_detector(
                model,
                image_path,
            ).pred_instances
            for model in self.models
        ]

        for idx, result in enumerate(results):
            results[idx] = self.filter_detections_by_confidence(
                result,
                self.conf_thrs[idx],
            )

        results = weighted_boxes_fusion(
            bboxes_list=[result.bboxes for result in results],
            scores_list=[result.scores for result in results],
            labels_list=[result.labels for result in results],
            **self.config["wbf"],
        )
        bboxes, scores, labels = results

        bboxes[:, 0] = bboxes[:, 0].clamp(min=0)
        bboxes[:, 1] = bboxes[:, 1].clamp(min=0)

        return InstanceData(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
        )

    def filter_detections_by_confidence(
        self,
        detections: InstanceData,
        conf_thr: float,
    ) -> InstanceData:
        bboxes = detections.bboxes
        scores = detections.scores
        labels = detections.labels

        valid_mask = scores >= conf_thr
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        return InstanceData(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
        )

    def add_datasample(
        self,
        detections: InstanceData,
        image_path: str,
        pred_score_thr: float = 0.2,
        dataset_meta: Dict[str, Any] | None = None,
        output_path: str = "./output.jpg",
    ):
        visualizer = DetLocalVisualizer()
        if dataset_meta is not None:
            visualizer.dataset_meta = dataset_meta

        visualizer.add_datasample(
            name="detections",
            image=mmcv.imread(image_path, channel_order="rgb"),
            data_sample=DetDataSample(pred_instances=detections),
            show=False,
            pred_score_thr=pred_score_thr,
            out_file=output_path,
        )
