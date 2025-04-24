from typing import Any, Dict

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
        "conf_thr": 0.25,
        "wbf": {
            "iou_thr": 0.4,
            "conf_type": "max",
        },
    }

    def __init__(
        self,
        config_file: str,
        checkpoint_files: str,
        device: str,
        opt_config: str | None = None,
    ):
        self.config = self.__DEFAULT_CONFIG.copy()
        if opt_config is not None:
            self.config.update(opt_config)

        cfg = Config.fromfile(config_file)
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

        self.models = [
            init_detector(
                cfg,
                checkpoint_file,
                device=device,
                cfg_options={},
            )
            for checkpoint_file in checkpoint_files
        ]

    @torch.no_grad()
    def inference(self, image_path: str):
        results = [
            inference_detector(
                model,
                image_path,
            )
            for model in self.models
        ]

        results = weighted_boxes_fusion(
            bboxes_list=[result.pred_instances.bboxes for result in results],
            scores_list=[result.pred_instances.scores for result in results],
            labels_list=[result.pred_instances.labels for result in results],
            **self.config["wbf"],
        )
        bboxes, scores, labels = results

        valid_mask = scores >= self.config["conf_thr"]
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        bboxes[:, 0] = bboxes[:, 0].clamp(min=0)
        bboxes[:, 1] = bboxes[:, 1].clamp(min=0)

        pred_instances = InstanceData(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
        )

        return DetDataSample(pred_instances=pred_instances)

    def add_datasample(
        self,
        result: DetDataSample,
        image_path: str,
        pred_score_thr: float = 0.2,
        dataset_meta: Dict[str, Any] | None = None,
        output_path: str = "./output.jpg",
    ):
        visualizer = DetLocalVisualizer()
        if dataset_meta is not None:
            visualizer.dataset_meta = dataset_meta

        visualizer.add_datasample(
            name="result",
            image=mmcv.imread(image_path, channel_order="rgb"),
            data_sample=result,
            show=False,
            pred_score_thr=pred_score_thr,
            out_file=output_path,
        )
