import glob
import os

import cv2
import pandas as pd
from torch.utils.data import Dataset

__all__ = ("BirdClassificationDataset", "CustomSubset")


class BirdClassificationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
    ):
        super().__init__()

        self.data_root = os.path.expanduser(data_root)
        assert os.path.isdir(
            self.data_root
        ), f"{self.data_root} is not a valid directory"
        self.split = split

        self._process_gt()

    def _process_gt(self):
        pos_sample_dir = os.path.join(self.data_root, self.split, "positive")
        pos_image_paths = glob.glob(os.path.join(pos_sample_dir, "*.jpg"))
        pos_labels = [1] * len(pos_image_paths)

        neg_sample_dir = os.path.join(self.data_root, self.split, "negative")
        neg_image_paths = glob.glob(os.path.join(neg_sample_dir, "*.jpg"))
        neg_labels = [0] * len(neg_image_paths)

        image_paths = pos_image_paths + neg_image_paths
        labels = pos_labels + neg_labels

        self.df = pd.DataFrame(
            {
                "image_path": image_paths,
                "label": labels,
            }
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        assert idx < self.__len__()

        return {
            "image_path": self.df["image_path"][idx],
            "label": self.df["label"][idx],
            "image": cv2.imread(self.df["image_path"][idx])[..., [2, 1, 0]],
        }


class CustomSubset(Dataset):
    def __init__(self, subset, transforms):
        self.subset = subset
        self.transforms = transforms

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        out = self.subset[idx]
        out["image"] = self.transforms(image=out["image"])["image"]

        return out
