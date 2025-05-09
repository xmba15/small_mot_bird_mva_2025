# 📝 [Small Multi-Object Tracking for Spotting Birds (SMOT4SB) Challenge 2025](https://mva-org.jp/mva2025/index.php?id=challenge)
***

## 🎛  Dependencies
***

- Install conda environment for both training and inference

```bash
conda create -n smot4sb python=3.11
conda activate smot4sb
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_mmdet.txt
```

## Download data ##
***

- [FBD dataset](FBD-SV-2024: Flying Bird Object Detection Dataset in Surveillance Video): This external dataset is used together with S0D4SB to train bird detectors.
```
# figshare link
https://figshare.com/s/1ca0193680f894a65371?file=50113731

# Repos:
https://github.com/Ziwei89/FBD-SV-2024_github.git
https://github.com/Ziwei89/FBOD

# Command
wget https://figshare.com/ndownloader/files/50113731?private_link=1ca0193680f894a65371 -O ./data/FBD-SV-2024.zip
```

- [SOD4SB](MVA2023 Small Object Detection Challenge for Spotting Birds: Dataset, Methods, and Results)
```
https://drive.google.com/drive/u/1/folders/1WnvpWi8C7GHu_OtXu8DHhDhrk3fCUliI

gdown https://drive.google.com/drive/u/1/folders/1cPscYgFrBiuYqmN1U2QXyh4ewWiwxUXw -O  ./data/sod4sb --folder --continue --remaining-ok
```

## How to Run Video tracking ##

- Download trained weights:
   + Download model weights from the [following Google Drive Url](https://drive.google.com/file/d/1vO8pa60JwgoN6DF4QpQ9XVbPX_4e3W8X/view?usp=drive_link)
   + Unzip the downloaded zip files into the following structure:

```
checkpoints/model_weights/
├── bird_classification_mambaout.pth
├── centernet_efficientnet_merged_dataset_best_coco_bbox_mAP_epoch_31.pth
├── centernet_efficientnet_smot4sb_best_coco_bbox_mAP_epoch_17.pth
└── centernet_rexnet_150_merged_dataset_best_coco_bbox_mAP_epoch_36.pth
```

- Run the tracking on each video with the following script:

```
python3 scripts/track_video.py --video_dir /path/to/video/video_name
```

   + The output tracking result for each frame will be saved into predictions/pub_test/{video_name}

## :gem: References ##
***

[1] [MVA2023 - Small Object Detection Challenge for Spotting Birds](https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds)

[2] [FBD-SV-2024: Flying Bird Object Detection Dataset in Surveillance Video]()
