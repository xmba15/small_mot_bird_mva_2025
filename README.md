# 📝 lib name
***

## :tada: TODO
***

- [x] a
- [ ] b

## 🎛  Dependencies
***

```bash
conda create -n smot4sb python=3.11
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_mmdet.txt

git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# fix requirements/build.txt and mmyolo/__init__.py to accept mmcv<2.2.0
mim install -v -e .
```

## Download data ##
***

- [FBD dataset](FBD-SV-2024: Flying Bird Object Detection Dataset in Surveillance Video)
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

## :gem: References ##
***

[1] [MVA2023 - Small Object Detection Challenge for Spotting Birds](https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds)

[2] [FBD-SV-2024: Flying Bird Object Detection Dataset in Surveillance Video]()
