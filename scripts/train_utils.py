import albumentations as alb
from albumentations.pytorch import ToTensorV2


def get_classification_transforms(hparams):
    image_size = hparams["augmentation"]["image_size"]

    return {
        "train": alb.Compose(
            [
                alb.Resize(
                    height=image_size,
                    width=image_size,
                ),
                alb.HorizontalFlip(p=0.5),
                alb.AdvancedBlur(p=0.5),
                alb.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                alb.ToFloat(max_value=255),
                ToTensorV2(),
            ],
        ),
        "val": alb.Compose(
            [
                alb.Resize(
                    height=image_size,
                    width=image_size,
                ),
                alb.ToFloat(max_value=255),
                ToTensorV2(),
            ],
        ),
    }
