import argparse
import json
import os
import os.path as osp

from sklearn.model_selection import KFold


def filter_by_video_ids(anno, target_video_names):
    video_ids = {
        video["id"] for video in anno["videos"] if video["name"] in target_video_names
    }
    images = [img for img in anno["images"] if img["video_id"] in video_ids]
    image_ids = {img["id"] for img in images}
    annotations = [ann for ann in anno["annotations"] if ann["image_id"] in image_ids]
    videos = [video for video in anno["videos"] if video["id"] in video_ids]

    return {
        "images": images,
        "annotations": annotations,
        "videos": videos,
        "categories": anno["categories"],
    }


def safe_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except FileExistsError:
        if os.path.islink(dst):
            os.remove(dst)
            os.symlink(src, dst)
        else:
            raise


def split_train_dir(
    train_video_names,
    val_video_names,
    src_dir,
    dst_dir,
):
    os.makedirs(f"{dst_dir}/train", exist_ok=True)
    os.makedirs(f"{dst_dir}/val", exist_ok=True)

    for video_name in train_video_names:
        safe_symlink(
            osp.abspath(f"{src_dir}/train/{video_name}"),
            f"{dst_dir}/train/{video_name}",
        )

    for video_name in val_video_names:
        safe_symlink(
            osp.abspath(f"{src_dir}/train/{video_name}"),
            f"{dst_dir}/val/{video_name}",
        )


def setup_train_val_split(
    indices,
    num_splits,
    fold_th,
    seed,
):
    kf = KFold(
        n_splits=num_splits,
        shuffle=True,
        random_state=seed,
    )

    train_indices, val_indices = list(
        kf.split(
            indices,
        )
    )[fold_th]

    return train_indices, val_indices


def get_args():
    parser = argparse.ArgumentParser("preprocess dataset")
    parser.add_argument(
        "--annot_json_path", type=str, default="./data/smot4sb/annotations/train.json"
    )
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--fold_th", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--src_dir", type=str, default="./data/smot4sb")
    parser.add_argument("--dst_dir", type=str, default="./data/processed_smot4sb")

    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.dst_dir, exist_ok=True)

    with open(args.annot_json_path, "r") as _file:
        train_anno = json.load(_file)

    videos = train_anno["videos"]
    train_indices, val_indices = setup_train_val_split(
        range(len(videos)),
        args.num_splits,
        args.fold_th,
        args.seed,
    )

    train_video_names = [videos[idx]["name"] for idx in train_indices]
    val_video_names = [videos[idx]["name"] for idx in val_indices]

    splitted_train_anno = filter_by_video_ids(train_anno, train_video_names)
    splitted_val_anno = filter_by_video_ids(train_anno, val_video_names)

    dst_dir = os.path.join(args.dst_dir, f"fold_{args.fold_th}")
    os.makedirs(dst_dir, exist_ok=True)
    split_train_dir(
        train_video_names,
        val_video_names,
        args.src_dir,
        dst_dir,
    )

    os.makedirs(f"{dst_dir}/annotations", exist_ok=True)
    with open(f"{dst_dir}/annotations/train.json", "w") as f:
        json.dump(splitted_train_anno, f)
    with open(f"{dst_dir}/annotations/val.json", "w") as f:
        json.dump(splitted_val_anno, f)


if __name__ == "__main__":
    main()
