import argparse
import glob
import json
import os


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--sod4sb_dir", type=str, default="./data/sod4sb")
    parser.add_argument("--fbd_sv_2024_dir", type=str, default="./data/FBD-SV-2024")
    parser.add_argument("--merged_dir", type=str, default="./data/merged_dataset")

    return parser.parse_args()


def safe_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except FileExistsError:
        if os.path.islink(dst):
            os.remove(dst)
            os.symlink(src, dst)
        else:
            raise


def process_sod4sb(
    sod4sb_dir,
    new_train_images_dir,
    new_val_images_dir,
):
    train_annot_json = os.path.join(sod4sb_dir, "annotations/split_train_coco.json")
    val_annot_json = os.path.join(sod4sb_dir, "annotations/split_val_coco.json")

    def process_annot_json(_annot_json):
        with open(_annot_json, "r") as _file:
            _anno = json.load(_file)

        for idx, image in enumerate(_anno["images"]):
            _anno["images"][idx] = {
                k: image[k] for k in ("file_name", "height", "width", "id")
            }

        for idx, annotation in enumerate(_anno["annotations"]):
            _anno["annotations"][idx] = {
                k: annotation[k] for k in ("area", "bbox", "id", "image_id", "iscrowd")
            }
            _anno["annotations"][idx]["category_id"] = 1

        _anno = {k: _anno[k] for k in ("images", "annotations")}
        _anno["categories"] = {"id": 1, "name": "bird"}

        return _anno

    train_anno = process_annot_json(train_annot_json)
    val_anno = process_annot_json(val_annot_json)

    for image in train_anno["images"]:
        file_name = image["file_name"]
        file_path = os.path.abspath(os.path.join(sod4sb_dir, "images", file_name))
        sym_link_path = os.path.join(new_train_images_dir, file_name)
        safe_symlink(file_path, sym_link_path)

    for image in val_anno["images"]:
        file_name = image["file_name"]
        file_path = os.path.abspath(os.path.join(sod4sb_dir, "images", file_name))
        sym_link_path = os.path.join(new_val_images_dir, file_name)
        safe_symlink(file_path, sym_link_path)

    return train_anno, val_anno


def parse_annotation(xml_file):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Basic info
    filename = root.find("filename").text
    path = root.find("path").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    depth = int(root.find("size/depth").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        objects.append(
            {"name": name, "difficult": difficult, "bbox": [xmin, ymin, xmax, ymax]}
        )

    return {
        "filename": filename,
        "path": path,
        "size": {"width": width, "height": height, "depth": depth},
        "objects": objects,
    }


def human_sort(s):
    """Sort list the way humans do"""
    import re

    pattern = r"([0-9]+)"
    return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]


def process_fbd(
    fbd_dir,
    split: str,
    cur_image_id: int,
    cur_bbox_id: int,
    new_image_dir: str,
):
    image_dir = os.path.join(fbd_dir, "images", split)
    annot_dir = os.path.join(fbd_dir, "labels", split)

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_paths.sort(key=human_sort)

    annot_paths = glob.glob(os.path.join(annot_dir, "*.xml"))
    annot_paths.sort(key=human_sort)

    images = []
    annotations = []

    for image_path, annot_path in zip(image_paths, annot_paths):
        assert (
            os.path.basename(image_path).split(".")[0]
            == os.path.basename(annot_path).split(".")[0]
        ), f"{image_path}, {annot_path}"
        file_name = os.path.basename(image_path)

        xml_annotations = parse_annotation(annot_path)

        if len(xml_annotations["objects"]) == 0:
            continue

        image = {
            "file_name": file_name,
            "height": xml_annotations["size"]["height"],
            "width": xml_annotations["size"]["width"],
            "id": cur_image_id,
        }
        images.append(image)
        cur_image_id += 1

        new_image_path = os.path.join(new_image_dir, file_name)
        safe_symlink(
            os.path.abspath(image_path),
            new_image_path,
        )

        for _object in xml_annotations["objects"]:
            xmin, ymin, xmax, ymax = _object["bbox"]
            w, h = xmax - xmin, ymax - ymin
            area = w * h
            anno = {
                "area": area,
                "bbox": [xmin, ymin, w, h],
                "category_id": 1,
                "id": cur_bbox_id,
                "image_id": image["id"],
                "iscrowd": 0,
            }
            cur_bbox_id += 1

            annotations.append(anno)

    split_anno = {
        "images": images,
        "annotations": annotations,
        "categories": {"id": 1, "name": "bird"},
    }

    return split_anno


def main():
    args = get_args()
    assert os.path.isdir(args.sod4sb_dir)
    assert os.path.isdir(args.fbd_sv_2024_dir)

    os.makedirs(args.merged_dir, exist_ok=True)
    new_train_images_dir = os.path.join(args.merged_dir, "train")
    new_val_images_dir = os.path.join(args.merged_dir, "val")
    new_annotations_dir = os.path.join(args.merged_dir, "annotations")
    for _dir in [new_train_images_dir, new_val_images_dir, new_annotations_dir]:
        os.makedirs(_dir, exist_ok=True)

    train_anno, val_anno = process_sod4sb(
        args.sod4sb_dir,
        new_train_images_dir,
        new_val_images_dir,
    )

    train_image_id_max = max([image["id"] for image in train_anno["images"]])
    train_annotation_id_max = max(
        [annotation["id"] for annotation in train_anno["annotations"]]
    )

    val_image_id_max = max([image["id"] for image in val_anno["images"]])
    val_annotation_id_max = max(
        [annotation["id"] for annotation in val_anno["annotations"]]
    )

    print(
        f"sod4sb: train {train_image_id_max}, {train_annotation_id_max}; val {val_image_id_max}, {val_annotation_id_max}"
    )

    fbd_train_anno = process_fbd(
        args.fbd_sv_2024_dir,
        "train",
        train_image_id_max + 1,
        train_annotation_id_max + 1,
        new_train_images_dir,
    )

    fbd_val_anno = process_fbd(
        args.fbd_sv_2024_dir,
        "val",
        val_image_id_max + 1,
        val_annotation_id_max + 1,
        new_val_images_dir,
    )

    merged_train_anno = {
        "categories": train_anno["categories"],
        "images": train_anno["images"] + fbd_train_anno["images"],
        "annotations": train_anno["annotations"] + fbd_train_anno["annotations"],
    }

    merged_val_anno = {
        "categories": val_anno["categories"],
        "images": val_anno["images"] + fbd_val_anno["images"],
        "annotations": val_anno["annotations"] + fbd_val_anno["annotations"],
    }

    os.makedirs(f"{args.merged_dir}/annotations", exist_ok=True)
    with open(f"{args.merged_dir}/annotations/train.json", "w") as f:
        json.dump(merged_train_anno, f)

    with open(f"{args.merged_dir}/annotations/val.json", "w") as f:
        json.dump(merged_val_anno, f)


if __name__ == "__main__":
    main()
