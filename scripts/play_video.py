import argparse
import os

import cv2


def human_sort(s):
    """Sort list the way humans do"""
    import re

    pattern = r"([0-9]+)"
    return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]


def get_args():
    parser = argparse.ArgumentParser("play video")
    parser.add_argument("--video_dir", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    video_dir = args.video_dir

    images = [f for f in os.listdir(video_dir) if f.lower().endswith(".jpg")]
    images.sort(key=human_sort)

    if not images:
        print("No jpg images found in the directory.")
        return

    for img_name in images:
        img_path = os.path.join(video_dir, img_name)
        img = cv2.imread(img_path)
        assert img is not None, f"Failed to read image: {img_path}"
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Video", img)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
