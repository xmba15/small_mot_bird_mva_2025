#!/bin/bash

dir_list=( $(seq -w 0001 0038) )

for _dir in "${dir_list[@]}"; do
    echo "Run video ${_dir}"
    python3 scripts/track_video.py --video_dir ./data/smot4sb/pub_test/${_dir}
done
