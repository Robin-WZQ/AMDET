#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

for len in {1..15}
do
  for id in {1..20}
  do
    config_path="/mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Backdoor_Attack/configs/backdoor_analysis_len${len}/default_TPA_${id}.yaml"
    echo "Running model: $config_path"
    python /mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Backdoor_Attack/backdoor_injection_text_on_text_CLIP.py --config "$config_path"
  done
done

