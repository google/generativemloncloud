#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Runs preprocessing of user images and starts a training job for the
# generative model.
#
# Assumes user has Cloud Platform project setup and cloud sdk installed.
# Flags:
#
# Required:
# [-d DATA_DIRECTORY] : specifies directory containing images (jpg / png)
# Optional:
# [-c ] : if set, center-crops images. If not set, randomly crops images.
# [-p PORT] : port to start Tensorboard monitoring.

ORIGINAL_DATA_DIRECTORY=''
DATA_DIR_PRESENT=false
CENTER_CROP='False'
PORT=6006

while getopts 'd::cp:' flag; do
  case "${flag}" in
    d) ORIGINAL_DATA_DIRECTORY="${OPTARG%/}"
       DATA_DIR_PRESENT=true
       ;;
    c) CENTER_CROP='True'
       ;;
    p) PORT="${OPTARG}"
  esac
done

readonly ORIGINAL_DATA_DIRECTORY
readonly DATA_DIR_PRESENT
readonly CENTER_CROP
readonly PORT

if [[ -z "${ORIGINAL_DATA_DIRECTORY}" ]]; then
  echo "Error: -d flag required"
  echo "Usage: [-d DATA_DIRECTORY]"
  echo "Specifies directory containing image files"
  exit 1
fi

readonly PROJECT=$(gcloud config list project --format "value(core.project)")
readonly JOB_ID="generative_${USER}_$(date +%Y%m%d_%H%M%S)"
readonly BUCKET="gs://${PROJECT}"
readonly GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"

echo
echo "Using job id: ${JOB_ID}"
set -e

python build_image_data.py \
  --data_directory "${ORIGINAL_DATA_DIRECTORY}" \
  --output_directory "${BUCKET}/${JOB_ID}/test_output"

gcloud ml-engine jobs submit training "${JOB_ID}" \
  --stream-logs \
  --module-name trainer.task \
  --package-path "../trainer" \
  --staging-bucket "${BUCKET}" \
  --region us-east1 \
  --config "../config.yaml" \
  -- \
  --batch_size 64 \
  --data_dir "${BUCKET}/${JOB_ID}/test_output" \
  --output_path "${BUCKET}/${JOB_ID}/output" \
  --center_crop "${CENTER_CROP}" &

tensorboard \
  --logdir "${BUCKET}/${JOB_ID}/output" \
  --port="${PORT}"

wait && echo "Finished training"
