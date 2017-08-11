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
# Creates and deploys model to Cloud ML engine from training job.
#
# Assumes user has Cloud Platform project setup and cloud sdk installed.
# Flags:
#
# Required if -l not set:
# [-j JOB_NAME] : specifies training job from which to create / deploy model.
# Optional:
# [-l ] : if set, lists 10 most recent jobs run by user.

JOB_NAME=''
JOB_NAME_PRESENT=false
LIST_JOBS=false

while getopts 'j:l' flag; do
  case "${flag}" in
    j) JOB_NAME="${OPTARG}"
       JOB_NAME_PRESENT=true
       ;;
    l) LIST_JOBS=true
  esac
done

if [[ -z "${JOB_NAME}" && "${LIST_JOBS}" == false ]]; then
  echo "Error: -j flag required"
  echo "Usage: [-j JOB_NAME]"
  echo "Specifies job name to generate model from"
  exit 1
fi

if [[ "${LIST_JOBS}" == true && -z "${JOB_NAME}" ]]; then
  echo "Most recent jobs"
  gcloud ml-engine jobs list --limit=10
  echo "Exiting...."
  exit 1
fi

readonly PROJECT=$(gcloud config list project --format "value(core.project)")
readonly JOB_ID="${JOB_NAME}"
readonly BUCKET="gs://${PROJECT}"
readonly GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"

readonly EMBED_MODEL_NAME="${JOB_ID}_embed_to_image"
readonly IMAGE_MODEL_NAME="${JOB_ID}_image_to_embed"
readonly VERSION_NAME=v1

OUTPUT=$(gcloud ml-engine jobs describe "${JOB_ID}")

if [[ $(echo "${OUTPUT}" | grep -i 'state: running') ]]; then
  echo
  echo "Training task is running."
  echo "Please wait for task to succeeed before creating model."
  echo "Exiting..."
  exit 1
elif [[ $(echo "${OUTPUT}" | grep -i 'state: failed') ]]; then
  echo
  echo "Training task failed. Please rerun training job."
  echo "Exiting..."
  exit 1
elif [[ $(echo "${OUTPUT}" | grep -i 'state: cancelled') ]]; then
  echo
  echo "Training task cancelled. Please rerun training job."
  echo "Exiting..."
  exit 1
elif [[ $(echo "${OUTPUT}" | grep -i 'state: succeeded') ]]; then
  echo
  echo "Training task succeeded."
  echo "Creating embedding to image model...."
  gcloud ml-engine models create "${EMBED_MODEL_NAME}" \
    --regions us-central1

  echo "Deploying embedding to image model...."

  gcloud ml-engine versions create "${VERSION_NAME}" \
    --model "${EMBED_MODEL_NAME}" \
    --origin "${BUCKET}/${JOB_ID}/output/model/saved_model_embed_in" \
    --runtime-version=1.0

  echo
  echo "Training task succeeded."
  echo "Creating image to embedding model...."
  gcloud ml-engine models create "${IMAGE_MODEL_NAME}" \
    --regions us-central1

  echo "Deploying image to embedding model...."

  gcloud ml-engine versions create "${VERSION_NAME}" \
    --model "${IMAGE_MODEL_NAME}" \
    --origin "${BUCKET}/${JOB_ID}/output/model/saved_model_image_in" \
    --runtime-version=1.0
else
  echo
  echo "Task in unknown state. Please check cloud console."
  echo "Use -l to list 10 most recent jobs"
  echo "Exiting..."
  exit 1
fi

