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
# Generates and Displays Random Image.
#
# Assumes user has Cloud Platform project setup and cloud sdk installed.
# Flags:
#
# Required if -l not set:
# [-n MODEL_NAME] : specifies model to generate image.
# Optional:
# [-l ] : if set, lists all models associated with user.
# [-d TEMP_DIR] : directory to which to write json file.

MODEL_NAME=''
LIST_MODELS=false
TMP_DIR='/tmp'

while getopts 'm:ld:' flag; do
  case "${flag}" in
    m) MODEL_NAME="${OPTARG}"
       ;;
    l) LIST_JOBS=true
       ;;
    d) TMP_DIR="${OPTARG%/}"
  esac
done

TMP_FILE="${TMP_DIR}/temp.json"

if [[ -z "${MODEL_NAME}" && "${LIST_MODELS}" == false ]]; then
  echo "Error: -m flag required"
  echo "Usage: [-m MODEL_NAME]"
  echo "Specifies job name to generate model from"
  echo "If model name unknown, use -l flag to list model names"
  exit 1
fi

if [[ "${LIST_MODELS}" == true && -z "${MODEL_NAME}" ]]; then
  echo "Your models"
  gcloud ml-engine models list
  echo "Exiting...."
  exit 1
fi

JSON_OBJ=$(python create_random_embedding.py)

echo -e "${JSON_OBJ}" > "${TMP_FILE}"

OUTPUT=$(gcloud ml-engine predict --model "${MODEL_NAME}" --json-instances "${TMP_FILE}")
IMAGE=$(echo "${OUTPUT}"| awk 'NR==2 {print $2}')
python display_image.py --base64_image "${IMAGE}"
