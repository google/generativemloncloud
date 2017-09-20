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
# ==============================================================================
"""Displays randomly generated image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cStringIO
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--base64_image', type=str, default='')
args, _ = parser.parse_known_args()

im = args.base64_image.replace('_', '/').replace('-', '+')

missing_base64_padding = len(im) % 4
if missing_base64_padding != 0:
  im += ('=' * (4 - missing_base64_padding))

img = Image.open(cStringIO.StringIO(im.decode('base64')))
img.show()
