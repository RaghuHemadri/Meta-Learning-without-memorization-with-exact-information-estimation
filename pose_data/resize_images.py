from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import multiprocessing
import os
from absl import app
from absl import flags
from PIL import Image

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None,
                    'Root directory where images are stored.')


def resize(image_file):
  im = Image.open(image_file)
  im = im.resize((128, 128), resample=Image.LANCZOS)
  im = im.convert('L')
  im.save(image_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  all_images = glob.glob(os.path.join(FLAGS.data_dir, '*/*/*.png'))

  p = multiprocessing.Pool(10)
  p.map(resize, all_images)

if __name__ == '__main__':
  app.run(main)
