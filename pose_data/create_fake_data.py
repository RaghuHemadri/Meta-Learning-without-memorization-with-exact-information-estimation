"""Create fake train_data.pkl and val_data.pkl for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS


def create_example():
  x = []
  y = []
  for _ in range(100):
    x.append(np.zeros((128, 128), dtype=np.uint8))
    y.append(np.random.rand(3).tolist())
  return x, y


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train = zip(*[create_example() for _ in range(10)])
  valid = zip(*[create_example() for _ in range(10)])

  with open('train_data.pkl', 'wb') as out:
    pickle.dump(train, out)
  with open('val_data.pkl', 'wb') as out:
    pickle.dump(valid, out)


if __name__ == '__main__':
  app.run(main)
