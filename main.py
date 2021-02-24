# Check that imports for the rest of the file work.

import os
import numpy as np
from PIL import Image
from matplotlib import pyplot


def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    return pixels


def load_creatures(directory):
    faces = [load_image(directory + filename) for filename in os.listdir(directory)]
    print('Creatures: ', len(faces))
    return faces


def plot_creatures(creatures, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(creatures[i].astype('uint8'))
    pyplot.show()


directoryName = 'resources/dataset/'
compressedName = 'img_all_creatures_128.npz'

all_creatures = load_creatures(directoryName)

plot_creatures(all_creatures, 10)
