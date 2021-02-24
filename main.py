# Check that imports for the rest of the file work.

from src import DatasetManagement as Dm
from src.DatasetManagement import load_real_samples, train
from src.Layers import define_discriminator, define_generator, define_composite

directory = 'resources/dataset/'
compressed = 'img_all_creatures_128.npz'

all_creatures = Dm.load_creatures(directory)

# number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
n_blocks = 6
# size of the latent space
latent_dim = 100
# define models
d_models = define_discriminator(n_blocks)
# define models
g_models = define_generator(latent_dim, n_blocks)
# define composite models
gan_models = define_composite(d_models, g_models)
# load image data
dataset = load_real_samples(all_creatures)
print('Loaded', dataset.shape)
# train model
n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)