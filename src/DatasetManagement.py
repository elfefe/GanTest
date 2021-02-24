import os
import numpy as np
from PIL import Image
from matplotlib import pyplot
from numpy.matlib import randn
from numpy.random.mtrand import randint
from tensorflow.python.keras import backend

from src.Layers import WeightedSum


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


# load dataset
def load_real_samples(array):
    # convert from ints to floats
    array = array.astype('float32')
    # scale from [0,255] to [-1,1]
    array = (array - 127.5) / 127.5
    return array


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images
    x = dataset[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return x, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = generator.predict(x_input)
    # create class labels
    y = -np.ones((n_samples, 1))
    return x, y


# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # update alpha for all WeightedSum layers when fading in new blocks
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        # prepare real and fake samples
        x_real, y_real = generate_real_samples(dataset, half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model
        d_loss1 = d_model.train_on_batch(x_real, y_real)
        d_loss2 = d_model.train_on_batch(x_fake, y_fake)
        # update the generator via the discriminator's error
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        # summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, d_loss1, d_loss2, g_loss))


# scale images to preferred size
def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = np.resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
    # devise name
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    x, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    x = (x - x.min()) / (x.max() - x.min())
    # plot real images
    square = int(np.sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(x[i])
    # save plot to file
    filename1 = 'plot_%s.png' % name
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%s.h5' % name
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train the generator and discriminator
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    # fit the baseline model
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
    # scale dataset to appropriate size
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print('Scaled Data', scaled_data.shape)
    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
    summarize_performance('tuned', g_normal, latent_dim)
    # process each level of growth
    for i in range(1, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print('Scaled Data', scaled_data.shape)
        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
        summarize_performance('faded', g_fadein, latent_dim)
        # train normal or straight-through models
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
        summarize_performance('tuned', g_normal, latent_dim)
