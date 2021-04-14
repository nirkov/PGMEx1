"""
67800 - Probabilistic Methods in AI
Spring 2021
Programming Assignment 1 - Bayesian Networks
(Complete the missing parts (TODO))
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
from scipy.special import logsumexp


def get_p_z1(z1_val):
    """
    Get the prior probability for variable Z1 to take value z1_val.
    """
    return bayes_net['prior_z1'][z1_val]


def get_p_z2(z2_val):
    """
    Get the prior probability for variable Z2 to take value z2_val.
    """
    return bayes_net['prior_z2'][z2_val]


def get_p_x_cond_z1_z2(z1_val, z2_val):
    """
    Get the conditional probabilities of variables X_1 to X_784 to take the value 1 given z1 = z1_val and z2 = z2_val.
    """
    return bayes_net['cond_likelihood'][(z1_val, z2_val)]


def get_pixels_sampled_from_p_x_joint_z1_z2():
    z1_probability = []
    z2_probability = []

    for z1, z2 in zip(z1_vals, z2_vals):
        z1_probability.append(get_p_z1(z1))
        z2_probability.append(get_p_z2(z2))

    pixels_probability = get_p_x_cond_z1_z2(np.random.choice(z1_vals, p=z1_probability),
                                            np.random.choice(z2_vals, p=z2_probability))

    pixels_probability = np.squeeze(pixels_probability)
    image_pixels = []
    for pixel_probability in pixels_probability:
        image_pixels.append(np.random.choice([0, 1], p=[1.-pixel_probability, pixel_probability]))

    return np.array(image_pixels)

def get_expectation_x_cond_z1_z2(z1_val, z2_val):
    return get_p_x_cond_z1_z2(z1_val, z2_val)



def get_conditional_expectation(data):
    """
    TODO. Calculate the conditional expectation E((z1, z2) | X = data[i]) for each data point
    :param data: Row vectors of data points X (n x 784)
    :return: array of E(z1 | X = data), array of E(z2 | X = data)
    """
    pz1_pz2 = np.zeros(25 * 25)
    p_x_1 = np.zeros((25 * 25, len(data[0])))
    p_x_0 = np.zeros((25 * 25, len(data[0])))
    z_val = np.zeros(25 * 25)

    i = 0
    for z1 in z1_vals:
        for z2 in z2_vals:
            pz1_pz2[i] = np.log(get_p_z1(z1)) + np.log(get_p_z2(z2))
            p_x_1[i] = np.log(get_p_x_cond_z1_z2(z1, z2))
            p_x_0[i] = np.log(1 - get_p_x_cond_z1_z2(z1, z2))
            z_val[i] = z1
            i = i + 1

    mean_z1 = []
    for x in data:
        denominator = pz1_pz2 + np.sum(np.where(x == 1, p_x_1, p_x_0), axis=1)
        numerator = z_val * np.exp(denominator)
        mean_z1.append(np.sum(numerator) / np.exp(logsumexp(denominator)))

    i = 0
    for z2 in z2_vals:
        for z1 in z1_vals:
            pz1_pz2[i] = np.log(get_p_z1(z1)) + np.log(get_p_z2(z2))
            p_x_1[i] = np.log(get_p_x_cond_z1_z2(z1, z2))
            p_x_0[i] = np.log(1 - get_p_x_cond_z1_z2(z1, z2))
            z_val[i] = z2
            i = i + 1

    mean_z2 = []
    for x in data:
        denominator = pz1_pz2 + np.sum(np.where(x == 1, p_x_1, p_x_0), axis=1)
        numerator = z_val * np.exp(denominator)
        mean_z2.append(np.sum(numerator) / np.exp(logsumexp(denominator)))


    return mean_z1, mean_z2


# Below are two functions it is suggested you implement and use (but feel free to implement something else).

def get_log_p_x(data):
    """
    TODO. Compute the marginal log likelihood: log P(X)
    :param data: Row vectors of data points X (n x 784)
    :return: Array of log-likelihood values
    """
    pass


def get_log_p_x_joint_z1_z2(data, z1_val, z2_val):
    """
    TODO. Compute the joint log probability log P(X, z1, z2)
    :param data: Row vectors of data points X (n x 784)
    :param z1_val: z1 value (scalar)
    :param z2_val: z2 value (scalar)
    :return: Array of log probability values
    """
    pass


def q_1():
    """
    Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
    Your job is to implement get_pixels_sampled_from_p_x_joint_z1_z2.
    """
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2().reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('q_4', bbox_inches='tight')
    plt.show()
    plt.close()


def q_2():
    """
    Plots the expected images for each latent configuration on a 2D grid.
    Your job is to implement get_expectation_x_cond_z1_z2.
    """

    canvas = np.empty((28*len(z1_vals), 28*len(z1_vals)))
    for i, z1_val in enumerate(z1_vals):
        for j, z2_val in enumerate(z2_vals):
            canvas[(len(z1_vals)-i-1)*28:(len(z1_vals)-i)*28, j*28:(j+1)*28] = \
                get_expectation_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

    plt.figure()
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.tight_layout()
    plt.savefig('q_2', bbox_inches='tight')
    plt.show()
    plt.close()


def q_3():
    """
    Loads the data and plots the histograms. Rest is TODO.
    Your job is to compute real_marginal_log_likelihood and corrupt_marginal_log_likelihood below.
    """

    mat = loadmat('q_3.mat')
    val_data = mat['val_x']
    test_data = mat['test_x']

    z1_z2 = np.zeros(25 * 25)
    p_x_1 = np.zeros((25 * 25, len(val_data[0])))
    p_x_0 = np.zeros((25 * 25, len(val_data[0])))

    loglike = []
    i = 0
    for z1 in z1_vals:
        for z2 in z2_vals:
            z1_z2[i] = np.log(get_p_z1(z1)) + np.log(get_p_z2(z2))
            p_x_1[i] = np.log(get_p_x_cond_z1_z2(z1, z2))
            p_x_0[i] = np.log(1 - get_p_x_cond_z1_z2(z1, z2))
            i = i + 1

    for x in val_data:
        product_log = np.sum(np.where(x == 1, p_x_1, p_x_0), axis=1) + z1_z2
        loglike.append(logsumexp(product_log))

    loglike = np.array(loglike)
    std = np.std(loglike)
    mean = np.mean(loglike)

    loglike_test = []
    for x in test_data:
        product_log = np.sum(np.where(x == 1, p_x_1, p_x_0), axis=1) + z1_z2
        loglike_test.append(logsumexp(product_log))


    loglike_test = np.array(loglike_test)
    real_images_indices = np.abs(loglike_test - mean) < (3 * std)
    real_marginal_log_likelihood = loglike_test[real_images_indices]
    corrupt_marginal_log_likelihood = loglike_test[np.logical_not(real_images_indices)]

    plot_histogram(real_marginal_log_likelihood, title='Histogram of marginal log-likelihood for real test data',
             xlabel='marginal log-likelihood', savefile='q_3_hist_real')

    plot_histogram(corrupt_marginal_log_likelihood, title='Histogram of marginal log-likelihood for corrupted test data',
        xlabel='marginal log-likelihood', savefile='q_3_hist_corrupt')

    plt.show()
    plt.close()


def q_4():
    """
    Loads the data and plots a color coded clustering of the conditional expectations.
    Your job is to implement the get_conditional_expectation function
    """

    mat = loadmat('q_4.mat')
    data = mat['x']
    labels = mat['y']

    mean_z1, mean_z2 = get_conditional_expectation(data)

    plt.figure()
    plt.scatter(mean_z2, mean_z1, c=np.squeeze(labels))
    plt.colorbar()
    plt.grid()
    plt.savefig('q_4', bbox_inches='tight')
    plt.show()
    plt.close()


def load_model(model_file):
    """
    Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
    """

    with open(model_file+'.pkl', 'rb') as infile:
        cpts = pkl.load(infile, encoding='bytes')

    model = {}
    model['prior_z1'] = cpts[0]
    model['prior_z2'] = cpts[1]
    model['cond_likelihood'] = cpts[2]

    return model


def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
    """
    Plots a histogram.
    """

    plt.figure()
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')


def main():
    global bayes_net, z1_vals, z2_vals
    bayes_net = load_model('trained_mnist_model')
    z1_vals = sorted(bayes_net['prior_z1'].keys())
    z2_vals = sorted(bayes_net['prior_z2'].keys())


    # TODO: Using the above Bayesian Network model, complete the following parts.
    # q_1()
    q_2()
    # q_3()
    q_4()

if __name__== '__main__':
    main()
