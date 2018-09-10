import numpy as np
import tensorflow as tf

def tf_2d_normal(g, x, y, mux, muy, sx, sy, rho):
    '''
    Function that computes a multivariate Gaussian
    Equation taken from 24 & 25 in Graves (2013)
    '''
    with g.as_default():
        # Calculate (x-mux) and (y-muy)
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)
        # Calculate sx*sy
        sxsy = tf.multiply(sx, sy)
        # Calculate the exponential factor
        z = tf.square(tf.divide(normx, sx)) + tf.square(tf.divide(normy, sy)) - 2*tf.divide(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
        negatedRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.divide(-z, 2*negatedRho))
        # Normalization constant
        denominator = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negatedRho))
        # Final PDF calculation
        result = tf.divide(result, denominator)

        return result

def get_mean_error(predicted_traj, true_traj, observed_length):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    taken from: https://github.com/vvanirudh/social-lstm-tf
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)

def sample_2d_normal(o_mux, o_muy, o_sx, o_sy, o_corr):
    '''
    Function that samples from a multivariate Gaussian
    That has the statistics computed by the network.
    '''
    mean = [o_mux[0][0], o_muy[0][0]]
    # Extract covariance matrix
    cov = [[o_sx[0][0]*o_sx[0][0], o_corr[0][0]*o_sx[0][0]*o_sy[0][0]], [o_corr[0][0]*o_sx[0][0]*o_sy[0][0], o_sy[0][0]*o_sy[0][0]]]
    # Sample a point from the multivariate normal distribution
    sampled_x = np.random.multivariate_normal(mean, cov, 1)

    return sampled_x[0][0], sampled_x[0][1]