import tensorflow as tf
import numpy as np

# Constant for the logarithm of 2π, used in entropy calculations.
log_2_pi = np.log(2.0 * np.pi)

def kl_div_loss_analytically_from_logvar_and_precision(mu1, logvar1, mu2, logvar2, omega):
    """
    Calculate KL divergence with precision term, analytically, given mean and log variance.

    Args:
        mu1, logvar1: Mean and log variance of the first distribution.
        mu2, logvar2: Mean and log variance of the second distribution.
        omega: The precision term for the divergence calculation.

    Returns:
        The calculated KL divergence.
    """
    return 0.5 * (logvar2 - tf.math.log(omega) - logvar1) + \
           (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2) / omega) - 0.5

def kl_div_loss_analytically_from_logvar(mu1, logvar1, mu2, logvar2):
    """
    Calculate KL divergence analytically, given mean and log variance, without precision term.

    Args:
        mu1, logvar1: Mean and log variance of the first distribution.
        mu2, logvar2: Mean and log variance of the second distribution.

    Returns:
        The calculated KL divergence.
    """
    return 0.5 * (logvar2 - logvar1) + \
           (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2)) - 0.5

def kl_div_loss(mu1, var1, mu2, var2, axis=1):
    """
    Calculate KL divergence given mean and variance, wrapping the analytical calculation.

    Args:
        mu1, var1: Mean and variance of the first distribution.
        mu2, var2: Mean and variance of the second distribution.
        axis: The dimension over which to sum the divergence.

    Returns:
        Sum of KL divergence over the specified axis.
    """
    return tf.reduce_sum(kl_div_loss_analytically_from_logvar(mu1, tf.math.log(var1), mu2, tf.math.log(var2)), axis)

# Constant for the logarithm of 2πe, used in entropy calculations.
log_2_pi_e = np.log(2.0 * np.pi * np.e)

@tf.function
def entropy_normal_from_logvar(logvar):
    """
    Calculate the entropy of a normal distribution given its log variance.

    Args:
        logvar: The log variance of the distribution.

    Returns:
        The calculated entropy.
    """
    return 0.5 * (log_2_pi_e + logvar)

def entropy_bernoulli(p, displacement=0.00001):
    """
    Calculate the entropy of a Bernoulli distribution given probability p.

    Args:
        p: Probability of success.
        displacement: Small constant to avoid log(0) calculation.

    Returns:
        The calculated entropy.
    """
    return - (1 - p) * tf.math.log(displacement + 1 - p) - p * tf.math.log(displacement + p)

def log_bernoulli(x, p, displacement=0.00001):
    """
    Calculate the log likelihood of a Bernoulli distribution given data x and probability p.

    Args:
        x: The observed data.
        p: Probability of success.
        displacement: Small constant to avoid log(0) calculation.

    Returns:
        The log likelihood of observing x under the given Bernoulli distribution.
    """
    return x * tf.math.log(displacement + p) + (1 - x) * tf.math.log(displacement + 1 - p)

def calc_reward(o, resolution=64):
    """
    Calculate a reward based on a specific pattern (half ones, half zeros) in observed data.

    Args:
        o: The observed data.
        resolution: The resolution of the pattern.

    Returns:
        The calculated reward.
    """
    perfect_reward = np.zeros((3, resolution, 1), dtype=np.float32)
    perfect_reward[:, :int(resolution / 2)] = 1.0
    return log_bernoulli(o[:, 0:3, 0:resolution, :], perfect_reward)

def total_correlation(data):
    """
    Calculate the total correlation of a dataset.

    Args:
        data: The dataset with features as columns.

    Returns:
        The total correlation of the dataset.
    """
    cov = np.cov(data.T)
    return 0.5 * (np.log(np.diag(cov)).sum() - np.linalg.slogdet(cov)[1])
