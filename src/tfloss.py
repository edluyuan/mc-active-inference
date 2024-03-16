import tensorflow as tf
import numpy as np

# Import local utility functions
from src import tfutils


def compute_omega(kl_pi, a, b, c, d):
    """
    Computes the omega value based on the provided parameters.

    Parameters:
        kl_pi (float): The KL divergence value.
        a, b, c, d (float): Coefficients used in the omega calculation.

    Returns:
        float: The computed omega value.
    """
    return a * (1.0 - 1.0 / (1.0 + np.exp(- (kl_pi - b) / c))) + d


@tf.function
def compute_kl_div_pi(model, o0, log_ppi):
    """
    Computes the KL divergence between Q(pi|s1,s0) and P(pi).

    Parameters:
        model (tf.Model): The model containing necessary submodels.
        o0 (tf.Tensor): The input observation.
        log_ppi (tf.Tensor): Log probabilities of policy P(pi).

    Returns:
        tf.Tensor: KL divergence for each element in the batch.
    """
    # Encode observation and sample state
    qs0 = model.model_down.encode_o_and_sample_s(o0)
    _, qpi, log_qpi = model.model_top.encode_s(qs0)
    # Compute KL divergence
    return tf.reduce_sum(qpi * (log_qpi - log_ppi), 1)


@tf.function
def compute_loss_top(model_top, s, log_ppi):
    """
    Computes the loss for the top model component based on KL divergence.

    Parameters:
        model_top (tf.Model): The top component of the model.
        s (tf.Tensor): The state tensor.
        log_ppi (tf.Tensor): Log probabilities of policy P(pi).

    Returns:
        Tuple containing the total loss, KL divergence, and other details.
    """
    _, qpi, log_qpi = model_top.encode_s(s)
    kl_div_pi_anal = qpi * (log_qpi - log_ppi)
    kl_div_pi = tf.reduce_sum(kl_div_pi_anal, 1)

    f_top = kl_div_pi
    return f_top, kl_div_pi, kl_div_pi_anal, qpi


@tf.function
def compute_loss_mid(model_mid, s0, ppi_sampled, qs1_mean, qs1_logvar, omega):
    """
    Computes the middle component loss involving the transition dynamics.

    Parameters:
        model_mid (tf.Model): The middle component of the model.
        s0 (tf.Tensor): The initial state tensor.
        ppi_sampled (tf.Tensor): Sampled policy actions.
        qs1_mean (tf.Tensor): Mean of the encoded state.
        qs1_logvar (tf.Tensor): Log variance of the encoded state.
        omega (float): Weighting factor.

    Returns:
        Tuple containing the total loss, detailed loss terms, and state details.
    """
    # Compute transition and its KL divergence
    ps1, ps1_mean, ps1_logvar = model_mid.transition_with_sample(ppi_sampled, s0)
    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(
        qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = tf.reduce_sum(kl_div_s_anal, 1)

    f_mid = kl_div_s
    loss_terms = (kl_div_s, kl_div_s_anal)
    return f_mid, loss_terms, ps1, ps1_mean, ps1_logvar


@tf.function
def compute_loss_down(model_down, o1, ps1_mean, ps1_logvar, omega, displacement=0.00001):
    """
    Computes the loss for the lower model component, focusing on observation reconstruction.

    Parameters:
        model_down (tf.Model): The lower component of the model.
        o1 (tf.Tensor): The target observation.
        ps1_mean (tf.Tensor): Mean of the predicted next state.
        ps1_logvar (tf.Tensor): Log variance of the predicted next state.
        omega (float): Weighting factor.
        displacement (float): Small value to avoid log(0) in binary cross-entropy.

    Returns:
        Tuple containing the total loss, detailed loss terms, and observation reconstruction.
    """
    # Encode observation, reparameterize, and decode to reconstruct observation
    qs1_mean, qs1_logvar = model_down.encoder(o1)
    qs1 = model_down.reparameterize(qs1_mean, qs1_logvar)
    po1 = model_down.decoder(qs1)

    # Compute binary cross-entropy for reconstructed observation
    bin_cross_entr = o1 * tf.math.log(displacement + po1) + \
                     (1 - o1) * tf.math.log(displacement + 1 - po1)
    logpo1_s1 = tf.reduce_sum(bin_cross_entr, axis=[1, 2, 3])

    # Compute naive and adjusted KL divergences
    kl_div_s_naive_anal = kl_div_loss_analytically_from_logvar_and_precision(
        qs1_mean, qs1_logvar, 0.0, 0.0, omega)
    kl_div_s_naive = tf.reduce_sum(kl_div_s_naive_anal, 1)

    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(
        qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = tf.reduce_sum(kl_div_s_anal, 1)

    # Compute final loss based on model parameters
    if model_down.gamma <= 0.05:
        f = -model_down.beta_o * logpo1_s1 + model_down.beta_s * kl_div_s_naive
    elif model_down.gamma >= 0.95:
        f = -model_down.beta_o * logpo1_s1 + model_down.beta_s * kl_div_s
    else:
        f = -model_down.beta_o * logpo1_s1 + \
            model_down.beta_s * (model_down.gamma * kl_div_s +
                                 (1.0 - model_down.gamma) * kl_div_s_naive)
    loss_terms = (-logpo1_s1, kl_div_s, kl_div_s_anal, kl_div_s_naive, kl_div_s_naive_anal)
    return f, loss_terms, po1, qs1


@tf.function
def train_model_top(model_top, s, log_ppi, optimizer):
    """
    Train the top part of the model using gradient descent.

    Args:
        model_top: The model instance for the top part of the architecture.
        s: The current state inputs to the model.
        log_ppi: The logarithm of policy distribution probabilities.
        optimizer: The optimizer instance to apply gradients.

    Returns:
        The computed KL divergence for the policy distribution.
    """
    # Preventing the computation of gradients for the inputs
    s_stopped = tf.stop_gradient(s)
    log_ppi_stopped = tf.stop_gradient(log_ppi)

    with tf.GradientTape() as tape:
        # Compute the loss and KL divergence
        f, kl_pi, _, _ = compute_loss_top(model_top=model_top, s=s_stopped, log_Ppi=log_ppi_stopped)

        # Calculate gradients of loss with respect to model parameters
        gradients = tape.gradient(f, model_top.trainable_variables)

        # Apply gradients to model parameters
        optimizer.apply_gradients(zip(gradients, model_top.trainable_variables))

    return kl_pi


@tf.function
def train_model_mid(model_mid, s0, qs1_mean, qs1_logvar, ppi_sampled, omega, optimizer):
    """
    Train the middle part of the model using gradient descent.

    Args:
        model_mid: The model instance for the middle part of the architecture.
        s0: Initial state inputs to the model.
        qs1_mean: Mean of the encoded state s1.
        qs1_logvar: Log variance of the encoded state s1.
        ppi_sampled: Sampled policy actions.
        omega: Omega parameter for KL divergence computation.
        optimizer: The optimizer instance to apply gradients.

    Returns:
        Mean and log variance of the posterior state s1.
    """
    # Prevent gradient computation for these variables
    s0_stopped = tf.stop_gradient(s0)
    qs1_mean_stopped = tf.stop_gradient(qs1_mean)
    qs1_logvar_stopped = tf.stop_gradient(qs1_logvar)
    ppi_sampled_stopped = tf.stop_gradient(ppi_sampled)
    omega_stopped = tf.stop_gradient(omega)

    with tf.GradientTape() as tape:
        # Compute the middle part loss and other related terms
        f, loss_terms, ps1, ps1_mean, ps1_logvar = compute_loss_mid(
            model_mid=model_mid, s0=s0_stopped, Ppi_sampled=ppi_sampled_stopped,
            qs1_mean=qs1_mean_stopped, qs1_logvar=qs1_logvar_stopped, omega=omega_stopped)

        # Calculate gradients of loss with respect to model parameters
        gradients = tape.gradient(f, model_mid.trainable_variables)

        # Apply gradients to model parameters
        optimizer.apply_gradients(zip(gradients, model_mid.trainable_variables))

    return ps1_mean, ps1_logvar


@tf.function
def train_model_down(model_down, o1, ps1_mean, ps1_logvar, omega, optimizer):
    """
    Train the bottom part of the model using gradient descent.

    Args:
        model_down: The model instance for the bottom part of the architecture.
        o1: The observed data inputs to the model.
        ps1_mean: Mean of the prior state s1.
        ps1_logvar: Log variance of the prior state s1.
        omega: Omega parameter for KL divergence computation.
        optimizer: The optimizer instance to apply gradients.

    Returns:
        None. The function updates the model's parameters based on the computed gradients.
    """
    # Prevent gradient computation for these variables
    ps1_mean_stopped = tf.stop_gradient(ps1_mean)
    ps1_logvar_stopped = tf.stop_gradient(ps1_logvar)
    omega_stopped = tf.stop_gradient(omega)

    with tf.GradientTape() as tape:
        # Compute the bottom part loss
        f, _, _, _ = compute_loss_down(
            model_down=model_down, o1=o1, ps1_mean=ps1_mean_stopped,
            ps1_logvar=ps1_logvar_stopped, omega=omega_stopped)

        # Calculate gradients of loss with respect to model parameters
        gradients = tape.gradient(f, model_down.trainable_variables)

        # Apply gradients to model parameters
        optimizer.apply_gradients(zip(gradients, model_down.trainable_variables))