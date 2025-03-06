import os

import numpy as np
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist


from functools import partial
from numpyro.infer import Predictive, MCMC, NUTS, SVI, Trace_ELBO


from utils import disentangle_uncertainty


def mix_prior(shape1, shape2=None, param_name=None):
    def create_mixed_prior(shape):
        pi = numpyro.sample(f"pi_{param_name}", dist.Beta(1, 1).expand(shape))
        normal = numpyro.sample(f"{param_name}_normal", dist.Normal(
            jnp.zeros(shape), jnp.ones(shape)))
        laplace = numpyro.sample(f"{param_name}_laplace", dist.Laplace(
            jnp.zeros(shape), jnp.ones(shape)))
        return pi * normal + (1 - pi) * laplace

    shape = (shape1, shape2) if shape2 is not None else (shape1,)
    return create_mixed_prior(shape)


class BNN_NUTS(object):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.rng_key, self.rng_key_predict = random.split(random.PRNGKey(666))
        self.hidden_dims = hidden_dims
        self.net = partial(
            self.jax_model, hidden_dims=hidden_dims, output_dims=output_dims)

    # a two-layer bayesian neural network with computational flow

    def jax_model(self, X, Y, hidden_dims, output_dims=1):
        N, input_dims = X.shape

        w1 = mix_prior(input_dims, hidden_dims, param_name="w1")
        b1 = mix_prior(hidden_dims, param_name="b1") / 10.
        assert w1.shape == (input_dims, hidden_dims)
        z1 = jnp.matmul(X, w1) + b1
        z1 = jnn.relu(z1)
        assert z1.shape == (N, hidden_dims)

        # sample final layer of weights and neural network output
        w2 = mix_prior(hidden_dims, output_dims, param_name="w2") / 10.
        b2 = mix_prior(output_dims, param_name="b2") / 10.
        assert w2.shape == (hidden_dims, output_dims)
        z2 = jnp.matmul(z1, w2) + b2
        assert z2.shape == (N, output_dims)

        mean = jnn.sigmoid(z2).squeeze(-1)
        prob = numpyro.deterministic("prob", mean)

        # observe data
        with numpyro.plate("data", N):
            numpyro.sample("Y", dist.Bernoulli(mean), obs=Y)

    def train(self, train_x, train_y):
        kernel = NUTS(self.net)
        mcmc = MCMC(
            kernel,
            num_warmup=200,
            num_samples=100,
            num_chains=4,
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        )
        mcmc.run(self.rng_key, train_x, train_y)
        self.samples = mcmc.get_samples()
        # gelman rubin diagnostics
        # from numpyro.diagnostics import gelman_rubin
        # rhat_values = {k: gelman_rubin(v.reshape(4, -1)) for k, v in self.samples.items()}
        # print(rhat_values)

    def evaluate(self, test_x, num_models=100):
        # obtain predictive posterior of the probability of the bernoulli
        self.predictive_probs = Predictive(
            self.net, self.samples, return_sites=["prob"])
        probs_posterior = self.predictive_probs(
            self.rng_key_predict, test_x, None)["prob"]
        self.save_model("./models/BNN_NUTS.npz")
        predictions = np.array(probs_posterior.T)
        mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(
            predictions)
        return mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty

    def save_model(self, save_path):
        np.savez(save_path, **self.predictive_probs.posterior_samples)


class BNN_SVI(object):
    def __init__(self, input_dims, hidden_dims, output_dims, lr):
        super().__init__()

        self.rng_key, self.rng_key_predict = random.split(random.PRNGKey(666))
        self.hidden_dims = hidden_dims
        self.optimizer = numpyro.optim.Adam(lr)

    def jax_model(self, X, Y, hidden_dims, output_dims=1):
        N, input_dims = X.shape

        w1 = mix_prior(input_dims, hidden_dims, "w1")
        b1 = mix_prior(hidden_dims, param_name="b1") / 10.

        z1 = jnn.relu(jnp.matmul(X, w1) + b1)

        w2 = mix_prior(hidden_dims, output_dims, "w2") / 10.
        b2 = mix_prior(output_dims, param_name="b2") / 10.
        z2 = jnp.matmul(z1, w2) + b2

        mean = jnn.sigmoid(z2).squeeze(-1)
        prob = numpyro.deterministic("prob", mean)
        with numpyro.plate("data", N):
            numpyro.sample("Y", dist.Bernoulli(mean), obs=Y)

    def guide(self, X, Y, hidden_dims, output_dims):
        N, input_dims = X.shape

        def variational_param(shape, param_name, suffix):
            loc = numpyro.param(f"{param_name}_{suffix}_loc", jnp.zeros(shape))
            scale = numpyro.param(f"{param_name}_{suffix}_scale", jnp.ones(
                shape), constraint=dist.constraints.positive)
            return dist.Normal(loc, scale)

        w1_normal = variational_param(
            (input_dims, hidden_dims), "w1", "normal")
        w1_laplace = variational_param(
            (input_dims, hidden_dims), "w1", "laplace")

        b1_normal = variational_param(hidden_dims, "b1", "normal")
        b1_laplace = variational_param(hidden_dims, "b1", "laplace")

        w2_normal = variational_param(
            (hidden_dims, output_dims), "w2", "normal")
        w2_laplace = variational_param(
            (hidden_dims, output_dims), "w2", "laplace")

        b2_normal = variational_param(1, "b2", "normal")
        b2_laplace = variational_param(1, "b2", "laplace")

        pi_w1_param = numpyro.param("pi_w1_param", jnp.full(
            (input_dims, hidden_dims), 0.5), constraint=dist.constraints.unit_interval)
        numpyro.sample("w1_normal", w1_normal)
        numpyro.sample("w1_laplace", w1_laplace)

        pi_b1_param = numpyro.param("pi_b1_param", jnp.full(
            (hidden_dims, ), 0.5), constraint=dist.constraints.unit_interval)
        numpyro.sample("b1_normal", b1_normal)
        numpyro.sample("b1_laplace", b1_laplace)

        pi_w2_param = numpyro.param("pi_w2_param", jnp.full(
            (hidden_dims, output_dims), 0.5), constraint=dist.constraints.unit_interval)
        numpyro.sample("w2_normal", w2_normal)
        numpyro.sample("w2_laplace", w2_laplace)

        pi_b2_param = numpyro.param("pi_b2_param", jnp.full(
            (output_dims, ), 0.5), constraint=dist.constraints.unit_interval)
        numpyro.sample("pi_w1", dist.Delta(pi_w1_param))
        numpyro.sample("pi_b1", dist.Delta(pi_b1_param))
        numpyro.sample("pi_w2", dist.Delta(pi_w2_param))
        numpyro.sample("pi_b2", dist.Delta(pi_b2_param))

        numpyro.sample("b2_normal", b2_normal)
        numpyro.sample("b2_laplace", b2_laplace)

    def train(self, train_x, train_y):
        svi = SVI(self.jax_model, self.guide,
                  self.optimizer, loss=Trace_ELBO())

        num_iterations = 500
        svi_result = svi.run(self.rng_key, num_iterations,
                             train_x, train_y,  self.hidden_dims, 1)
        self.params = svi_result.params

    def evaluate(self, test_x, num_models):

        predictive = Predictive(
            self.guide, params=self.params, num_samples=num_models)
        posterior_samples = predictive(
            self.rng_key_predict, test_x, None, self.hidden_dims, 1)
        self.predictive_probs = Predictive(
            self.jax_model, posterior_samples, params=self.params, return_sites=["prob"])
        probs_posterior = self.predictive_probs(
            self.rng_key_predict, test_x, None, self.hidden_dims, 1)["prob"]
        self.save_model("./models/BNN_SVI.npz")
        predictions = np.array(probs_posterior.T)
        mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(
            predictions)
        return mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty

    def save_model(self, save_path):
        np.savez(save_path, **self.predictive_probs.posterior_samples)
