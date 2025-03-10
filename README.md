# zero_inflated_bayesian_ab

## Overview

This package provides a Bayesian approach for conducting AB tests on data following a Zero-Inflated Lognormal distribution. It enables users to analyze revenue per user when the data contains a significant number of zeros.

## Model Description

This package assumes that the observed values follow a Zero-Inflated Lognormal distribution. Since both Group A and Group B share the same distributional structure, we describe the model for Group A:

```math
y_A \sim (1-\pi_A)\delta(y_A) + \pi_A \mathcal{LN}(\mu_A, \sigma_A^2)
```

where $\delta(y)$ is the Dirac delta function, and $\mathcal{LN}(\mu, \sigma^2)$ is the probability density function of a lognormal distribution. The prior distributions for the parameters are assumed as follows:

```math
\begin{align*}
\pi_A &\sim \mathcal{B}(\alpha_A, \beta_A) \\
\mu_A &\sim \mathcal{N}(\theta_A, \sigma_A^2 / \lambda_A) \\
\sigma_A^2 &\sim \mathcal{IG}(a_A, b_A)
\end{align*}
```

where $\mathcal{B}(\alpha_A, \beta_A)$ denotes a Beta distribution, $\mathcal{N}(\theta, \sigma^2)$ represents a Normal distribution, and $mathcal{IG}(a, b)$ is an Inverse Gamma distribution.

## Installation

```bash
!pip install git+https://github.com/Kuno3/zero_inflated_bayesian_ab
```

## Usage

```python
from zero_inflated_bayesian_ab.zero_inflated_bayesian_ab import (
    zero_inflated_bayesian_ab,
)

# Example Data (Group A and Group B)
model = zero_inflated_bayesian_ab(data_A, data_B)
# Run Sampling
model.sampling(100000)
# Summary plot
model.summary()
```

## API Reference

### `zero_inflated_bayesian_ab(data_A, data_B, alpha_A,  alpha_B, beta_A, beta_B, theta_A, theta_B, lambda_A, lambda_B, a_A, a_B, b_A, b_B))`

- **data_A**: List or numpy array of observations from group A.
- **data_B**: List or numpy array of observations from group B.
- **, alpha_A, alpha_B, beta_A, beta_B, theta_A, theta_B, lambda_A, lambda_B, a_A, a_B, b_A, b_B**: (Optional) Parameters for priors.
- **Returns**: An object containing posterior estimates and visualization methods.

### `sampling(num_samples)`

- **num_samples**: Number of samples.

### `summary()`

- Generates plots for posterior distributions.

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`

## License

MIT License
