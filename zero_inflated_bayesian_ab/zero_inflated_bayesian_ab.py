import numpy as np
from scipy.stats import beta, bernoulli, normal_inverse_gamma, norm, lognorm
import matplotlib.pyplot as plt


class zero_inflated_bayesian_ab:
    def __init__(
        self,
        data_A,
        data_B,
        pi_A_prior_param_1=1,
        pi_B_prior_param_1=1,
        pi_A_prior_param_2=1,
        pi_B_prior_param_2=1,
        mu_A_prior_param_1=0,
        mu_B_prior_param_1=0,
        mu_A_prior_param_2=100,
        mu_B_prior_param_2=100,
        sigma_A_prior_param_1=1,
        sigma_B_prior_param_1=1,
        sigma_A_prior_param_2=1,
        sigma_B_prior_param_2=1,
    ):
        self.data_A = data_A
        self.data_B = data_B
        self.pi_A_prior_param_1 = pi_A_prior_param_1
        self.pi_B_prior_param_1 = pi_B_prior_param_1
        self.pi_A_prior_param_2 = pi_A_prior_param_2
        self.pi_B_prior_param_2 = pi_B_prior_param_2
        self.mu_A_prior_param_1 = mu_A_prior_param_1
        self.mu_B_prior_param_1 = mu_B_prior_param_1
        self.mu_A_prior_param_2 = mu_A_prior_param_2
        self.mu_B_prior_param_2 = mu_B_prior_param_2
        self.sigma_A_prior_param_1 = sigma_A_prior_param_1
        self.sigma_B_prior_param_1 = sigma_B_prior_param_1
        self.sigma_A_prior_param_2 = sigma_A_prior_param_2
        self.sigma_B_prior_param_2 = sigma_B_prior_param_2
        self.pi_A_samples = None
        self.pi_B_samples = None
        self.mu_A_samples = None
        self.sigma_A_samples = None
        self.mu_B_samples = None
        self.sigma_B_samples = None
        self.Y_A_samples = None
        self.Y_B_samples = None

        n_A = len(data_A)
        n_B = len(data_B)
        n_non_zero_A = np.count_nonzero(data_A > 0)
        n_non_zero_B = np.count_nonzero(data_B > 0)
        log_data_non_zero_A = np.log(data_A[data_A > 0])
        log_data_non_zero_B = np.log(data_B[data_B > 0])

        self.pi_A_posterior_param_1 = self.pi_A_prior_param_1 + n_non_zero_A
        self.pi_B_posterior_param_1 = self.pi_B_prior_param_1 + n_non_zero_B
        self.pi_A_posterior_param_2 = self.pi_A_prior_param_2 + n_A - n_non_zero_A
        self.pi_B_posterior_param_2 = self.pi_B_prior_param_2 + n_B - n_non_zero_B
        self.mu_A_posterior_param_1 = (
            self.mu_A_prior_param_2 * self.mu_A_prior_param_1
            + log_data_non_zero_A.sum()
        ) / (self.mu_A_prior_param_2 + n_non_zero_A)
        self.mu_B_posterior_param_1 = (
            self.mu_B_prior_param_2 * self.mu_B_prior_param_1
            + log_data_non_zero_B.sum()
        ) / (self.mu_B_prior_param_2 + n_non_zero_B)
        self.mu_A_posterior_param_2 = self.mu_A_prior_param_2 + n_non_zero_A
        self.mu_B_posterior_param_2 = self.mu_B_prior_param_2 + n_non_zero_B
        self.sigma_A_posterior_param_1 = self.sigma_A_prior_param_1 + n_non_zero_A / 2
        self.sigma_B_posterior_param_1 = self.sigma_B_prior_param_1 + n_non_zero_B / 2
        self.sigma_A_posterior_param_2 = (
            self.sigma_A_prior_param_2
            + (
                n_non_zero_A * log_data_non_zero_A.var()
                + (
                    self.mu_A_prior_param_2
                    * n_non_zero_A
                    * (log_data_non_zero_A.mean() - self.mu_A_prior_param_1) ** 2
                )
                / (self.mu_A_prior_param_2 + n_non_zero_A)
            )
            / 2
        )
        self.sigma_B_posterior_param_2 = (
            self.sigma_B_prior_param_2
            + (
                n_non_zero_B * log_data_non_zero_B.var()
                + (
                    self.mu_B_prior_param_2
                    * n_non_zero_B
                    * (log_data_non_zero_B.mean() - self.mu_B_prior_param_1) ** 2
                )
                / (self.mu_B_prior_param_2 + n_non_zero_B)
            )
            / 2
        )

    def sampling(self, num_samples=100000):
        self.pi_A_samples = beta(
            self.pi_A_posterior_param_1, self.pi_A_posterior_param_2
        ).rvs(num_samples)
        self.pi_B_samples = beta(
            self.pi_B_posterior_param_1, self.pi_B_posterior_param_2
        ).rvs(num_samples)
        delta_A_samples = bernoulli(self.pi_A_samples).rvs()
        delta_B_samples = bernoulli(self.pi_B_samples).rvs()

        self.mu_A_samples, self.sigma_A_samples = normal_inverse_gamma(
            self.mu_A_posterior_param_1,
            self.mu_A_posterior_param_2,
            self.sigma_A_posterior_param_1,
            self.sigma_A_posterior_param_2,
        ).rvs(num_samples)
        self.mu_B_samples, self.sigma_B_samples = normal_inverse_gamma(
            self.mu_B_posterior_param_1,
            self.mu_B_posterior_param_2,
            self.sigma_B_posterior_param_1,
            self.sigma_B_posterior_param_2,
        ).rvs(num_samples)

        self.tilde_Y_A_samples = np.exp(
            norm(loc=self.mu_A_samples, scale=np.sqrt(self.sigma_A_samples)).rvs()
        )
        self.tilde_Y_B_samples = np.exp(
            norm(loc=self.mu_B_samples, scale=np.sqrt(self.sigma_B_samples)).rvs()
        )
        self.Y_A_samples = delta_A_samples * self.tilde_Y_A_samples
        self.Y_B_samples = delta_B_samples * self.tilde_Y_B_samples

    def summary(self):
        if self.Y_A_samples is None:
            print("Model has not been sampled yet.")
        fig, ax = plt.subplots(3, 1, figsize=(6, 12))

        probability_Y_A_over_Y_B = round(
            100 * sum(self.Y_A_samples > self.Y_B_samples) / len(self.Y_A_samples)
        )
        ax[0].hist(self.Y_A_samples - self.Y_B_samples, bins=100, density=True)
        ax[0].set_title(
            f"Probability of Y_A - Y_B: p(Y_A > Y_B) = {probability_Y_A_over_Y_B}%"
        )

        x = np.arange(0, 1, 0.001)
        ax[1].plot(
            x,
            beta(self.pi_A_posterior_param_1, self.pi_A_posterior_param_2).pdf(x),
            label="group A",
        )
        ax[1].plot(
            x,
            beta(self.pi_B_posterior_param_1, self.pi_B_posterior_param_2).pdf(x),
            label="group B",
        )
        ax[1].set_title("Probability of non-zero")
        ax[1].legend()

        bins = np.logspace(
            -3,
            np.log(max(self.tilde_Y_A_samples.max(), self.tilde_Y_B_samples.max())),
            100,
        )
        ax[2].hist(
            self.tilde_Y_A_samples, label="group A", alpha=0.5, bins=bins, density=True
        )
        ax[2].hist(
            self.tilde_Y_B_samples, label="group B", alpha=0.5, bins=bins, density=True
        )
        ax[2].set_title("Probability of values over 0")
        ax[2].set_xscale("log")
        ax[2].legend()

        plt.show()
