import numpy as np
from scipy.stats import beta, bernoulli, normal_inverse_gamma, norm
import matplotlib.pyplot as plt


class zero_inflated_bayesian_ab:
    def __init__(
        self,
        data_A,
        data_B,
        alpha_A=1,
        alpha_B=1,
        beta_A=1,
        beta_B=1,
        theta_A=0,
        theta_B=0,
        lambda_A=100,
        lambda_B=100,
        a_A=1,
        a_B=1,
        b_A=1,
        b_B=1,
    ):
        self.data_A = data_A
        self.data_B = data_B
        self.alpha_A = alpha_A
        self.alpha_B = alpha_B
        self.beta_A = beta_A
        self.beta_B = beta_B
        self.theta_A = theta_A
        self.theta_B = theta_B
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.a_A = a_A
        self.a_B = a_B
        self.b_A = b_A
        self.b_B = b_B
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

        self.posterior_alpha_A = self.alpha_A + n_non_zero_A
        self.posterior_alpha_B = self.alpha_B + n_non_zero_B
        self.posterior_beta_A = self.beta_A + n_A - n_non_zero_A
        self.posterior_beta_B = self.beta_B + n_B - n_non_zero_B
        self.posterior_theta_A = (
            self.lambda_A * self.theta_A + log_data_non_zero_A.sum()
        ) / (self.lambda_A + n_non_zero_A)
        self.posterior_theta_B = (
            self.lambda_B * self.theta_B + log_data_non_zero_B.sum()
        ) / (self.lambda_B + n_non_zero_B)
        self.posterior_lambda_A = self.lambda_A + n_non_zero_A
        self.posterior_lambda_B = self.lambda_B + n_non_zero_B
        self.posterior_a_A = self.a_A + n_non_zero_A / 2
        self.posterior_a_A = self.a_B + n_non_zero_B / 2
        self.posterior_b_A = (
            self.b_A
            + (
                n_non_zero_A * log_data_non_zero_A.var()
                + (
                    self.lambda_A
                    * n_non_zero_A
                    * (log_data_non_zero_A.mean() - self.theta_A) ** 2
                )
                / (self.lambda_A + n_non_zero_A)
            )
            / 2
        )
        self.posterior_a_B = (
            self.b_B
            + (
                n_non_zero_B * log_data_non_zero_B.var()
                + (
                    self.lambda_B
                    * n_non_zero_B
                    * (log_data_non_zero_B.mean() - self.theta_B) ** 2
                )
                / (self.lambda_B + n_non_zero_B)
            )
            / 2
        )

    def sampling(self, num_samples=100000, seed=1234):
        np.random.seed(seed)
        self.pi_A_samples = beta(self.posterior_alpha_A, self.posterior_beta_A).rvs(
            num_samples
        )
        self.pi_B_samples = beta(self.posterior_alpha_B, self.posterior_beta_B).rvs(
            num_samples
        )
        delta_A_samples = bernoulli(self.pi_A_samples).rvs()
        delta_B_samples = bernoulli(self.pi_B_samples).rvs()

        self.mu_A_samples, self.sigma_A_samples = normal_inverse_gamma(
            self.posterior_theta_A,
            self.posterior_lambda_A,
            self.posterior_a_A,
            self.posterior_b_A,
        ).rvs(num_samples)
        self.mu_B_samples, self.sigma_B_samples = normal_inverse_gamma(
            self.posterior_theta_B,
            self.posterior_lambda_B,
            self.posterior_a_A,
            self.posterior_a_B,
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
            beta(self.posterior_alpha_A, self.posterior_beta_A).pdf(x),
            label="group A",
        )
        ax[1].plot(
            x,
            beta(self.posterior_alpha_B, self.posterior_beta_B).pdf(x),
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
