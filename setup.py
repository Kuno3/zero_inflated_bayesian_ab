from setuptools import setup, find_packages

setup(
    name="zero_inflated_bayesian_ab",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib"],
)
