# DistriPy

DistriPy is a lightweight Python library for learning, experimenting, visualising and symbolically manipulating probability distributions and stochastic processes.

This library is intended for educational and visualisation purposes and supports symbolic representations to make it academically user-friendly. Currently, this is a solo project but contributions in the form of commits or feedback are highly appreciated. For more information check out CONTRIBUTIONS.md 

This package was initially created by <github id: no-one-knows-gourav> to compile basic knowledge of probabilities, distributions and stochastics. DistriPy prioritizes clarity, intuition, and extensibility over heavy abstraction. This is not meant to replace libraries like `scipy.stats`, but rather serve as a personal, open-source playground to consolidate and explore the theoretical foundations of probability and stochastic calculus.

## Features 

- Define **symbolic continuous random variables** with custom PDF and CDF
- Plot **PDF and CDF** interactively using `matplotlib`
- Compute **expectation** of a function of a random variable
- Support for **stochastic processes** like:
  - Brownian Motion
  - Geometric Brownian Motion
  - Ornstein-Uhlenbeck Process
  - Symmetric Random Walk
  - Poisson Process
- Simulate and visualize sample paths
- Modular, extensible codebase written using `sympy`, `numpy`, and `matplotlib` 

## Documentation 

Full documentation, references and theory are currently under development and will be available via Github Wiki and in-code docstrings.

## Motivation

This project was born out of a desire to consolidate understanding of probability theory, stochastic processes, and quantitative finance through code. Rather than just using powerful tools like scipy.stats, Distripy is about building from the ground up—understanding every line and every integral. Further contributions are expected to uphold this motivation and constructively contribute to the expansion and efficiency of the project.

## Installation

Coming soon on PyPI.

For now, clone the repo:
```bash
git clone https://github.com/no-one-knows-gourav/DistriPy.git
cd DistriPy
pip install -e .
