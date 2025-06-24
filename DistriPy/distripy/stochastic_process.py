from sympy import symbols, Function, Eq, simplify
from sympy.abc import n, i
from typing import Union
import matplotlib.pyplot as plt
import random

def is_stochastic_process(expr, index_variable):
    """
    Checks if the given symbolic expression represents a valid stochastic process.
    Parameters:
    - expr: sympy expression, e.g., X(n)
    - index_variable: sympy symbol representing time, e.g., n

    Returns:
    - bool: True if expr depends on index_variable and is a function of it.
    """
    # A stochastic process should be a function of the index_variable
    return expr.has(index_variable) and expr.is_Function

class StochasticProcess:
  
  def __init__(self, expression, index_variable, domain=None, symbolic=True):
    self.expression = expression
    self.index_variable = index_variable
    self.domain = domain
    self.symbolic = symbolic
    # Validation: ensure it's a proper function of the index_variable
    if not self.is_stochastic_process():
      raise ValueError(f"{expression} is not a valid stochastic process in terms of {index_variable}.")

class SymmetricRandomWalk:
  
  def __init__(self, steps: int, scale: float = 1.0):
    self.steps = steps
    self.scale = scale
    self.path = [0]  # Starting at origin
    self.generate_path()

  def generate_path(self):
    current = 0
    for _ in range(self.steps):
      step = random.choice([-self.scale, self.scale])
      current += step
      self.path.append(current)

  def get_path(self):
    return self.path

  def expectation(self):
    return 0  # Since symmetric

  def variance(self):
    return self.steps * (self.scale ** 2)

  def plot_walk(self):
    plt.plot(range(self.steps + 1), self.path, marker='o', linestyle='-')
    plt.title(f'Symmetric Random Walk ({self.steps} steps, scale={self.scale})')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True)
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt

class BrownianMotion:
  
  def __init__(self, T=1.0, N=1000, B0=0.0, seed=None):
    """
    Simulates a standard Brownian motion path.

    Args:
      T (float): Total time horizon.
      N (int): Number of time steps.
      B0 (float): Initial value of Brownian motion.
      seed (int): Random seed for reproducibility.
    """
    self.T = T
    self.N = N
    self.B0 = B0
    self.dt = T / N
    self.times = np.linspace(0, T, N + 1)
    self.seed = seed
    self.path = None

    def simulate(self):
      """Simulates the Brownian motion path."""
      if self.seed is not None:
        np.random.seed(self.seed)

      dW = np.sqrt(self.dt) * np.random.randn(self.N)
      W = np.concatenate(([self.B0], np.cumsum(dW)))
      self.path = W
      return self.times, W

  def get_path(self):
    """Returns the previously simulated path if available."""
    if self.path is None:
      raise ValueError("No path simulated yet. Call simulate() first.")
    return self.times, self.path

  def plot(self, title="Simulated Brownian Motion Path"):
    """Plots the Brownian motion path."""
    if self.path is None:
      raise ValueError("No path simulated yet. Call simulate() first.")

    plt.figure(figsize=(10, 5))
    plt.plot(self.times, self.path, color="mediumslateblue", lw=2)
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("B(t)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

  def ito_integral_W_dW(self):
    """Numerical method to calculate Ito integral W(T)dW(t)"""
    if self.path is None:
      raise ValueError("No path simulated yet. Call simulate() first.")
    dW = np.diff(self.path)
    W_mid = self.path[:-1]
    ito_sum = np.sum(W_mid * dW)
    return ito_sum 

  def ito_integral_closed_form(self):
    if self.path is None:
      raise ValueError("No path simulated yet. Call simulate() first.")
    W_T = self.path[-1]
    return 0.5 * W_T**2 - 0.5 * self.T
  
class GeometricBrownianMotion:
  
  def __init__(self, S0=1, mu=0.1, sigma=0.2, T=1.0, N=1000, seed=None):
    self.S0 = S0
    self.mu = mu
    self.sigma = sigma
    self.T = T
    self.N = N
    self.seed = seed
    self.t = None
    self.S = None
    self.W = None
    
  def simulate(self):
    if self.seed is not None:
      np.random.seed(self.seed)
    dt = self.T/self.N
    self.t = np.linspace(0, self.T, self.N + 1)
    dW = np.random.normal(0, np.sqrt(dt), size=self.N)
    W = np.concatenate(([0], np.cumsum(dW)))
    self.W = W
    exponent = (self.mu - 0.5 * self.sigma**2) * self.t + self.sigma * W
    self.S = self.S0 * np.exp(exponent)

  def plot(self):
    if self.S is None:
      raise ValueError("Simulate path first using simulate()")
    plt.plot(self.t, self.S, label="S(t)", color='forestgreen')
    plt.title("Geometric Brownian Motion")
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.legend()
    plt.show()
    
  def get_path(self):
    if self.S is None:
      raise ValueError("Simulate path first using simulate()")
    return self.t, self.S
    
class PoissonProcess:
  
  def __init__(self, rate, T):
    """args:
        rate lambda, average rate of increase
        time T, end time to simulate
        inter arrival ~ exp(rate)"""
    self.rate = rate
    self.T = T
    self.num_events = None
    self.event_times = None
    
  def simulate_event_times(self):
    # simulate the time of happening of events based on exponential interarrival times
    time = []
    current_time = 0
    while current_time < self.T:
      inter_arrival = np.random.exponential(1/self.rate)
      current_time += inter_arrival
      if current_time < self.T:
        time.append(current_time)
    self.event_times = np.array(time)
    return self.event_times
  
  def simulate_num_events(self, t):
    # simulate number of events occured by time t
    if self.event_times is None:
      self.simulate_event_times()
    return np.sum(self.event_times <= t)
  
  def plot(self):
    if self.event_times is None:
      self.simulate_event_times()
    times = self.event_times
    counts = np.arange(1, len(times)+1)
    plt.step(times, counts, where='post')
    plt.xlabel('Time')
    plt.ylabel('Number of Events')
    plt.title(f'Poisson Process (λ = {self.rate})')
    plt.grid(True)
    plt.show()
    
class OUProcess:
  
  def __init__(self, theta=0.7, mu=0.0, sigma=0.3, x0=0):
    self.theta = theta
    self.mu = mu
    self.sigma = sigma
    self.x0 = x0
    
  def simulate(self, T=1.0, N=1000, seed=None):
    if seed is not None:
      np.random.seed(seed)
    
    dt = T/N
    t = np.linspace(0, T, N+1)
    X = np.zeroes(N+1)
    
    for i in range(1, N+1):
      dW = np.random.normal(0, np.sqrt(dt))
      X[i] = X[i-1] + self.theta*(self.mu-X[i-1])*dt + self.sigma*dW 
    self.last_t = t
    self.last_X = X
    return t, X 
  
  def plot(self, T=1.0, N=1000):
    t, X = self.simulate(T, N)
    plt.plot(t, X, label='OU Process')
    plt.axhline(self.mu, color='red', linestyle='--', label='Mean $\mu$')
    plt.title('Ornstein–Uhlenbeck Process')
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.legend()
    plt.grid(True)
    plt.show()
  
  