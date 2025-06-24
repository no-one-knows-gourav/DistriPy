import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class ContinuousRV:
  def __init__(self, pdf_expr, var, support):
    self.pdf = pdf_expr
    self.var = var
    self.a, self.b = support
    self.cdf = sp.integrate(self.pdf, (self.var, self.a, self.var))
    self.expectation = sp.integrate(self.var * self.pdf, (self.var, self.a, self.b))
    
  def get_pdf(self):
    return self.pdf 
  
  def get_cdf(self):
    return self.cdf 
  
  def get_expectation(self):
    return self.expectation 
  
  def evaluate_pdf(self, value, subs={}):
    return self.pdf.subs(self.var, value).subs(subs).evalf()
  
  def evaluate_cdf(self, value, subs={}):
    return self.cdf.subs(self.var, value).subs(subs).evalf()
  
  def get_variance(self):
    mu = self.expectation
    variance_expr = sp.integrate((self.var - mu)**2 * self.pdf, (self.var, self.a, self.b))
    return variance_expr.simplify()
  
  def normalize_pdf(self):
    Z = sp.integrate(self.pdf, (self.var, self.a, self.b))
    self.pdf = (self.pdf / Z).simplify()
    self.cdf = sp.integrate(self.pdf, (self.var, self.a, self.var))
    self.expectation = sp.integrate(self.var * self.pdf, (self.var, self.a, self.b))
  
  def standardize_pdf(self):
    mu = self.get_expectation()
    sigma2 = self.get_variance()
    sigma = sp.sqrt(sigma2)
    Z = (self.var - mu) / sigma
    return Z
  
  def plot_pdf(self, subs={}, num_points = 200):
    f_lambdified = sp.lambdify(self.var, self.pdf.subs(subs), "numpy")
    x_vals = np.linspace(float(self.a.subs(subs)), float(self.b.subs(subs)), num_points)
    y_vals = f_lambdified(x_vals)
        
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, label="PDF", color="darkblue")
    plt.fill_between(x_vals, y_vals, alpha=0.2, color="skyblue")
    plt.title("Probability Density Function")
    plt.xlabel(f"${sp.latex(self.var)}$")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.show()
    
  def plot_cdf(self, subs={}, num_points=200):
    F_lambdified = sp.lambdify(self.var, self.cdf.subs(subs), "numpy")
    x_vals = np.linspace(float(self.a.subs(subs)), float(self.b.subs(subs)), num_points)
    y_vals = F_lambdified(x_vals)

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, label="CDF", color="darkgreen")
    plt.title("Cumulative Distribution Function")
    plt.xlabel(f"${sp.latex(self.var)}$")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.show()