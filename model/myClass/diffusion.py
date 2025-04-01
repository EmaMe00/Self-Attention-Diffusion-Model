from myClass.package import *
from model_param import *

beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def q_xt_x0(x0, t):
    n, c, h, w = x0.shape
    x0 = x0.to(device)  # Move x0 to the device
    eps = torch.randn(n, c, h, w, device=device)  # Ensure eps is on the correct device

    a_bar = alpha_bar[t].to(device)  # Ensure alpha_bar[t] is on the correct device

    noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eps
    return noisy, eps 

def p_xt(xt, noise, t):
    xt = xt.to(device)  # Ensure xt is on the correct device
    noise = noise.to(device)  # Ensure noise is on the correct device
    
    alpha_t = alpha[t].to(device)  # Ensure alpha[t] is on the correct device
    alpha_bar_t = alpha_bar[t].to(device)  # Ensure alpha_bar[t] is on the correct device
    
    eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
    mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise)
    var = beta[t].to(device)  # Ensure beta[t] is on the correct device
    eps = torch.randn(xt.shape, device=device)
    
    return mean + (var ** 0.5) * eps
