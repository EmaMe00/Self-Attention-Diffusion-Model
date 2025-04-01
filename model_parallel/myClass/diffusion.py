from myClass.package import *
from model_param import *

class DiffusionModel:
    def __init__(self, device):
        self.n_steps = 1000

        # Inizializza beta, alpha e alpha_bar
        self.beta = torch.linspace(0.0001, 0.02, self.n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.device = device

        # Porta i tensori sul dispositivo corretto
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)

    def q_xt_x0(self, x0, t):
        n, c, h, w = x0.shape
        eps = torch.randn(n, c, h, w, device=self.device).contiguous()  # Usa il dispositivo corretto e garantisce contiguità

        a_bar = self.alpha_bar[t].contiguous()  # Garantisce che a_bar sia contiguo

        noisy = (
            a_bar.sqrt().reshape(n, 1, 1, 1).contiguous() * x0 +
            (1 - a_bar).sqrt().reshape(n, 1, 1, 1).contiguous() * eps
        )
        return noisy, eps

    def p_xt(self, xt, noise, t):
        xt = xt.to(self.device).contiguous()  # Muove xt al dispositivo corretto e garantisce contiguità
        noise = noise.to(self.device).contiguous()  # Muove noise al dispositivo corretto e garantisce contiguità

        alpha_t = self.alpha[t].contiguous()
        alpha_bar_t = self.alpha_bar[t].contiguous()

        eps_coef = ((1 - alpha_t) / (1 - alpha_bar_t) ** 0.5).contiguous()
        mean = (1 / (alpha_t ** 0.5).contiguous()) * (xt - eps_coef * noise).contiguous()

        var = self.beta[t].contiguous()
        eps = torch.randn(xt.shape, device=self.device).contiguous()

        return mean + (var ** 0.5).contiguous() * eps
