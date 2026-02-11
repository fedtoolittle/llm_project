import math
import torch


class ManualAdam:
    """
    DirectML-safe Adam implementation.
    Avoids torch.lerp_ (which causes DML CPU fallback).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.state = {}
        self.step_count = 0

        # Initialize state for each parameter
        for p in self.params:
            if p.requires_grad:
                self.state[p] = {
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_count += 1

        beta1 = self.beta1
        beta2 = self.beta2
        lr = self.lr
        eps = self.eps
        wd = self.weight_decay

        bias_correction1 = 1 - beta1 ** self.step_count
        bias_correction2 = 1 - beta2 ** self.step_count

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad

            if wd != 0:
                grad = grad.add(p, alpha=wd)

            state = self.state[p]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # m = beta1 * m + (1 - beta1) * grad
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # v = beta2 * v + (1 - beta2) * grad^2
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1

            # Parameter update
            p.addcdiv_(exp_avg, denom, value=-step_size)
