import torch


class ManualLion:
    """
    DirectML-safe Lion optimizer.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.state = {}

        for p in self.params:
            if p.requires_grad:
                self.state[p] = {
                    "exp_avg": torch.zeros_like(p)
                }

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.beta1, self.beta2
        lr = self.lr
        wd = self.weight_decay

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad

            if wd != 0:
                grad = grad.add(p, alpha=wd)

            exp_avg = self.state[p]["exp_avg"]

            # m = beta1*m + (1-beta1)*grad
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # update with sign of momentum
            p.add_(exp_avg.sign(), alpha=-lr)
