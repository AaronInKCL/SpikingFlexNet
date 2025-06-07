import torch
import torch.nn as nn

import torch.nn.functional as F

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # piecewise-linear approximation
        mask = (input.abs() < 1).float()
        return grad_input * mask * 0.3  # scale factor


class SpikingReLU(nn.Module):
    def __init__(self, threshold=1.0, decay=0.95):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = None
        self.spike_times = []

    def forward(self, x):
        if self.membrane_potential is None or self.membrane_potential.shape != x.shape:
            self.membrane_potential = torch.zeros_like(x)

        self.membrane_potential = self.membrane_potential * self.decay + x
        spikes = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spikes)
        return spikes


# class SpikingActivation(nn.Module):
#     def __init__(self, surrogate=False, record_neurons=128):
#         super().__init__()
#         self.surrogate = surrogate
#         self.spike_history = []
#         self.record_neurons = record_neurons

#     def forward(self, x):
#         if self.surrogate:
#             out = (x > 0).float()
#             surrogate_grad = 0.3 * torch.tanh(x)
#             spike = out.detach()
#             out = out + surrogate_grad - surrogate_grad.detach()
#         else:
#             out = torch.sigmoid(10 * x)
#             spike = (out > 0.5).float().detach()

#         # 保留前 N 个神经元
#         spike_flat = spike.mean(dim=0).view(-1)
#         fixed_length_spike = spike_flat[:self.record_neurons]
#         self.spike_history.append(fixed_length_spike)

#         return out

#     def reset_spike_history(self):
#         self.spike_history.clear()

class SpikingActivation(nn.Module):
    def __init__(self, surrogate=False):
        super().__init__()
        self.surrogate = surrogate

    def forward(self, x):
        if self.surrogate:
            spike = (x > 0).float()
            surrogate_grad = 0.3 * torch.tanh(x)
            return spike + surrogate_grad - surrogate_grad.detach()
        else:
            return torch.clamp(x, 0, 1)




def forward_STBP(self, x, T=10):
    # shape: [B, C, H, W]
    mem_output = 0
    for t in range(T):
        # 对每一帧重复输入（或给真实帧，取决于你的输入形式）
        xt = x  # 如果有动态输入，可加 spike encoder
        out = self.features(xt)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        mem_output += out  # 输出累积
    return mem_output / T  # 时间平均
