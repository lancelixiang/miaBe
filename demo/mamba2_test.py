# 依赖包有问题，需要修改包中的代码，这里略过

import torch
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 128
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim,  # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    headdim=4
).to("cuda")
y = model(x)
assert y.shape == x.shape
