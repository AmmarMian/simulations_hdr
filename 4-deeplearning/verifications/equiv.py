import torch
from yetanotherspdnet.random.stiefel import random_stiefel
from yetanotherspdnet.functions.stiefel import (
    stiefel_projection_polar as polar,
    stiefel_projection_tangent_orthogonal as proj_tan,
)

torch.set_default_dtype(torch.float64)
d0, d1, K = 40, 20, 8
g = torch.Generator().manual_seed(0)
W = random_stiefel(d0, d1, 1, generator=g).squeeze(0)

print(f"{'eps':>10s} {'||projavg-rlavg||_F':>22s}   slope")
prev = None
for eps in [10.0**-k for k in range(0, 7)]:
    # K local weights at distance ~eps from W, obtained by retracting random tangent vectors
    Ws = []
    for c in range(K):
        E = torch.randn(d0, d1, generator=g)
        T = proj_tan(E, W); T = T / T.norm() * eps
        Ws.append(polar(W + T))
    Ws = torch.stack(Ws)
    projavg = polar(Ws.mean(0))
    rlavg = polar(W + torch.stack([proj_tan(Wc - W, W) for Wc in Ws]).mean(0))
    d = (projavg - rlavg).norm().item()
    slope = "" if prev is None else f"{(torch.log10(torch.tensor(prev/d))/1.0).item():.2f}"
    print(f"{eps:10.0e} {d:22.6e}   {slope}")
    prev = d
