import torch
from yetanotherspdnet.random.spd import random_SPD
from yetanotherspdnet.functions.spd_linalg import whitening, Whitening, congruence_SPD, CongruenceSPD

def run(fn, cond, seed=0, n=8, N=4):
    g = torch.Generator().manual_seed(seed)
    X = random_SPD(n, N, cond=cond, dtype=torch.float64, generator=g).clone().requires_grad_(True)
    g2 = torch.Generator().manual_seed(seed+7)
    M = random_SPD(n, 1, cond=5.0, dtype=torch.float64, generator=g2).clone().requires_grad_(True)
    W = torch.arange(1., N*n*n+1., dtype=torch.float64).reshape(N,n,n)
    (fn(X, M)*W).sum().backward()
    return X.grad.clone(), M.grad.clone()

for name, f_auto, f_man in [("whitening", whitening, Whitening.apply),
                            ("congruence", congruence_SPD, CongruenceSPD.apply)]:
    for cond in [2.0, 1e5]:
        (xa, ma), (xm, mm) = run(f_auto, cond), run(f_man, cond)
        print(f"{name:10s} cond={cond:7.0e}  dX rel={(xa-xm).norm()/xa.norm():.3e}   dM rel={(ma-mm).norm()/ma.norm():.3e}")

print("--- sym check")
for name, f_auto, f_man in [("whitening", whitening, Whitening.apply),
                            ("congruence", congruence_SPD, CongruenceSPD.apply)]:
    (xa, ma), (xm, mm) = run(f_auto, 2.0), run(f_man, 2.0)
    sym = 0.5*(ma+ma.transpose(-1,-2))
    print(f"{name:10s} ||man - sym(auto)||/||.|| = {(mm-sym).norm()/sym.norm():.3e}   ||man-2*sym||={(mm-2*sym).norm()/sym.norm():.3e}")
