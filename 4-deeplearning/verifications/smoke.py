import time, torch, tracemalloc
from yetanotherspdnet.random.spd import random_SPD
from yetanotherspdnet.nn.batchnorm import BatchNormSPDMean

torch.manual_seed(0)
for mean_type in ["affine_invariant", "geometric_arithmetic_harmonic", "arithmetic", "harmonic", "log_euclidean"]:
    for ag in [False, True]:
        n, N = 32, 32
        X = random_SPD(n_features=n, n_matrices=N, cond=1e5, dtype=torch.float64)
        X.requires_grad_(True)
        layer = BatchNormSPDMean(n, mean_type=mean_type, use_autograd=ag,
                                 mean_options={"n_iterations": 5} if mean_type == "affine_invariant" else None)
        t0 = time.perf_counter()
        out = layer(X)
        loss = out.sum()
        loss.backward()
        dt = time.perf_counter() - t0
        print(f"{mean_type:35s} autograd={ag!s:5s} t={dt*1e3:7.1f} ms grad_norm={X.grad.norm():.4e}")
