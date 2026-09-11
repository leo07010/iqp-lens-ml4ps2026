"""Structural claim: with 1 ancilla at the MSB, the image marginal is EXACTLY an equal-weight
INCOHERENT mixture of two 10-qubit holograms,
    q_img = 2|u|^2 + 2|v|^2,  u = 2^-11 FWHT(e^{i phi(a=0,.)}),  v = 2^-11 FWHT(e^{i phi(a=1,.)})
with each branch carrying mass exactly 1/2 for EVERY theta. So the ancilla generators cannot
change the mixing weight -- there is no 'mixing angle' to tune."""
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "<REPO_ROOT>/code")
sys.path.insert(0, HERE)
torch.set_num_threads(4)
import mech_test as M, mech_test2 as M2
from iqp_exact import fwht

def branches(theta, n_img=10):
    n = n_img + 1
    phi = fwht(theta)                       # (2^n,)
    f = torch.complex(torch.cos(phi), torch.sin(phi)).reshape(2, 1 << n_img)
    u = fwht(f[0]) / float(1 << n)
    v = fwht(f[1]) / float(1 << n)
    return 2*(u.abs()**2), 2*(v.abs()**2)

# 1) identity holds for ANY theta
g = torch.Generator().manual_seed(7)
for k in range(3):
    th = torch.randn(2048, generator=g, dtype=torch.float64) * (0.5*(k+1))
    b0, b1 = branches(th)
    q = M.marginal(M.IQPFull(11).probs(th), 10)
    print("random theta scale %.1f : branch masses %.12f / %.12f, |q_img-(b0+b1)|inf %.2e"
          % (0.5*(k+1), float(b0.sum()), float(b1.sum()), float((q-(b0+b1)).abs().max())))

# 2) what the FITTED (b, mu=.03) model does with its two branches
d,e1,e2,rad,ia,ib = M.load_pair()
S = float(0.5*(d.train_totals[ia]+d.train_totals[ib]))
pA = np.asarray(d.train_pmfs[ia]); pB = np.asarray(d.train_pmfs[ib])
o = M2.run("chk", 10, 1, M.latent_masks_ancilla(10,1), [pA,pB], S, steps=4000, mu=0.03)
print("\n(refit reproduces: endpoint corrA %.4f corrB %.4f)"
      % (o["wrapped"]["rows"][0]["corr_to"][0], o["wrapped"]["rows"][-1]["corr_to"][1]))
