"""3D CNN moment network for field-level cosmological inference.
Architecture follows Villaescusa-Navarro et al. 2021 (arXiv:2109.10360), adapted
to 3D 128^3 count-overdensity fields: stacked Conv3d+BN+LeakyReLU blocks with
stride-2 downsampling, global pool, FC head outputting posterior mean mu and std
sigma for the 5 cosmological parameters. Trained with the moment-network loss
(Jeffrey & Wandelt 2020) whose minimizer is (E[theta|x], Var[theta|x])."""
import torch, torch.nn as nn, torch.nn.functional as F

class FieldCNN(nn.Module):
    def __init__(self, n_params=5, ch=(32,64,128,256,256,256)):
        super().__init__()
        self.n_params=n_params
        blocks=[]; cin=1
        for k,cout in enumerate(ch):
            # downsample FIRST (channel expansion happens at lower resolution ->
            # avoids a 268MB/sample activation at 128^3), then a stride-1 conv.
            blocks += [
                nn.Conv3d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(8,cout),cout), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(cout, cout, 3, stride=1, padding=1, bias=False),
                nn.GroupNorm(min(8,cout),cout), nn.LeakyReLU(0.2, inplace=True),
            ]
            cin=cout
        self.features=nn.Sequential(*blocks)         # 128 -> 2 (6 stride-2 steps)
        self.head=nn.Sequential(
            nn.Linear(cin, 128), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 2*n_params),              # [mu (5), raw_sigma (5)]
        )
    def forward(self, x):                            # x: (B,1,128,128,128)
        h=self.features(x)                           # (B,C,2,2,2)
        h=F.adaptive_avg_pool3d(h,1).flatten(1)      # (B,C)
        o=self.head(h)
        mu=o[:,:self.n_params]
        sigma=F.softplus(o[:,self.n_params:])+1e-6   # positive
        return mu, sigma

def moment_loss(mu, sigma, theta):
    """VN 2021 / Jeffrey-Wandelt moment loss. mu,sigma,theta: (B, n_params).
    term1 drives mu->E[theta|x]; term2 drives sigma^2->Var[theta|x]."""
    e=(theta-mu)**2                                  # (B,P)
    t1=torch.log(e.sum(0)).sum()                     # sum_i log sum_j (theta-mu)^2
    t2=torch.log(((e - sigma**2)**2).sum(0)).sum()   # sum_i log sum_j ((theta-mu)^2 - sig^2)^2
    return t1+t2
