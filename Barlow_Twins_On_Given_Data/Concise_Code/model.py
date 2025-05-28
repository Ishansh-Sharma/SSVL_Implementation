#barlowtwins model
class BarlowTwinsModel(nn.Module):
    def __init__(self, projector_dim=8192):
        super(BarlowTwinsModel, self).__init__()
        self.backbone = models.resnet18(weights = None)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )

    def forward(self, x):
        y = self.backbone(x)
        z = self.projector(y)
        return z
#barlow twin class
class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z1, z2):
        N, D = z1.size()
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        c = torch.mm(z1_norm.T, z2_norm) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
