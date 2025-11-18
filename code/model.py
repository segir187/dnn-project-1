class MultiTaskNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 28 * 28, 256), nn.ReLU()
        )
        self.head_cls = nn.Sequential(
            nn.Linear(256, 135),
            nn.LogSoftmax(dim=-1)
        )
        self.head_cnt = nn.Linear(256, 6)

    def forward(self, x: Tensor):
        """
        x: (N, 1, 28, 28)
        returns:
          - log_probs: (N, 135)
          - counts:    (N, 6)
        """
        feat = self.backbone(x)
        log_probs = self.head_cls(feat)
        counts = self.head_cnt(feat)
        return log_probs, counts
