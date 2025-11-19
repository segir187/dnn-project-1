class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_cnt: float = 1.0, do_cls: bool = True):
        super().__init__()
        self.lambda_cls = 1.0 if do_cls else 0.0
        self.lambda_cnt = lambda_cnt
        self.nll = nn.NLLLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, inputs, targets):
        # inputs: (log_probs, counts_pred)
        # targets: (cls135, counts)
        log_probs, counts_pred = inputs
        cls135, counts = targets

        zero = log_probs.new_zeros(())
        loss_cls = self.nll(log_probs, cls135) if self.lambda_cls > 0.0 else zero
        loss_cnt = self.sl1(counts_pred, counts) if self.lambda_cnt > 0.0 else zero
        return self.lambda_cls * loss_cls + self.lambda_cnt * loss_cnt
