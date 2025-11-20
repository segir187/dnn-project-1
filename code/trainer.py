class MultiTaskTrainer:
    def __init__(
        self,
        trainset_size: int = 9000,
        train_batch_size: int = 64,
        test_batch_size: int = 1000,
        use_cuda: bool = torch.cuda.is_available(),
    ) -> None:
        self.device = torch.device("cuda" if use_cuda else "cpu")

        train_indices = list(range(0, trainset_size))
        test_indices = list(range(trainset_size, 10000))

        self.trainset = CountsDataset(
            data_dir="data", labels_csv="data/labels.csv", indices=train_indices, augment=True
        )
        self.trainloader = DataLoader(
            self.trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=use_cuda
        )

        self.testset = CountsDataset(
            data_dir="data", labels_csv="data/labels.csv", indices=test_indices, augment=None
        )
        self.testloader = DataLoader(
            self.testset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda
        )

    def train(
        self,
        net: nn.Module,
        num_epochs: int = 100,
        lr: float = 0.001,
        lambda_cnt: float = 1.0,
        do_cls: bool = True,
    ) -> None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        LossCalc = MultiTaskLoss(lambda_cnt=lambda_cnt, do_cls=do_cls)
        stopper = EarlyStopping(mode="min", restore_best=True)
        log_freq = 140

        net.to(self.device)
        for epoch in range(num_epochs):
            net.train()
            train_loss_sum = 0.0
            seen = 0

            for i, data in enumerate(self.trainloader):
                inputs, counts, cls135 = [t.to(self.device, non_blocking=True) for t in data]

                optimizer.zero_grad()
                log_probs, counts_pred = net(inputs)
                loss = LossCalc((log_probs, counts_pred), (cls135, counts))
                loss.backward()
                optimizer.step()

                bs = inputs.size(0)
                train_loss_sum += float(loss.item()) * bs
                seen += bs

            metrics = self.test(net, LossCalc)
            metrics['train_loss'] = train_loss_sum / seen
            print_metrics(epoch, metrics)

            if stopper.step(metrics, net, epoch):
                print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best model from epoch {stopper.best_epoch + 1}.")
                stopper.restore(net)
                print_metrics(stopper.best_epoch, stopper.best_metrics)
                break

    def test(self, net: nn.Module, LossCalc: MultiTaskLoss) -> dict:
        net.eval()
        num_classes = 135

        cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        sse = 0.0
        abs_err_sum = 0.0
        loss_sum = 0.0

        with torch.no_grad():
            for data in self.testloader:
                inputs, counts, cls135 = [t.to(self.device, non_blocking=True) for t in data]
                log_probs, counts_pred = net(inputs)

                bs = cls135.size(0)
                loss = LossCalc((log_probs, counts_pred), (cls135, counts))
                loss_sum += loss * bs

                # Update confusion matrix
                predicted = log_probs.argmax(dim=1)
                flat_idx = cls135 * num_classes + predicted
                cm += torch.bincount(flat_idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).cpu()

                err = counts_pred - counts
                sse += err.pow(2).sum().item()
                abs_err_sum += err.abs().sum().item()

        total = cm.sum().item()
        result = {
            'top1_acc': 100.0 * cm.diag().sum().item() / total,
            'f1_macro': macro_f1(cm),
            'rmse': np.sqrt(sse / (total * 6)),
            'mae': abs_err_sum / (total * 6),
            'val_loss': loss_sum / total,
        }

        return result
