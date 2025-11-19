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
        lambda_cls = float(do_cls)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        log_freq = 140
        NLL = nn.NLLLoss()
        SL1 = nn.SmoothL1Loss()

        net.to(self.device)
        for epoch in range(num_epochs):
            net.train()
            running_loss = 0.0

            for i, data in enumerate(self.trainloader):
                inputs, counts, cls135 = [t.to(self.device, non_blocking=True) for t in data]

                optimizer.zero_grad()
                log_probs, counts_pred = net(inputs)
                loss_cls = NLL(log_probs, cls135)
                loss_cnt = SL1(counts_pred, counts)
                loss = lambda_cls * loss_cls + lambda_cnt * loss_cnt
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % log_freq == log_freq - 1:
                    running_loss /= log_freq
                    print(f"[epoch {epoch + 1:3d}, batch {i + 1:3d}] loss: {running_loss:.3f}")
                    running_loss = 0.0

            acc, rmse = self.test(net)
            print(f"Metrics after epoch {epoch + 1:3d}: Top1 {acc:2.1f}, RMSE {rmse:2.5f}")
    
    def test(self, net: nn.Module) -> (float, float):
        net.eval()
        correct = 0
        total = 0
        rmse_sum = 0.0

        with torch.no_grad():
            for data in self.testloader:
                inputs, counts, cls135 = [t.to(self.device, non_blocking=True) for t in data]
                log_probs, counts_pred = net(inputs)

                predicted = log_probs.argmax(dim=1)
                total += cls135.size(0)
                correct += (predicted == cls135).sum().item()
                rmse_sum += (counts_pred - counts).pow(2).sum().item()

        acc = 100.0 * correct / total
        rmse = np.sqrt(rmse_sum / (total * 6)) # global RMSE over 6 targets
        return acc, rmse
