print("Classification-only training")
model = MultiTaskNetwork()
trainer = MultiTaskTrainer(
    trainset_size=9000,
    train_batch_size=64,
    test_batch_size=1000,
    use_cuda=torch.cuda.is_available(),
)

start_time = time.perf_counter()
cls_only_metrics = trainer.train(
    net=model,
    num_epochs=100,
    lr=0.001,
    lambda_cnt=0.0,
    do_cls=True,
)
elapsed = time.perf_counter() - start_time
print(f"Classification only training runtime: {elapsed:.2f}s")

print("Regression-only training")
model = MultiTaskNetwork()
trainer = MultiTaskTrainer(
    trainset_size=9000,
    train_batch_size=64,
    test_batch_size=1000,
    use_cuda=torch.cuda.is_available(),
)

start_time = time.perf_counter()
reg_only_metrics = trainer.train(
    net=model,
    num_epochs=100,
    lr=0.001,
    lambda_cnt=1.0,
    do_cls=False,
)
elapsed = time.perf_counter() - start_time
print(f"Regression only training runtime: {elapsed:.2f}s")

print("Multitask training")
model = MultiTaskNetwork()
trainer = MultiTaskTrainer(
    trainset_size=9000,
    train_batch_size=64,
    test_batch_size=1000,
    use_cuda=torch.cuda.is_available(),
)

start_time = time.perf_counter()
multitask_metrics = trainer.train(
    net=model,
    num_epochs=100,
    lr=0.001,
    lambda_cnt=2.0,
    do_cls=True,
)
elapsed = time.perf_counter() - start_time
print(f"Multitask training runtime: {elapsed:.2f}s")
