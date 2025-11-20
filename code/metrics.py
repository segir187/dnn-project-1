def print_metrics(epoch: int, metrics: dict):
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    top1_acc = metrics['top1_acc']
    rmse = metrics['rmse']
    print(f"Epoch {epoch + 1:2d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Acc {top1_acc:4.1f} | RMSE {rmse:.4f}")
