def macro_f1(cm: Tensor) -> float:
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    precision = tp / (tp + fp).clamp(min=1.0)
    recall    = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return 100 * f1.mean().item()  # return as percentage

def print_metrics(epoch: int, metrics: dict):
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    top1_acc = metrics['top1_acc']
    f1_macro = metrics['f1_macro']
    rmse = metrics['rmse']
    mae = metrics['mae']
    print(f"Epoch {epoch + 1:2d} | Train loss {train_loss:.4f}, Val loss {val_loss:.4f} | Acc {top1_acc:4.1f}%, F1 {f1_macro:4.1f}% | RMSE {rmse:.4f}, MAE {mae:.4f}")
