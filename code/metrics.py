def macro_f1(cm: Tensor) -> float:
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    precision = tp / (tp + fp).clamp(min=1.0)
    recall    = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return 100 * f1.mean().item()  # return as percentage

def pair_accuracy_matrix(cm: Tensor) -> dict:
    """
    Returns a 6x6 matrix M where M[i,j] = accuracy (%) of correctly predicting
    the unordered shape pair {i,j} ignoring counts (blocks of 9).
    Diagonal entries set to -1.
    Assumes class ordering: all (i,j) with i<j, each followed by its 9 count combos.
    """
    shapes = ['square','circle','up','right','down','left']
    pairs = [(i,j) for i in range(6) for j in range(i+1,6)]  # 15 blocks
    M = np.full((6,6), -1.0, dtype=np.float32)
    for block_idx,(a,b) in enumerate(pairs):
        start = block_idx * 9
        end = start + 9
        total = cm[start:end, :].sum().item()
        if total > 0:
            correct = cm[start:end, start:end].sum().item()
            acc = 100.0 * correct / total
        else:
            acc = 0.0
        M[a,b] = acc
        M[b,a] = acc
    return M

def format_pair_matrix(M: np.ndarray) -> str:
    shapes = ['square','circle','up','right','down','left']
    col_w = 9
    header = " " * (col_w) + "".join(f"{s:>{col_w}}" for s in shapes)
    rows = [header]
    for i,s in enumerate(shapes):
        line = f"{s:<{col_w}}"
        for j in range(6):
            if i == j:
                cell = "-"
            else:
                cell = f"{M[i,j]:.2f}"
            line += f"{cell:>{col_w}}"
        rows.append(line)
    return "\n".join(rows)

def print_metrics(epoch: int, metrics: dict, pair_acc=False):
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    top1_acc = metrics['top1_acc']
    f1_macro = metrics['f1_macro']
    rmse = metrics['rmse']
    mae = metrics['mae']
    print(f"Epoch {epoch + 1:2d} | Train loss {train_loss:.4f}, Val loss {val_loss:.4f} | Acc {top1_acc:4.1f}%, F1 {f1_macro:4.1f}% | RMSE {rmse:.4f}, MAE {mae:.4f}")
    if pair_acc:
        print(format_pair_matrix(metrics['pair_acc_matrix']))
