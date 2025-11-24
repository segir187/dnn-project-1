def _as_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return None if v is None else float(v)

def metrics_to_df(metrics_list):
    rows = []
    for i, m in enumerate(metrics_list):
        rows.append({
            "epoch": m.get("epoch", i),
            "train_loss": _as_float(m.get("train_loss")),
            "val_loss": _as_float(m.get("val_loss")),
            "top1_acc": _as_float(m.get("top1_acc")),
            "f1_macro": _as_float(m.get("f1_macro")),
            "rmse": _as_float(m.get("rmse")),
            "mae": _as_float(m.get("mae")),
        })
    return pd.DataFrame(rows)

def plot_cls(metrics_list, savepath=None):
    """
    Classification-focused: loss + (accuracy & macro F1).
    """
    df = metrics_to_df(metrics_list)
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    # Loss
    ax[0].plot(df.epoch, df.train_loss, label="Train")
    ax[0].plot(df.epoch, df.val_loss, label="Val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # Acc & F1
    ax[1].plot(df.epoch, df.top1_acc, label="Top-1 Acc", color="tab:blue")
    ax[1].plot(df.epoch, df.f1_macro, label="Macro F1", color="tab:green")
    ax[1].set_title("Accuracy & F1"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Score")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, bbox_inches="tight")

def plot_reg(metrics_list, savepath=None):
    """
    Regression-focused: loss + (RMSE & MAE).
    """
    df = metrics_to_df(metrics_list)
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    # Loss
    ax[0].plot(df.epoch, df.train_loss, label="Train")
    ax[0].plot(df.epoch, df.val_loss, label="Val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # RMSE & MAE
    ax[1].plot(df.epoch, df.rmse, label="RMSE", color="tab:orange")
    ax[1].plot(df.epoch, df.mae, label="MAE", color="tab:red")
    ax[1].set_title("RMSE & MAE"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Error")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    if savepath: fig.savefig(savepath, bbox_inches="tight")

def plot_all(metrics_list, savepath=None):
    """
    Three plots: top-left (Loss), top-right (Acc & F1), bottom (RMSE & MAE) centered.
    """
    df = metrics_to_df(metrics_list)
    fig = plt.figure(figsize=(10,7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1,0.9])

    ax_loss = fig.add_subplot(gs[0,0])
    ax_reg = fig.add_subplot(gs[0,1])

    # Manually add centered bottom axis (same width as a top axis)
    # Determine width from top-left axis, then center horizontally.
    top_pos = ax_loss.get_position()
    axis_width = top_pos.width
    bottom_height = top_pos.height * 0.9
    # Center: (1 - axis_width)/2 for left x.
    left = (1.0 - axis_width) / 2.0
    bottom = 0.07  # adjust vertical placement
    ax_cls = fig.add_axes([left, bottom, axis_width, bottom_height])

    # Loss
    ax_loss.plot(df.epoch, df.train_loss, label="Train")
    ax_loss.plot(df.epoch, df.val_loss, label="Val")
    ax_loss.set_title("Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    # RMSE & MAE centered
    ax_reg.plot(df.epoch, df.rmse, label="RMSE", color="tab:orange")
    ax_reg.plot(df.epoch, df.mae, label="MAE", color="tab:red")
    ax_reg.set_title("RMSE & MAE"); ax_reg.set_xlabel("Epoch"); ax_reg.set_ylabel("Error")
    ax_reg.legend(); ax_reg.grid(alpha=0.3)

    # Acc & F1
    ax_cls.plot(df.epoch, df.top1_acc, label="Top-1 Acc", color="tab:blue")
    ax_cls.plot(df.epoch, df.f1_macro, label="Macro F1", color="tab:green")
    ax_cls.set_title("Acc & F1"); ax_cls.set_xlabel("Epoch"); ax_cls.set_ylabel("Score")
    ax_cls.legend(); ax_cls.grid(alpha=0.3)

    if savepath: fig.savefig(savepath, bbox_inches="tight")

plot_cls(cls_only_metrics)
plot_reg(reg_only_metrics)
plot_all(multitask_metrics)
