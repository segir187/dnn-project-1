from copy import deepcopy

class EarlyStopping:
    def __init__(self, mode="min", patience=5, min_delta=0.0, restore_best=True):
        assert mode in ("min", "max")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best = None
        self.best_state = None
        self.best_epoch = -1
        self.num_bad_epochs = 0

    def _is_better(self, current, best):
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

    def step(self, current_value, model, epoch_idx):
        if self.best is None or self._is_better(current_value, self.best):
            self.best = current_value
            self.best_state = deepcopy(model.state_dict())
            self.best_epoch = epoch_idx
            self.num_bad_epochs = 0
            return False  # do not stop
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs > self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
