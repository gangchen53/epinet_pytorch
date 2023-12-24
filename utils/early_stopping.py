class EarlyStopping:
    def __init__(self, patience: int = 100, is_minimize: bool = True):
        self.patience = patience
        self.counter = 0

        self.best_score = float('inf')

        self.early_stop = False
        self.is_minimize = is_minimize

    def __call__(self, current_score: float):
        if not self.is_minimize:
            current_score = -current_score

        if current_score < self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
