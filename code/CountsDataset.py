from torch.utils.data import Dataset, DataLoader

class CountsDataset(Dataset):
    """
    Preloads labels and images once in __init__:
      - counts: tensor [N, 6] (float32)
      - images: tensor [N, 28*28] (float32), binary in {0.0, 1.0}, flattened
    No transformations.
    """
    def __init__(self, data_dir="data", labels_csv="data/labels.csv", indices=None, img_size=(28, 28)):
        self.data_dir = Path(data_dir)
        self.img_size = tuple(img_size)
        self.label_cols = ['squares','circles','up','right','down','left']

        # Read CSV rows
        rows = []
        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if indices is not None:
            rows = [rows[i] for i in indices]

        n = len(rows)
        H, W = self.img_size

        # Allocate tensors
        self.images = torch.empty((n, H * W), dtype=torch.float32)
        self.counts = torch.empty((n, 6), dtype=torch.float32)
        self.names = []

        # Load and store
        for i, row in enumerate(rows):
            img_path = self.data_dir / row['name']
            img = Image.open(img_path).convert('L')
            if img.size != (W, H):
                img = img.resize((W, H), Image.NEAREST)

            arr = np.asarray(img, dtype=np.uint8, copy=True).reshape(-1)  # (784,)
            arr = (arr > 128).astype(np.float32)
            self.images[i] = torch.from_numpy(arr)
            self.counts[i] = torch.tensor([int(row[c]) for c in self.label_cols], dtype=torch.float32)
            self.names.append(row['name'])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Returns: (image_flat_784_float32, counts_float32_6)
        return self.images[idx], self.counts[idx]

# Example split and loaders
labels_csv = "data/labels.csv"
train_indices = list(range(0, 9000))
val_indices   = list(range(9000, 10000))

train_ds = CountsDataset(data_dir="data", labels_csv=labels_csv, indices=train_indices)
val_ds   = CountsDataset(data_dir="data", labels_csv=labels_csv, indices=val_indices)

# Use <=2 workers on this system; pin memory only if CUDA is available
use_cuda = torch.cuda.is_available()
num_workers = 2
pin_memory = bool(use_cuda)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
val_loader   = DataLoader(val_ds,   batch_size=1000, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

# quick sanity check
batch_x, batch_counts = next(iter(train_loader))
print("train batch x:", batch_x.shape, batch_x.dtype, "| counts:", batch_counts.shape, batch_counts.dtype)
print("example row:", batch_x[0])
