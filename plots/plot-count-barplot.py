import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

label_cols = ['squares','circles','up','right','down','left']
df = pd.read_csv('data/labels.csv')

pairs = [(i, j) for i in range(6) for j in range(i+1, 6)]
pair_to_row = {p: r for r, p in enumerate(pairs)}

freq_135 = np.zeros((15, 9), dtype=int)
for vals in df[label_cols].to_numpy(dtype=int):
    nz = np.flatnonzero(vals)
    a, b = sorted(nz.tolist())
    row = pair_to_row[(a, b)]
    k = vals[nz[0]]  # first in CSV order
    col = k - 1
    freq_135[row, col] += 1

used_mask = freq_135 > 0
used_counts = freq_135[used_mask]

plt.figure(figsize=(5,3))
sns.histplot(used_counts, bins=16, color='#4C78A8')
plt.xlabel('Samples per used configuration cell')
plt.ylabel('Count of cells')
plt.title('Distribution of configuration cell frequencies (105 used cells)')
plt.tight_layout()
plt.show()

print("Min:", used_counts.min(), "Max:", used_counts.max(), "Mean:", used_counts.mean())