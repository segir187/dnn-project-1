import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

label_cols = ['squares','circles','up','right','down','left']
names = ['square','circle','up','right','down','left']
df = pd.read_csv('data/labels.csv')

pairs = [(i, j) for i in range(6) for j in range(i+1, 6)]
row_labels = [f"{names[i]}-{names[j]}" for (i, j) in pairs]
pair_to_row = {p: r for r, p in enumerate(pairs)}

freq = np.zeros((len(pairs), 9), dtype=int)
for vals in df[label_cols].to_numpy(dtype=int):
    i, j = sorted(np.flatnonzero(vals).tolist())
    r = pair_to_row[(i, j)]
    c = vals[i] - 1
    freq[r, c] += 1

plt.figure(figsize=(10, 6))
sns.heatmap(freq, cmap='Blues', annot=False,
            xticklabels=[str(k) for k in range(1,10)],
            yticklabels=row_labels, cbar_kws={'label': '# images'})
plt.xlabel('count of first shape in pair (1..9)')
plt.ylabel('unordered shape pair (i<j)')
plt.title('Distribution over 135 configuration classes')
plt.tight_layout()
plt.show()
