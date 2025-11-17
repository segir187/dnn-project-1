import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

label_cols = ['squares','circles','up','right','down','left']
df = pd.read_csv('data/labels.csv')

smaller = []
for vals in df[label_cols].to_numpy(dtype=int):
    nz_vals = vals[vals > 0]
    smaller.append(int(min(nz_vals)))
freq = pd.Series(smaller).value_counts().reindex(range(1,6), fill_value=0).sort_index()

plt.figure(figsize=(6,3))
sns.barplot(x=list(freq.index), y=freq.values, color='#F58518')
plt.title('Distribution of smaller count in (k, 10−k)')
plt.xlabel('smaller of two counts (k)')
plt.ylabel('# images')
plt.tight_layout()
plt.show()
