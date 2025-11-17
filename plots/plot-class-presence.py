import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

label_cols = ['squares','circles','up','right','down','left']
names = ['square','circle','up','right','down','left']

df = pd.read_csv('data/labels.csv')

present_counts = (df[label_cols] > 0).sum(axis=0).reindex(label_cols)

plt.figure(figsize=(6,3))
sns.barplot(x=names, y=present_counts.values, color='#4C78A8')
plt.title('Class presence (images containing class)')
plt.ylabel('# images')
plt.xlabel('class')
plt.tight_layout()
plt.show()
