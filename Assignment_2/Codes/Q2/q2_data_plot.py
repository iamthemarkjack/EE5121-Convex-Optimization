import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../Data/Data_Q2.csv", header=None)
X = df.iloc[1:, :-1].values.astype(float)
y = df.iloc[1:, -1].values.astype(float)

plt.figure(figsize=(10, 7))
colors = ['#FF6B6B', '#4ECDC4']
scatter = plt.scatter(X[:,0], X[:, 1], 
                     c=y, 
                     cmap=plt.cm.colors.ListedColormap(colors),
                     s=100, 
                     alpha=0.7, 
                     edgecolors='white',
                     linewidth=1.5)

plt.xlabel(r'$x_1$', fontsize=14, fontweight='bold')
plt.ylabel(r'$x_2$', fontsize=14, fontweight='bold')

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("../../Outputs/data_plot.png", dpi=300)
plt.show()