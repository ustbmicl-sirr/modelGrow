import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'


cifar10 = {
    "accuracy":   [56.57,      78.57,      86.75,      89.01,       88.76,       87.60,       86.12,       86.02,        86.67],
    "parameters": [1.20,       4.05,       6.22,       8.85,        11.69,       21.80,       25.56,       44.55,        60.19],
    "layers":     ["1",   "5",   "9", "13", "18", "34", "50", "101", "151"]
}


cifar100 = {
    "accuracy":   [35.55, 45.44, 52.11,      56.78,       55.80,       55.40,       53.16,       53.40,        54.18],
    "parameters": [1.20,  4.05,  6.22,       8.85,        11.69,       21.80,       25.56,       44.55,        60.19],
    "layers":     ["1",   "5",   "9", "13", "18", "34", "50", "101", "151"]
}


mnist = {
    "accuracy":   [95.66,      97.84,      99.35,      99.82,       99.80,       99.84,       99.76,       99.76,        99.85],
    "parameters": [1.20,       4.05,       6.22,       8.85,        11.69,       21.80,       25.56,       44.55,        60.19],
    "layers":     ["1",   "5",   "9", "13", "18", "34", "50", "101", "151"]
}


imagenet = {
    "accuracy":   [26.85,      39.60,      50.01,      63.35,       69.80,       73.30,       76.20,       77.40,        78.30],
    "parameters": [1.20,       4.05,       6.22,       8.85,        11.69,       21.80,       25.56,       44.55,        60.19],
    "layers":     ["1",   "5",   "9", "13", "18", "34", "50", "101", "151"]
}


data = cifar10
backbone = "cifar10"

plt.figure(figsize=(4, 3)) 

scaled_parameters = np.array(data["parameters"]) * 100
plt.scatter(data["layers"], data["accuracy"], s=scaled_parameters, alpha=0.5, edgecolors='w', linewidth=2, marker='o')

plt.plot(data["layers"], data["accuracy"], 'o-', markeredgecolor='k', markerfacecolor='r', markersize=5, linewidth=2)

for size in [1, 3, 5]:
    plt.scatter([], [], c='#3685bc', alpha=0.5, edgecolors='w', linewidth=2, s=size * 100, label=f'{size} million parameters', marker='o')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='ResNet Parameters Scale', loc='lower right')

plt.xlabel('Layers')
plt.ylabel('Accuracy (%)')
plt.ylim(50, 95)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(data["layers"])
plt.tight_layout()
plt.savefig("./"+backbone+".svg")
plt.show()