import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'


basic_resnet = {
    "layers" :    [2,     3,     4,     5,     6,     7,     8,     9,     10,    11,    12,    13,    14,    15,    16,    17],
    "accuracy":   [56.57, 78.57, 86.75, 88.41, 88.76, 89.46, 89.52, 89.35, 90.68, 89.51, 89.43, 89.63, 86.68, 84.01, 85.42, 86.34],
    "parameters": [0.376, 0.451, 0.746, 1.930, 2.000, 2.300, 3.480, 3.550, 3.850, 5.030, 5.100, 5.400, 6.580, 6.650, 6.950, 8.130]
}


bottleneck_resnet = {
    "layers" :    [2,     3,     4,     5,     6,     7,     8,     9,     10,    11,    12,    13,    14,    15,    16,    17],
    "accuracy":   [55.89, 68.91, 82.09, 85.26, 85.87, 85.69, 86.70, 86.75, 86.57, 86.46, 86.73, 87.36, 85.46, 85.07, 86.10, 84.72],
    "parameters": [0.376, 0.381, 0.399, 0.469, 0.474, 0.492, 0.562, 0.567, 0.585, 0.655, 0.659, 0.677, 0.748, 0.752, 0.770, 0.840]
}


vgg = {
    "layers" :    [2,     3,     4,     5,     6,     7,     8,     9,     10,    11,    12,    13,    14,    15,    16,    17],
    "accuracy":   [55.90, 72.95, 81.78, 84.72, 85.77, 87.31, 87.63, 87.96, 88.36, 88.09, 88.36, 88.00, 87.76, 87.97, 86.79, 87.22],
    "parameters": [0.376, 0.414, 0.561, 1.150, 1.190, 1.340, 1.930, 1.960, 2.110, 2.700, 2.740, 2.890, 3.480, 3.520, 3.660, 4.250]
}


mobilenetv3 = {
    "layers" :    [2,     3,     4,     5,     6,     7,     8,     9,     10,    11,    12,    13,    14,    15,    16,    17],
    "accuracy":   [54.41, 74.80, 86.37, 87.09, 87.94, 88.46, 88.25, 88.64, 89.05, 88.93, 89.09, 88.81, 88.33, 88.40, 88.97, 88.23],
    "parameters": [0.096, 0.114, 0.188, 0.466, 0.484, 0.558, 0.836, 0.854, 0.927, 1.210, 1.220, 1.300, 1.580, 1.590, 1.670, 1.950]
}

data = mobilenetv3
backbone = "mobilenetv3"
Backbone = "MobileNetV3"

plt.figure(figsize=(4, 3)) 

scaled_parameters = np.array(data["parameters"]) * 100
plt.scatter(data["layers"], data["accuracy"], s=scaled_parameters, alpha=0.5, edgecolors='w', linewidth=2, marker='o')

plt.plot(data["layers"], data["accuracy"], 'o-', markeredgecolor='k', markerfacecolor='r', markersize=5, linewidth=2)

for size in [1, 3, 5]:
    plt.scatter([], [], c='#3685bc', alpha=0.5, edgecolors='w', linewidth=2, s=size * 100, label=f'{size} million parameters', marker='o')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title=Backbone+' Parameters Scale', loc='lower right')

plt.xlabel('Layers')
plt.ylabel('Accuracy (%)')
plt.ylim(53, 95)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(data["layers"])
plt.tight_layout()
plt.savefig("./"+backbone+".svg")
plt.show()