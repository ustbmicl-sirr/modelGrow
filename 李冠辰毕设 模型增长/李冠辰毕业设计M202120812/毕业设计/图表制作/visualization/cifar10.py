import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from matplotlib import rcParams

config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


# Data
models = [
    "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "自增长ResNet 1-2-2", "自增长ResNet 1-2-2-1", 
    "自增长Bottleneck-ResNet 2-2-2", "自增长Bottleneck-ResNet 2-2-2-2", "VGG-11", "VGG-16", 
    "自增长VGG 1-2-2", "自增长VGG 1-1-2-2", "MobileNetV3-small", "MobileNetV3-large", 
    "自增长MobileNetV3 2-3-3", "自增长MobileNetV3 2-2-3-2"
]

architecture = [
    "2-2-2-2", "3-4-6-3", "3-4-6-3 bottleneck", "3-4-23-3 bottleneck", "1-2-2", "1-2-2-1", 
    "2-2-2", "2-2-2-2", "1-1-2-2-2", "2-2-3-3-3", "1-2-2", "1-1-2-2", "2-2-3-2-2", "2-2-3-4-2-2", 
    "2-3-3", "2-2-3-2"
]

params = np.array([
    11.69, 21.80, 25.56, 44.55, 1.17, 2.65, 0.23507, 0.67321, 140.72, 146.22, 
    0.66929, 2.06, 2.84, 5.74, 0.8114, 0.67052
])

compute = [
    558.39, 1160, 1310, 2530, 222.99, 246.66, 41.09, 48.14, 440.32, 601.25, 
    123.37, 125.72, 8.98, 17.78, 20.37, 7.24
]

latency = np.array([
    3.83, 6.92, 9.25, 17.87, 1.90, 2.11, 2.18, 2.89, 3.14, 3.26, 
    1.08, 1.37, 7.64, 16.98, 3.71, 4.40
])

performance = np.array([
    87.76, 87.60, 86.12, 86.02, 92.22, 93.14, 90.08, 90.94, 89.17, 90.48, 
    90.72, 91.44, 65.84, 70.32, 89.32, 88.31
])

# Separate the models into three groups
resnet_indices = [0, 1, 2, 3, 4, 5, 6, 7]
vgg_indices = [8, 9, 10, 11]
mobilenet_indices = [12, 13, 14, 15]


def sort_by_latency(indices):
    return sorted(indices, key=lambda i: latency[i])

# Sort indices
sorted_resnet_indices = sort_by_latency(resnet_indices)
sorted_vgg_indices = sort_by_latency(vgg_indices)
sorted_mobilenet_indices = sort_by_latency(mobilenet_indices)


data = {
    "models": models,
    "latency": latency,
    "performance": performance,
    "params": params
}


# Redefine plot with scatter to create circular backgrounds of same color as lines
plt.figure(figsize=(5, 3))

# Define colors
resnet_color = 'blue'
vgg_color = 'green'
mobilenet_color = 'red'

# Reduce the size scale
size_scale = 10
highlight_circle_radius = 150  # This is the size for scatter

# Plotting ResNet models
resnet_x = data["latency"][sorted_resnet_indices]
resnet_y = data["performance"][sorted_resnet_indices]
resnet_sizes = data["params"][sorted_resnet_indices] * size_scale
plt.plot(resnet_x, resnet_y, 'o-', label='ResNet', color=resnet_color, alpha=0.7)

# Plotting VGG models
vgg_x = data["latency"][sorted_vgg_indices]
vgg_y = data["performance"][sorted_vgg_indices]
vgg_sizes = data["params"][sorted_vgg_indices] * size_scale
plt.plot(vgg_x, vgg_y, 'o-', label='VGG', color=vgg_color, alpha=0.7)

# Plotting MobileNet models
mobilenet_x = data["latency"][sorted_mobilenet_indices]
mobilenet_y = data["performance"][sorted_mobilenet_indices]
mobilenet_sizes = data["params"][sorted_mobilenet_indices] * size_scale
plt.plot(mobilenet_x, mobilenet_y, 'o-', label='MobileNet', color=mobilenet_color, alpha=0.7)

# Highlighting "self-growing" models using scatter with the same color as their respective lines
for i in range(len(data["models"])):
    if "自增长" in data["models"][i]:
        if "ResNet" in data["models"][i]:
            color = resnet_color
        elif "VGG" in data["models"][i]:
            color = vgg_color
        elif "MobileNet" in data["models"][i]:
            color = mobilenet_color
        
        plt.scatter(data["latency"][i], data["performance"][i], 
                    s=highlight_circle_radius, marker='o', color=color, alpha=0.3)

plt.xlabel('推理延迟 (ms)')
plt.ylabel('性能 (acc%)')
plt.title('Cifar10')
plt.ylim([55, 100])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./cifar10.png", dpi=200)
