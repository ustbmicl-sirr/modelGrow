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
    "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "自增长ResNet 1-1-1", "自增长ResNet 1-1-1-1", 
    "自增长Bottleneck-ResNet 1-1-1", "自增长Bottleneck-ResNet 1-1-1-1", "VGG-11", "VGG-16", 
    "自增长VGG 1-1-1", "自增长VGG 1-1-1-1", "MobileNetV3-small", "MobileNetV3-large", 
    "自增长MobileNetV3 1-1-1", "自增长MobileNetV3 1-1-1-1"
]

architecture = [
    "2-2-2-2", "3-4-6-3", "3-4-6-3 bottleneck", "3-4-23-3 bottleneck", "1-1-1", "1-1-1-1", 
    "1-1-1", "1-1-1-1", "1-1-2-2-2", "2-2-3-3-3", "1-1-1", "1-1-1-1", "2-2-3-2-2", "2-2-3-4-2-2", 
    "1-1-1", "1-1-1-1"
]

params = np.array([
    11.68, 21.79, 25.55, 44.54, 0.70519, 2.18, 0.20143, 0.56911, 132.86, 138.36, 
    0.43707, 1.32, 2.84, 5.74, 0.32686, 0.33852
])

compute = [
    36.12, 73.97, 84.76, 160.94, 160.20, 183.88, 31.25, 37.15, 275.23, 435.66, 
    91.39, 105.61, 8.91, 17.71, 9.14, 3.69
]

latency = np.array([
    3.59, 6.46, 9.17, 17.24, 1.24, 1.55, 1.31, 2.03, 2.49, 3.16, 
    0.81, 1.10, 8.58, 16.46, 1.64, 2.20
])

performance = np.array([
    99.80, 99.84, 99.76, 99.76, 99.76, 99.84, 99.72, 99.76, 99.60, 99.48, 
    99.66, 99.78, 99.64, 99.56, 99.62, 99.64
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
plt.title('MNIST')
plt.ylim([99.4, 99.9])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./MNIST.png", dpi=200)
