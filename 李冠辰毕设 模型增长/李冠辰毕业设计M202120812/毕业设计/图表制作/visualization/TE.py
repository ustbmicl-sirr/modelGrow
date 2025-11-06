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

models = [
    "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "自增长ResNet 2-3-2", "自增长ResNet 2-2-2-2", 
    "自增长Bottleneck-ResNet 3-4-3", "自增长Bottleneck-ResNet 2-3-3-2", "VGG-11", "VGG-16", 
    "自增长VGG 3-3-3", "自增长VGG 2-3-3-3", "MobileNetV3-small", "MobileNetV3-large", 
    "自增长MobileNetV3 3-4-3", "自增长MobileNetV3 3-3-3-3"
]

architecture = [
    "2-2-2-2", "3-4-6-3", "3-4-6-3 bottleneck", "3-4-23-3 bottleneck", "2-3-2", "2-2-2-2", 
    "3-4-3", "2-3-3-2", "1-1-2-2-2", "2-2-3-3-3", "3-3-3", "2-3-3-3", "2-2-3-2-2", "2-2-3-4-2-2", 
    "3-4-3", "3-3-3-3"
]

params = np.array([
    11.69, 21.80, 25.56, 44.55, 1.41, 3.90, 0.27766, 0.70414, 140.72, 146.22, 
    0.97447, 3.01, 2.84, 5.74, 0.90318, 0.85529
])

compute = [
    27.34, 57.01, 64.27, 123.99, 16.74, 16.74, 2.57, 2.55, 15.64, 23.53, 
    11.28, 11.04, 0.37802, 0.74314, 1.25, 0.43985
]

latency = np.array([
    3.80, 6.81, 11.95, 23.41, 2.35, 2.70, 3.17, 3.30, 2.68, 3.24, 
    1.54, 1.90, 8.17, 17.50, 4.66, 4.97
])

performance = np.array([
    93.99, 94.30, 92.77, 95.72, 93.68, 95.47, 92.83, 94.40, 94.40, 94.71, 
    94.05, 95.42, 91.04, 91.96, 95.57, 95.98
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
plt.title('ImagenetTE')
plt.ylim([90, 97])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./TE.png", dpi=200)
