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
    "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "自增长ResNet 4-4-4", "自增长ResNet 3-3-4-3", 
    "自增长Bottleneck-ResNet 5-6-5", "自增长Bottleneck-ResNet 4-4-5-4", "VGG-11", "VGG-16", 
    "自增长VGG 4-5-5", "自增长VGG 4-4-5-4", "MobileNetV3-small", "MobileNetV3-large", 
    "自增长MobileNetV3 5-6-5", "自增长MobileNetV3 4-5-5-4"
]

architecture = [
    "2-2-2-2", "3-4-6-3", "3-4-6-3 bottleneck", "3-4-23-3 bottleneck", "4-4-4", "3-3-4-3", 
    "5-6-5", "4-4-5-4", "1-1-2-2-2", "2-2-3-3-3", "4-5-5", "4-4-5-4", "2-2-3-2-2", "2-2-3-4-2-2", 
    "5-6-5", "4-5-5-4"
]

params = np.array([
    11.69, 21.80, 25.56, 44.55, 6.60, 21.60, 0.87858, 3.16, 140.72, 146.22, 
    4.24, 14.74, 2.84, 5.74, 2.03, 2.95
])

compute = [
    27.34, 57.01, 64.27, 123.99, 46.53, 51.13, 5.63, 6.75, 15.64, 23.53, 
    27.95, 34.43, 0.37802, 0.74314, 2.17, 1.17
]

latency = np.array([
    3.80, 6.81, 11.95, 23.41, 2.90, 3.13, 4.20, 4.44, 2.68, 3.24, 
    1.82, 2.17, 8.17, 17.50, 6.41, 7.39
])

performance = np.array([
    80.23, 81.93, 80.77, 82.17, 82.18, 85.93, 73.35, 81.15, 78.83, 79.27, 
    82.00, 84.13, 73.43, 74.03, 81.30, 80.33
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
plt.title('Mini-Imagenet')
plt.ylim([72, 88])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./Miny.png", dpi=200)
