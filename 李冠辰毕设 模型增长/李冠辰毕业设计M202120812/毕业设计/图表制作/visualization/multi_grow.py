import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Update matplotlib settings to use SimSun font for Chinese characters
config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# Define the corrected data for plotting
models = [
    "RandGrow-ResNet", "RandGrow-ResNet", "RandGrow-Bottleneck-ResNet", "RandGrow-Bottleneck-ResNet",
    "SplitGrow-ResNet", "SplitGrow-ResNet", "SplitGrow-Bottleneck-ResNet", "SplitGrow-Bottleneck-ResNet",
    "MoGrow-ResNet", "MoGrow-ResNet", "MoGrow-Bottleneck-ResNet", "MoGrow-Bottleneck-ResNet",
    "AdaGrow-ResNet", "AdaGrowResNet", "AdaGrow-Bottleneck-ResNet", "AdaGrow-Bottleneck-ResNet",
    "RandGrow-VGG", "RandGrow-VGG", "SplitGrow-VGG", "SplitGrow-VGG",
    "MoGrow-VGG", "MoGrow-VGG", "AdaGrow-VGG", "AdaGrow-VGG",
    "RandGrow-MobileNetV3", "RandGrow-MobileNetV3", "SplitGrow-MobileNetV3", "SplitGrow-MobileNetV3",
    "MoGrow-MobileNetV3", "MoGrow-MobileNetV3", "AdaGrow-MobileNetV3", "AdaGrow-MobileNetV3"
]

architecture = [
    "1-3-2", "1-2-2-2", "2-3-2", "2-2-3-2", 
    "1-2-2", "1-2-2-1", "2-2-2", "2-2-2-2", 
    "2-2-2", "1-2-2-2", "2-2-2", "2-3-3-2", 
    "1-2-2", "1-2-2-1", "2-2-2", "2-2-2-2", 
    "2-3-2", "2-2-2-2", "1-2-2", "1-1-2-2", 
    "2-2-2", "2-2-2-2", "1-2-2", "1-1-2-2", 
    "3-4-3", "3-3-4-2", "2-3-3", "2-2-3-2", 
    "3-3-3", "3-3-3-2", "2-3-3", "2-2-3-2"
]

params = np.array([
    3.69, 14.02, 0.57639, 2.37, 
    2.33, 5.291, 0.43547, 1.81, 
    3.48, 14.03, 0.55908, 2.39, 
    1.17, 2.65, 0.23507, 0.67321, 
    2.07, 7.82, 1.34, 4.12, 
    1.92, 7.83, 0.66929, 2.06, 
    1.27, 1.82, 1.07, 0.93177, 
    1.20, 1.67, 0.8114, 0.67052
])

compute = np.array([
    492.90, 587.34, 72.42, 100.28, 
    443.58, 490.90, 71.21, 120.21, 
    494.73, 588.78, 68.03, 105.19, 
    222.99, 246.66, 41.09, 48.14, 
    304.16, 360.85, 244.35, 249.01, 
    367.32, 361.83, 123.37, 125.72, 
    28.46, 16.86, 30.16, 11.59, 
    26.51, 15.86, 20.37, 7.24
])

latency = np.array([
    3.28, 5.49, 4.12, 5.22, 
    2.8, 3.37, 2.84, 4.77, 
    2.09, 2.28, 2.22, 3.15, 
    1.9, 2.11, 2.18, 2.89, 
    2.06, 2.51, 1.77, 2.17, 
    1.06, 1.95, 1.08, 1.37, 
    7.05, 8.31, 5.02, 5.69, 
    4.11, 4.85, 3.71, 4.4
])

performance = np.array([
    89.76, 89.68, 89.48, 90.28, 
    91.33, 91.47, 90.52, 91.3, 
    90.41, 90.88, 88.05, 90.71, 
    92.22, 93.14, 90.08, 90.94, 
    89.52, 90.08, 90.01, 90.48, 
    88.33, 89.65, 90.72, 91.44, 
    85.52, 86.52, 84.59, 85.1, 
    87.03, 87.6, 89.32, 88.31
])

# Create data dictionary
data = {
    "models": models,
    "architecture": architecture,
    "params": params,
    "compute": compute,
    "latency": latency,
    "performance": performance
}

# Separate indices for different model types and architectures
# resnet_indices = [i for i, model in enumerate(models) if "ResNet" in model and "Bottleneck" not in model]
# resnet_indices = [i for i, model in enumerate(models) if "Bottleneck" in model]
# resnet_indices = [i for i, model in enumerate(models) if "VGG" in model]
resnet_indices = [i for i, model in enumerate(models) if "MobileNet" in model]

# Define colors for different growth methods
randgrow_color = 'blue'
splitgrow_color = 'green'
mogrow_color = 'purple'
adagrow_color = 'red'

# Size scale for scatter plot
size_scale = 50

# Create subplots for each model type
fig, ax = plt.subplots(figsize=(4, 3))

# Plot for ResNet models with lines connecting points of the same growth method
for growth_method, color in [('RandGrow', randgrow_color), ('SplitGrow', splitgrow_color), ('MoGrow', mogrow_color), ('AdaGrow', adagrow_color)]:
    indices = [i for i in resnet_indices if growth_method in models[i]]
    ax.plot(latency[indices], performance[indices], 'o-', color=color, alpha=0.7, label=growth_method)
    ax.scatter(latency[indices], performance[indices], s=params[indices] * size_scale, edgecolors='w', linewidth=2, facecolor=color, alpha=0.7)

ax.legend(loc='best')
ax.grid(True)
ax.set_ylim([84, 90])
# plt.show()
plt.savefig("./multi-mob.png", dpi=200)
