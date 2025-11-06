# Re-importing matplotlib after a reset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

# Redefining the data after reset

file_name = "c:/Users/Lenovo/OneDrive/毕业设计/研究/visualization/visualizer.txt"

with open(file_name, 'r') as file:
    # 读取文件全部内容到一个字符串
    file_content = file.read()

# Processing the data to extract epochs and test_acc values
lines = file_content.strip().split('\n')
epochs = []
test_accs = []

for line in lines:
    data_dict = eval(line)
    epochs.append(data_dict['epoch'])
    test_accs.append(data_dict['test_acc'])

# Plotting the relationship between epoch and test_acc
plt.figure(figsize=(5, 3)) 
plt.plot(epochs, test_accs, 'o-', markeredgecolor='k', markerfacecolor='r', markersize=5, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.tight_layout()

# Saving the plot as an SVG file
plt.savefig('visualization.svg')
plt.show()
