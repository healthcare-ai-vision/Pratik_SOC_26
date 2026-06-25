import matplotlib.pyplot as plt
import numpy as np
import os
from ultralytics import YOLO

# 1. Load your model
model = YOLO('/home/pratik/Desktop/code/runs/classify/train/weights/best.pt')

# 2. Run validation
data_path = '/media/pratik/PENDRIVE/downloads/kaggle/test_al_kaggle_zip (1)/Combined Dataset'
metrics = model.val(data=data_path, split='test')

# 3. Extract confusion matrix and class names
cm = metrics.confusion_matrix.matrix
class_names = list(model.names.values())

# 4. Calculate total images per class for accuracy percentage
# We count files in each subdirectory of the 'test' folder
test_dir = os.path.join(data_path, 'test')
total_per_class = []
for name in class_names:
    count = len(os.listdir(os.path.join(test_dir, name)))
    total_per_class.append(count)

# 5. Calculate correct counts and percentages
correct_counts = np.diag(cm)[:len(class_names)]
percentages = (correct_counts / total_per_class) * 100

# 6. Print to Command Line
print("\n--- Per-Class Accuracy Report ---")
for i, name in enumerate(class_names):
    print(f"{name}: {correct_counts[i]} / {total_per_class[i]} correct ({percentages[i]:.2f}%)")

# 7. Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(class_names, percentages, color='skyblue', edgecolor='black')
plt.xlabel('Classes')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Percentage per Category')
plt.ylim(0, 100) # Set Y-axis to 100%
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('accuracy_percentages.png')
plt.show()
