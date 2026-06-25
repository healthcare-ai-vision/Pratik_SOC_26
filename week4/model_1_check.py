import pandas as pd
import matplotlib.pyplot as plt

# Load your results
df = pd.read_csv('/home/pratik/Desktop/code/runs/classify/train/results.csv')

# Clean up column names (remove leading spaces)
df.columns = df.columns.str.strip()

# Plot Training vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train/loss'], label='Training Loss')
plt.plot(df['epoch'], df['val/loss'], label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
