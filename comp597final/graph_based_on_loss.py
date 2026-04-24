import matplotlib.pyplot as plt

paths = ['training_log.txt', 'frozen_training_log.txt']

with open(paths[1], 'r') as file:
    froze_data = file.read()
epochs, losses = [], []


for line in froze_data.splitlines():
    if line.startswith('epoch') or line.startswith('Training'):
        continue
    split = line.split(",")
    epochs.append(int(split[0].strip()))
    losses.append(float(split[1].strip()))

print("Epochs:", epochs)
print("Losses:", losses)
filename = 'graph'

plt.plot(epochs, losses, marker='o')
plt.title('ProxyNCA Training Loss (Frozen Model)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.annotate(f"Final Loss: {losses[-1]:.4f}", xy=(epochs[-1], losses[-1]), xytext=(epochs[-1]-5, losses[-1]+.25))
plt.show()