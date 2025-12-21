import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("training_log.csv")

epochs = df["epoch"]
loss = df["loss"]
acc = df["train_accuracy"]

fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)

ax1.plot(epochs, loss, label="Loss")
ax2.plot(epochs, acc, label="Train Accuracy")

ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.set_title("Loss")
ax2.set_xlabel("epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
