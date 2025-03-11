import matplotlib.pyplot as plt


rounds = [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6,6,7,7,7]
epochs = [2, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4,5,2,3,4]
accuracy = [58.35, 70.82, 78.22, 81.82, 82.88, 69.34, 78.01, 84.14, 87.10, 87.53, 82.03, 
            89.01, 88.79, 90.27, 87.74, 89.64, 93.02, 92.18, 89.01, 92.60, 93.45,93.87,89.85,95.14,94.71]
loss = [0.94, 0.74, 0.61, 0.56, 0.57, 0.78, 0.58, 0.48, 0.42, 0.45, 0.49, 
        0.40, 0.38, 0.34, 0.36, 0.33, 0.25, 0.29, 0.32, 0.21, 0.22, 0.24, 0.29,0.21,0.21]
dice_coefficient = [0.47, 0.61, 0.70, 0.76, 0.80, 0.57, 0.69, 0.78, 0.84, 0.85, 0.75,
                    0.84, 0.87, 0.88, 0.81, 0.87, 0.90, 0.90, 0.84, 0.90, 0.91, 0.92, 0.87,0.93,0.93]


# Accuracy Plot (First Window)
plt.figure(figsize=(10, 5))
plt.plot(range(len(accuracy)), accuracy, marker='o', linestyle='-', color='blue', label='Accuracy')
plt.title("Accuracy Over Rounds & Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Rounds & Epochs (Index)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(range(len(rounds)), [f"R{r}-E{e}" for r, e in zip(rounds, epochs)], rotation=45, fontsize=10)
plt.savefig("accuracy_plot.png", bbox_inches='tight')  # Save the figure
plt.show()  # Show first window

# Loss Plot (Second Window)
plt.figure(figsize=(10, 5))
plt.plot(range(len(loss)), loss, marker='s', linestyle='-', color='red', label='Loss')
plt.title("Loss Over Rounds & Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Rounds & Epochs (Index)", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(range(len(rounds)), [f"R{r}-E{e}" for r, e in zip(rounds, epochs)], rotation=45, fontsize=10)
plt.savefig("loss_plot.png", bbox_inches='tight')  # Save the figure
plt.show()  # Show second window

# Dice Coefficient Plot (Third Window)
plt.figure(figsize=(10, 5))
plt.plot(range(len(dice_coefficient)), dice_coefficient, marker='^', linestyle='-', color='green', label='Dice Coefficient')
plt.title("Dice Coefficient Over Rounds & Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Rounds & Epochs (Index)", fontsize=12)
plt.ylabel("Dice Coefficient", fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(range(len(rounds)), [f"R{r}-E{e}" for r, e in zip(rounds, epochs)], rotation=45, fontsize=10)
plt.savefig("dice_coefficient_plot.png", bbox_inches='tight')  # Save the figure
plt.show()  # Show third window