import matplotlib.pyplot as plt
import csv

history_file = "models/unet3d-1581270355-history.csv"

epoch = []
dice_coefficient = []
loss = []
val_dice_coefficient = []
val_loss = []

with open(history_file, "r") as csvfile:
    plots = csv.reader(csvfile, delimiter=" ")
    first_line = True
    for row in plots:
        if first_line:
            first_line = False
            continue
        epoch.append(int(row[0]))
        dice_coefficient.append(float(row[1]))
        loss.append(float(row[2]))
        val_dice_coefficient.append(float(row[3]))
        val_loss.append(float(row[4]))

print(epoch, dice_coefficient, loss)

fig, ax2 = plt.subplots()
ax2.plot(epoch, loss, color="red", label="loss")
ax2.set_ylabel("loss", color="red")
ax2.set_xlabel("epoch")

ax1 = ax2.twinx()
ax1.plot(epoch, dice_coefficient, color="blue", label="dice coefficient")
ax1.set_ylabel("dice_coefficient", color="blue")
plt.xlim(0,10)

plt.savefig("docs/loss.png")
