import tensorflow
import data
import unet
import matplotlib.pyplot
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import plot_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--plot-model", help="Plot model to image", action="store_true")
parser.add_argument("--epochs", help="Number of epochs")
parser.parse_args()

data_train, data_test, label_train, label_test = data.load_data()

print(data_train.shape)
print(label_train.shape)

# matplotlib.pyplot.matshow(data_train[60].reshape((128, 128)))
# matplotlib.pyplot.matshow(label_train[60].reshape((128, 128)))
# matplotlib.pyplot.show()

# model = unet.unet2d((128,128,1))
model = load_model("models/unet2d.h5")

# plot_model(model, to_file='docs/model.png')

# data_train = data_train[0:128]
# label_train = data_train[0:128]

EPOCHS = 1
VAL_SUBSPLITS = 4
BATCH_SIZE = 64
VALIDATION_STEPS = BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = data_train.shape[0]

model_history = model.fit(
    data_train,
    label_train,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=(data_test, label_test),
)

model.save("models/unet2d.h5")
