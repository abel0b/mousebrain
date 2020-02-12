import tensorflow
import data
import network
import matplotlib.pyplot
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load-model", help="Load a model")
parser.parse_args()
args = parser.parse_args()

epochs = 10
size_train = 128
batch_size = 4
step_per_epochs = size_train // batch_size

data_train, data_test, label_train, label_test = data.load_data()
train_sequence = data.MouseBrainSequence((data_train, label_train), size=size_train, batch_size=batch_size)
data_test, label_test = data.fill_augment(data_test, label_test, batch_size)


if args.load_model:
    model_file = args.load_model
    dependencies = {
        "dice_coefficient": network.dice_coefficient,
        "dice_coefficient_loss": network.dice_coefficient_loss
    }
    model = load_model("{}.h5".format(model_file), custom_objects=dependencies)
else:
    model_file = "models/unet3d-{}".format(int(time.time()))
    model = network.unet_model_3d((128,128,128,1), depth=2)

model.summary()


checkpoint = ModelCheckpoint("{}.h5".format(model_file), save_best_only=True)
csv_logger = CSVLogger("{}-history.csv".format(model_file), append=True, separator=" ")

model_history = model.fit(
    train_sequence,
    epochs=epochs,
    steps_per_epoch=step_per_epochs,
    validation_steps=1,
    validation_data=(data_test, label_test),
    callbacks=[csv_logger, checkpoint],
)
