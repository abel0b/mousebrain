import tensorflow
import data
import unet
import matplotlib.pyplot
from tensorflow.keras import backend as K

data_train, data_test, label_train, label_test = data.load_data()

print(data_train.shape)
print(label_train.shape)

data_train = data_train.reshape((data_train.shape[0]*128, 128, 128, 1))
data_test = data_test.reshape((data_test.shape[0]*128, 128, 128, 1))
label_train = label_train.reshape((label_train.shape[0]*128, 128, 128, 1))
label_test = label_test.reshape((label_test.shape[0]*128, 128, 128, 1))

print(data_train.shape)
print(label_train.shape)

# matplotlib.pyplot.matshow(data_train[60].reshape((128, 128)))
# matplotlib.pyplot.matshow(label_train[60].reshape((128, 128)))
# matplotlib.pyplot.show()

EPOCHS = 5
VAL_SUBSPLITS = 4
BATCH_SIZE = 64
VALIDATION_STEPS = BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = data_train.shape[0]

model = unet.unet2d((128,128,1))

model_history = model.fit(
    data_train,
    label_train,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=(data_test, label_test),
)

model.save("models/unet.h5")
