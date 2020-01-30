import data
from keras.models import load_model

_data_train, data_test, _label_train, label_test = data.load_data()
data_test = data_test.reshape((data_test.shape[0]*128, 128, 128, 1))
label_test = data_test.reshape((label_test.shape[0]*128, 128, 128, 1))

model = models.load("models/unet.h5")
model.evaluate(data_test, label_test)
