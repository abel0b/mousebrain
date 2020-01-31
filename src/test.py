import data
from keras.models import load_model

_data_train, data_test, _label_train, label_test = data.load_data()
data_test = data_test.reshape((data_test.shape[0]*128, 128, 128, 1))
label_test = data_test.reshape((label_test.shape[0]*128, 128, 128, 1))

data_test = data_test.reshape((data_test.shape[0]*128, 128, 128, 1))
label_test = label_test.reshape((label_test.shape[0]*128, 128, 128, 1))

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

model = models.load("models/unet2d.h5")
score = model.evaluate(data_test, label_test)
print(score)

prediction = model.predict(data_train)

expected_mask = label_train[60]
actual_mask = create_mask(prediction[60])
matplotlib.pyplot.matshow(expected_mask.reshape((128, 128)))
matplotlib.pyplot.matshow(actual_mask.reshape((128, 128)))
matplotlib.pyplot.show()
