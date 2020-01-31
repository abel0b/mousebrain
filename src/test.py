import data
import tensorflow
from tensorflow.keras.models import load_model
from metrics import dice
import matplotlib.pyplot

_data_train, data_test, _label_train, label_test = data.load_data()
data_test = data_test.reshape((data_test.shape[0]*128, 128, 128, 1))
label_test = label_test.reshape((label_test.shape[0]*128, 128, 128, 1))

# def create_mask(pred_mask):
#     pred_mask =
#     return pred_mask

dependencies = {
    'dice': dice
}

model = load_model("models/unet2d.h5", custom_objects=dependencies)
score = model.evaluate(data_test, label_test)
print(score)

prediction = model.predict(data_test)

expected_mask = label_test[64]
actual_mask = prediction[64]
matplotlib.pyplot.matshow(expected_mask.reshape((128, 128)))
matplotlib.pyplot.savefig("docs/expected_mask.png")
matplotlib.pyplot.matshow(actual_mask.reshape((128, 128)))
matplotlib.pyplot.savefig("docs/actual_mask.png")
