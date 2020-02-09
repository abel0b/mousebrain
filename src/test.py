import data
import tensorflow
from tensorflow.keras.models import load_model
from network import dice_coefficient, dice_coefficient_loss
import matplotlib.pyplot

_data_train, data_test, _label_train, label_test = data.load_data()
data_test, label_test = data.fill_augment(data_test, label_test, 128)

dependencies = {
    "dice_coefficient": dice_coefficient,
    "dice_coefficient_loss": dice_coefficient_loss,
}
model = load_model("models/unet3d.h5", custom_objects=dependencies)
score = model.evaluate(data_test, label_test)
print(score)

prediction = model.predict(data_test)

expected_mask = label_test[64]
actual_mask = prediction[64]
matplotlib.pyplot.matshow(expected_mask.reshape((128, 128)))
matplotlib.pyplot.savefig("docs/expected_mask.png")
matplotlib.pyplot.matshow(actual_mask.reshape((128, 128)))
matplotlib.pyplot.savefig("docs/actual_mask.png")
