import tensorflow
import data

data_train, data_test, label_train, label_test = data.load_data()

print(data_train.shape)
print(label_train.shape)

print(data_train[0])
print(label_train[0])
