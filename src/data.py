import numpy
import nibabel
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot

"""
Load dataset
"""
def load_data():
    data_files = ["data/496/Shank_496.ibw.nii", "data/499/Shank_499.ibw.nii", "data/506/Shank_506.ibw.nii", "data/515/Shank_515.ibw.nii", "data/520/Shank_520.ibw.nii", "data/521/Shank_521.ibw.nii", "data/522/Shank_522.ibw.nii", "data/527/Shank_527.ibw.nii", "data/528/Shank_528.ibw.nii", "data/534/Shank_534.ibw.nii", "data/536/Shank_536.ibw.nii", "data/541/Shank_541.ibw.nii", "data/542/Shank_542.ibw.nii", "data/550/Shank_550c.ibw.nii"] # "data/nifti_em/MetOD1_Day2.ibw.nii", "data/nifti_em/MetOD1_Day30.ibw.nii", "data/nifti_em/MetOG2_Day15.ibw.nii", "data/nifti_em/MetOG2_Day24.ibw.nii"]

    # TODO: add missing data

    labels_files = list(map(lambda image: image.replace(".ibw", "-labels"), data_files))

    def load_nii_file(filename):
        return nibabel.load(filename).get_fdata()

    data = numpy.array(list(map(load_nii_file, data_files)))
    labels = numpy.array(list(map(load_nii_file, labels_files)))
    print("Data successfully loadded")

    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2)

    data_train = data_train.reshape((data_train.shape[0]*128, 128, 128, 1))
    data_test = data_test.reshape((data_test.shape[0]*128, 128, 128, 1))
    label_train = label_train.reshape((label_train.shape[0]*128, 128, 128, 1))
    label_test = label_test.reshape((label_test.shape[0]*128, 128, 128, 1))

    return data_train, data_test, label_train, label_test

"""
Augment dataset
"""
def data_generator(data):
    data_train, data_test, label_train, label_test = data

    datagen_train = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen_train.fit(data_train)

    datagen_test = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen_test.fit(data_test)

    return datagen_train, datagen_test

if __name__ == '__main__':
    data_train, data_test, label_train, label_test = load_data()
    datagen_train, datagen_test = data_generator((data_train, data_test, label_train, label_test))

    for x_batch, y_batch in datagen_train.flow(data_train, label_train, batch_size=32):
        print(x_batch.shape, y_batch.shape)
