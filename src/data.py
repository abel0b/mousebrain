import numpy
import nibabel
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, random_rotation
import matplotlib.pyplot
import time
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa
from scipy.ndimage import rotate
from matplotlib.animation import FuncAnimation

def random_angle(rotation_range=4.0):
    return numpy.random.uniform(-rotation_range/2.0, rotation_range/2.0)

def threshold(val):
    return 1.0 if val>0.8 else 0.0

thresholdv = numpy.vectorize(threshold)

def augment(orig_data, orig_label):
    new_data, new_label = orig_data, orig_label

    # Flip image
    if numpy.random.randint(0,2) == 0:
        new_data = numpy.flip(orig_data, axis=2) 
        new_label = numpy.flip(orig_label, axis=2) 
    
    # Rotate image in all planes
    angle = random_angle()
    new_data = rotate(new_data, angle, axes=(0,1), reshape=False, mode="nearest")
    new_label = rotate(new_label, angle, axes=(0,1), reshape=False, mode="constant")
    
    angle = random_angle()
    new_data = rotate(new_data, angle, axes=(1,2), reshape=False, mode="nearest")
    new_label = rotate(new_label, angle, axes=(1,2), reshape=False, mode="constant")
    
    angle = random_angle()
    new_data = rotate(new_data, angle, axes=(0,2), reshape=False, mode="nearest")
    new_label = rotate(new_label, angle, axes=(0,2), reshape=False, mode="constant")
    
    return new_data, thresholdv(new_label)

def fill_augment(data, labels, size):
    new_data = []
    new_labels = []

    for i in range(size-data.shape[0]):
        orig = numpy.random.randint(0, data.shape[0])
        newx, newy = augment(data[orig], labels[orig])
        new_data.append(newx)
        new_labels.append(newy)

    new_data = numpy.array(new_data)
    new_labels = numpy.array(new_labels)

    return numpy.concatenate((data, new_data)), numpy.concatenate((labels, new_labels))

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

    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2)

    data_train = numpy.swapaxes(data_train,1,3).reshape((data_train.shape[0], 128, 128, 128, 1))
    data_test = numpy.swapaxes(data_test,1,3).reshape((data_test.shape[0], 128, 128, 128, 1))
    label_train = numpy.swapaxes(label_train,1,3).reshape((label_train.shape[0], 128, 128, 128, 1))
    label_test = numpy.swapaxes(label_test,1,3).reshape((label_test.shape[0], 128, 128, 128, 1))
    
    return data_train, data_test, label_train, label_test

class MouseBrainSequence(Sequence):
    def __init__(self, dataset, size, batch_size=32):
        self.data, self.labels = dataset
        self.batch_size = batch_size
        self.size = size
        self.seed = int(time.time())
        self.batch_shape = (batch_size, 128, 128, 128, 1)
        self.rotation_range = 4.0
    
    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, index):
        numpy.random.seed(self.seed+index)
        batch_data = []
        batch_labels = []
    
        for idx in range(self.batch_size):
            orig = numpy.random.randint(0, self.data.shape[0]) 
            new_data, new_label = augment(self.data[orig], self.labels[orig])
            batch_data.append(new_data)
            batch_labels.append(new_label)

        batch_data = numpy.array(batch_data).reshape(self.batch_shape)
        batch_labels = numpy.array(batch_labels).reshape(self.batch_shape)
        
        return batch_data, batch_labels

if __name__ == "__main__":
    data_train, data_test, label_train, label_test = load_data()

    example = data_train[1].reshape((128,128,128))
    example_label = label_train[1].reshape((128,128,128))

    example_flip = numpy.flip(example, axis=2)
    angle = 15.0
    example_rotate = rotate(example, angle, axes=(1,2), reshape=False, mode="nearest")

    matplotlib.pyplot.matshow(example[64])
    matplotlib.pyplot.savefig("docs/example.png")
    
    matplotlib.pyplot.matshow(example_flip[64])
    matplotlib.pyplot.savefig("docs/example_flip.png")
    
    matplotlib.pyplot.matshow(example_rotate[64])
    matplotlib.pyplot.savefig("docs/example_rotate.png")

    fig, ax = matplotlib.pyplot.subplots()
    full_example = numpy.concatenate((example, example_label), axis=2)
    ax.axis("off")
    ax.margins(0.0)
    def update(i):
        ax.matshow(full_example[i])
    anim = FuncAnimation(fig, update, frames=numpy.arange(0, 128), interval=80)
    anim.save("docs/example.gif", dpi=80, writer="imagemagick")

