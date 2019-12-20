import numpy
import nibabel
from sklearn.model_selection import train_test_split

def load_data():
    data_files = ["data/496/Shank_496.ibw.nii", "data/499/Shank_499.ibw.nii", "data/506/Shank_506.ibw.nii", "data/515/Shank_515.ibw.nii", "data/520/Shank_520.ibw.nii", "data/521/Shank_521.ibw.nii", "data/522/Shank_522.ibw.nii", "data/527/Shank_527.ibw.nii", "data/528/Shank_528.ibw.nii", "data/534/Shank_534.ibw.nii", "data/536/Shank_536.ibw.nii", "data/541/Shank_541.ibw.nii", "data/542/Shank_542.ibw.nii", "data/550/Shank_550c.ibw.nii"] # "data/nifti_em/MetOD1_Day2.ibw.nii", "data/nifti_em/MetOD1_Day30.ibw.nii", "data/nifti_em/MetOG2_Day15.ibw.nii", "data/nifti_em/MetOG2_Day24.ibw.nii"]

    # TODO: add missing data

    labels_files = list(map(lambda image: image.replace(".ibw", "-labels"), data_files))

    def load_nii_file(filename):
        return nibabel.load(filename).get_data()

    data = numpy.array(list(map(load_nii_file, data_files)))
    labels = numpy.array(list(map(load_nii_file, labels_files)))
    print("Data successfully loadded")

    return train_test_split(data, labels, test_size=0.2)
