import tensorflow
import nibabel

images_files = ["data/496/Shank_496.ibw.nii", "data/498/Shank_498.ibw.nii", "data/499/Shank_499.ibw.nii", "data/506/Shank_506.ibw.nii", "data/515/Shank_515.ibw.nii", "data/516/Shank_516.ibw.nii", "data/520/Shank_520.ibw.nii", "data/521/Shank_521.ibw.nii", "data/522/Shank_522.ibw.nii", "data/523/Shank_523.ibw.nii", "data/527/Shank_527.ibw.nii", "data/528/Shank_528.ibw.nii", "data/534/Shank_534.ibw.nii", "data/536/Shank_536.ibw.nii", "data/541/Shank_541.ibw.nii", "data/542/Shank_542.ibw.nii", "data/550/Shank_550c.ibw.nii", "data/nifti_em/MetOD1_Day2.ibw.nii", "data/nifti_em/MetOD1_Day30.ibw.nii", "data/nifti_em/MetOG2_Day15.ibw.nii", "data/nifti_em/MetOG2_Day24.ibw.nii"]

labels_files = list(map(lambda image: image.replace(".ibw", "-labels"), images_files))

nibabel.load("data/496/Shank_496.ibw.nii")
images = list(map(nibabel.load, images_files))
labels = list(map(nibabel.load, labels_files))

print("Data successfully loadded")
print(images)
print(labels)
