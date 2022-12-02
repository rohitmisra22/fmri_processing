import nibabel as nib
import numpy as np
import sys
from scipy.stats import t
from scipy.stats import norm  

def reshape_and_export(data, x, y, z, template, output_filename):
	volume = data.reshape(x, y, z)
	output_file = nib.Nifti1Image(volume, template.affine, template.header) # generate nifti image
	nib.save(output_file, output_filename + ".nii.gz")

#  python group_analysis.py <file list.txt> <output prefix>

nii_list_filename = sys.argv[1]
output_filename = sys.argv[2]

nii_list_file = open(nii_list_filename, 'r')
nii_list =  [line.rstrip("\r\n") for line in nii_list_file.readlines()]

# template nii file to extract dimensions
template_nii = nib.load(nii_list[0])
data = template_nii.get_fdata() 
template_header = template_nii.header
x_max = data.shape[0]
y_max = data.shape[1]
z_max = data.shape[2]
n_voxels = x_max * y_max * z_max

num_subjects = len(nii_list)
group_data = np.ndarray((n_voxels, num_subjects))

# for each cope file, flatten and append subject data to 2D matrix 
for index, subject_filename in enumerate(nii_list):
	subject_file = nib.load(subject_filename)
	subject_data = subject_file.get_fdata().reshape(n_voxels)
	group_data[:,index] = subject_data

mean_values = group_data.mean(axis = 1)  # mean cope for each voxel across all subjects
variance_values = np.var(group_data, axis = 1, ddof = 1) # # cope variance for each voxel across all subjects
SE_values = np.sqrt(variance_values/num_subjects) # # cope SE for each voxel across all subjects

dof = num_subjects - 1
t_stats = np.nan_to_num(np.divide(mean_values, SE_values)) # get t stats
z_stats = norm.isf(t.sf(t_stats, dof)) # get z stats

# export statistics as nii 
reshape_and_export(t_stats, x_max, y_max, z_max, template_nii, output_filename+".tstat")
reshape_and_export(z_stats, x_max, y_max, z_max, template_nii, output_filename+".zstat")





