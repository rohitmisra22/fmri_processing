import nibabel as nib
import numpy as np
import numpy.linalg as npl
import sys
from scipy.stats import gamma
from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt

# cmd line: python glm.py func.nii.gz ev.txt contrasts.txt output_name
# 

# step 1: parse arguments
# 	extract:
	# func file
	# EVs file
	# contrasts file. preferebly as matrix
	# func header info, TR, dims, dimension sizes

# step 2: make design matrix from EV file
	# extract square wave
	# convolve with double gamma HRF. (get hrf from scipy.stats)
	#  Double-Gamma HRF is a preset function which is a mixture of two Gamma functions - a standard
	#  positive function at normal lag, and a small, delayed, inverted Gamma, which attempts to model the late undershoot.

# step 3: for each voxel, find beta^ vector.

	# assume N = num voxels, T = num timepoints, K = num EVs

	# GLM: 
	#  	[Y] = [X][B] + [e]
	#  	Y: T x 1 (N)
	#	X: T x K
	#	B: K x 1 (N)
	#	e: T x 1 (N)
	# 
	# 	B^ = ((X'X)-1)X'Y : K x 1----------- for each voxel 
	# 	export beta values for each EV  as nii.gz image
	# 	find residual for each voxel and its variance estimate v^ = (r'r/T-K-1)

# 
# step 4: stat test for contrasts 
	# contrast matrix: C : K x c
	# D = C'B^ : c x N (COPE values for each voxel)

	# for each contrast:

		#  find t-value for each voxel: C'B^[contrast]/ sqrt(v^ C (X'X)-1 C'[contrast, contrast])
		#  export  nii.gz file for each contrast.

		#	once you have t-values for all voxels, convert to p value and then z values

		#  export z values for each contrast


def generate_design_matrix(ev_filename, num_volumes, TR, hrf_duration):
	# Generate Design Matrix from EV file 
	EVs =  np.genfromtxt(fname=ev_filename, delimiter="\t", skip_header=0, filling_values=0)  
	num_EVs = int(max(EVs[:,3]))
	design_matrix = np.zeros((num_volumes, 1 + num_EVs))
	design_matrix[:,0] = 1.0
	for line in range(EVs.shape[0]):
		onset = int(EVs[line, 0]/TR)
		end = int((EVs[line, 0] + EVs[line, 1])/TR) + 1
		design_matrix[onset:end, int(EVs[line,3])] = EVs[line, 2]
	# Convolve Blocked Design Matrix with Double Gamma HRF
	dg_hrf = generate_hrf(hrf_duration, TR)
	for ev in range(1, num_EVs + 1):
		design_matrix[:, ev] = np.convolve(design_matrix[:,ev], dg_hrf)[: -(len(dg_hrf) - 1)]
	#plt.imshow(design_matrix)
	#plt.show()
	return design_matrix

def generate_hrf(time_duration, TR):
	time_steps = np.arange(0, time_duration,  TR)
	gamma_1 = gamma.pdf(time_steps, 6)
	gamma_2 = gamma.pdf(time_steps, 16)
	hrf = gamma_1 - (1/6.0) * gamma_2
	return (hrf)/np.max(hrf)

def reshape_and_export(data, x, y, z, template, output_filename):
	volume = data.reshape(x, y, z)
	output_file = nib.Nifti1Image(volume, template.affine, template.header) # generate nifti image
	nib.save(output_file, output_filename + ".nii.gz")


########################## MAIN #############

#  python glm.py <functional file> <ev file> <contrast file> <output prefix>
func_filename = sys.argv[1]
ev_filename = sys.argv[2]
contrast_filename = sys.argv[3]
output_filename = sys.argv[4]

input_nii = nib.load(func_filename) 
data = input_nii.get_fdata() 

x_max = data.shape[0]
y_max = data.shape[1]
z_max = data.shape[2]
n_voxels = x_max * y_max * z_max
num_volumes  = data.shape[3]
TR = input_nii.header["pixdim"][4]

bold_time_series = data.reshape(n_voxels, num_volumes).T 	# Y matrix with all voxels

hrf_duration = 20 #seconds
design_matrix = generate_design_matrix(ev_filename, num_volumes, TR, hrf_duration)


################################## Determine Estimate for Beta ####################

beta_hat = np.dot(np.dot(npl.inv(np.dot(design_matrix.T, design_matrix)), design_matrix.T),bold_time_series)
# beta_hat = beta_hat[1:, :]

#################################### Generate Contrast Matrix ####################

contrast_temp = np.genfromtxt(fname=contrast_filename, delimiter="\t", skip_header=0, filling_values=0).T
contrast = np.zeros((contrast_temp.shape[0]+1, contrast_temp.shape[1]))
contrast[1:,:] = contrast_temp

cope_values = np.dot(contrast.T, beta_hat)

##################################### Variance of Residuals ####################

residuals = bold_time_series - np.dot(design_matrix, beta_hat)
dof = bold_time_series.shape[0] - npl.matrix_rank(design_matrix)

variance_hat = np.zeros((residuals.shape[1]))

for current_voxel in range(residuals.shape[1]):
	variance_hat[current_voxel] = np.dot(residuals[:,current_voxel].T, residuals[:,current_voxel])/dof

##################################### Statistics ###############################

t_stats = np.ndarray(cope_values.shape)
z_stats = np.ndarray(cope_values.shape)

for contrast_num in range(contrast.shape[1]):
	temp = np.dot(contrast[:,contrast_num].T, np.dot(npl.inv(np.dot(design_matrix.T, design_matrix)), contrast[:,contrast_num]))
	t_stats[contrast_num,:] = np.nan_to_num(np.divide(cope_values[contrast_num,:], np.sqrt(variance_hat * (temp))))
	z_stats[contrast_num,:] = norm.isf(t.sf(t_stats[contrast_num,:], dof))

########################## Exporting Files ####################################

for ev in range(beta_hat.shape[0]):
	filename_prefix = output_filename + '.pe' + str(ev+1)
	reshape_and_export(beta_hat[ev,:], x_max, y_max, z_max, input_nii, filename_prefix)

for con in range(cope_values.shape[0]):
	cope_filename_prefix = output_filename + '.cope' + str(con+1)
	tstat_filename_prefix = output_filename + '.tstat' + str(con+1)
	zstat_filename_prefix = output_filename + '.zstat' + str(con+1)
	reshape_and_export(cope_values[con,:], x_max, y_max, z_max, input_nii, cope_filename_prefix)	
	reshape_and_export(t_stats[con,:], x_max, y_max, z_max, input_nii, tstat_filename_prefix)	
	reshape_and_export(z_stats[con,:], x_max, y_max, z_max, input_nii, zstat_filename_prefix)	



# print("Done!!")



