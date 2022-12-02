########################################################################
########################################################################

# This code has been created by 
#	Rohit Misra
#	Entry Number = 2021EEY7520
#	Course = COL786
# 	Date = 21st March 2022
# Submitted towards the completion of Assignment 4 for the course COL786

########################################################################
########################################################################

import nibabel as nib
import numpy as np
import sys


##################### FUNCTIONS ###########################

def is_slice_time_valid(slice_times, target_time, TR, num_slices):

# This function checks whether the provided slice time parameters are valid.
# It performs 3 checks:
#	1. whether target time is within 0 and TR
#	2. whether slice acquisition times are within 0 and TR
#	3. whether number of lines in the slice time acquisition file are same as number of slices in image
#
# The function returns False if the data dissatisfies any one of the criteria. It returns True otherwise.
#
# Arguments: 
# 	slice_times = array containing slice acquisition times
# 	target_time = target time
#	TR = Repitition time TR 
#	num_slices = number of slices in the fMRI image

	if target_time<0 or target_time>TR:
		return False

	for slice_time in slice_times:
		if slice_time<0 or slice_time>TR:
			return False

	if len(slice_times) != num_slices:
		return False

	return True


def slice_linear_interpolate(y1, y2, x1, x2, x_target):
	
# This function performs linear interpolation to find the y-value for an x-value given 2 points (x1,y1) and (x2,y2)
# Arguments:
#	x1 = x value for first point
#	y1 = y value for first point
#	x2 = x value for second point
#	y2 = y value for second point
#	x_target = x value for point with unknown y value
# Returns: y value for given target x value

	slope = (y2 - y1)/(x2 - x1)
	return y1 + slope*(x_target - x1)


def slice_time_correct(data, TR, slice_times, target_time):

# This function performs slice time correction using the following arguments:
#	data = 4D array with input BOLD data
#	TR = Repitition time TR
#	slice_times = array containing acquisition times for all slices
#	target_time = Target time
#
# The function returns a 4D array (resultant) containing slice-time corrected data

	data_dimensions= data.shape
	num_slices = data_dimensions[2] # number of slices in data 
	num_volumes = data_dimensions[3] # number of volumes in data

	resultant = np.ndarray(data_dimensions)
	resultant[:,:,:,-1] = data[:,:,:,-1] # As per problem statement, last volume is assumed constant and copied to output without change

	# for all volumes
	for current_vol_index in range(1, num_volumes-1):
		# for all slices per volume
		for current_slice_index in range(0,num_slices):

			current_slice = data[:,:, current_slice_index, current_vol_index]
			current_slice_time = slice_times[current_slice_index]

			# If slice time is greater than target time, interpolate using previous volume
			if current_slice_time >= target_time:
				interpolator_slice = data[:,:, current_slice_index, current_vol_index -1]
				interpolator_slice_time = slice_times[current_slice_index] - TR
				corrected_slice = slice_linear_interpolate(current_slice, interpolator_slice, current_slice_time, interpolator_slice_time, target_time)
			else:
			# If slice time is less than target time, interpolate using next volume
				interpolator_slice = data[:,:, current_slice_index, current_vol_index +1]
				interpolator_slice_time = slice_times[current_slice_index] + TR
				corrected_slice = slice_linear_interpolate(current_slice, interpolator_slice, current_slice_time, interpolator_slice_time, target_time)
			
			# add corercted slice to output 4D array
			resultant[:,:,current_slice_index, current_vol_index] = corrected_slice

	# Interpolation for first volume uses the 2nd volume regardless of value of slice time. 
	# For slice time > target time, value is interpolated backwards.
	current_vol_index = 0;
	# for all slices in first volume
	for current_slice_index in range(0,num_slices):
		
		current_slice = data[:,:, current_slice_index, current_vol_index]
		current_slice_time = slice_times[current_slice_index]

		interpolator_slice = data[:,:, current_slice_index, current_vol_index +1]
		interpolator_slice_time = slice_times[current_slice_index] +  TR
		corrected_slice = slice_linear_interpolate(current_slice, interpolator_slice, current_slice_time, interpolator_slice_time, target_time)

		resultant[:,:,current_slice_index, current_vol_index] = corrected_slice

	return resultant


## FUNCTION FOR TEMPORAL FILTERING
def temporal_filtering(data, sampling_time, cutoff_low_s, cutoff_high_s):
	
	# This function performs temporal filtering using a band-pass filter with given cutoff frequencies.
	# Arguments:
	#	data = 4D array containing input BOLD data
	#	sampling_time = sampling time for fMRI data is the TR
	#	cutoff_low_s = the cutoff for the low pass filter, expressed in seconds
	#	cutoff_high_s = the cutoff for the high pass filter, expressed in seconds

	data_dimensions= data.shape
	x_dim = data_dimensions[0]
	y_dim = data_dimensions[1]
	z_dim = data_dimensions[2]
	num_volumes  = data_dimensions[3]

	## FIND CUTOFF FREQUENCIES AND SAMPING FREQ
	sampling_freq = 1/sampling_time
	cutoff_low_hz = min ( 1/cutoff_low_s, sampling_freq/2) # cutoff frequency for low pass filter. Maximum value can be (sampling frequency/2)
	cutoff_high_hz = 1/cutoff_high_s # cutoff frequency for high pass filter
	freq_step = sampling_freq/num_volumes # frequency resolution
	cutoff_low_index = int(cutoff_low_hz/freq_step) # cutoff frequencty for LPF expressed as array index
	cutoff_high_index = int(cutoff_high_hz/freq_step) # cutoff frequencty for HPF expressed as array index

	# empty array for result
	resultant = np.ndarray(data_dimensions)

	# extract time series for each voxel and perform temporal filtering on it using FFT
	for x in range(0,x_dim):
		for y in range(0,y_dim):
			for z in range(0,z_dim):
				voxel_time_series = data[x,y,z,:] # time series of voxel
				dft = np.fft.fft(voxel_time_series) # FFT of voxel time series
				# remove frequecies outside the provided band of filter
				for f in range(0,len(dft)): 
					if (f<cutoff_high_index) or (f>cutoff_low_index and f<(len(dft)-cutoff_low_index)) or (f>(len(dft) - cutoff_high_index)):
						dft[f] = 0
				# IFFT of filtered FFT gives filtered time series
				idft = abs(np.fft.ifft(dft)) + np.mean(voxel_time_series)*np.ones(len(voxel_time_series))
				resultant[x,y,z,:] = idft
	return resultant


# FUNCTIONS FOR SPATIAL SMOOTHING
def gaussian_kernel_generator(kernel_size, fwhm, pix_dim):

# This function returns a 1D , 0 mean, Gaussian kernel of given size and FWHM value
# Arguments:
#	kernel_size = size of gaussian kernel required
#	fwhm = Full Width Half Maximum value for the desired Gaussian kernel 
#	pix_dim = size of the voxel in mm 

	ind_var = np.arange(-kernel_size, kernel_size + 1)
	sigma = fwhm / np.sqrt(8 * np.log(2)) / (float(pix_dim)) # calculating standard deviation using fwhm 
	kernel = np.exp(-(ind_var**2) / (2 * (sigma**2))) # generating values for gaussian kernel
	return kernel/sum(kernel) # returning normalised kernel 

def convolve_1D(x, y):
# This function performs 1D convolution of the vectors x and y.
	x_len = len(x) # kernel
	y_len = len(y) # data

	y_pad = np.concatenate((np.flip(y), y, np.flip(y))) # padding mirrored values to data
	x_conv_y = np.zeros(y_len)
	for i in range(0, y_len):
		x_conv_y[i] = np.dot(x, y_pad[y_len-(x_len//2)+i : y_len+(1 + (x_len//2))+i]) 
	return x_conv_y


def spatial_smoothing(data, fwhm, x_pix_dim, y_pix_dim, z_pix_dim):
# This function performs spatial smoothing using cumulative 1D convolutions with a Gaussian kernel
# Arguments:
#	data = 4D array containing BOLD data
#	fwhm = Full Width Half Maximum value for the desired Gaussian kernel 
#	x_pix_dim = size of voxel in x direction
#	y_pix_dim = size of voxel in y direction
#	z_pix_dim = size of voxel in z direction

	data_dimensions= data.shape
	x_max = data_dimensions[0]
	y_max = data_dimensions[1]
	z_max = data_dimensions[2]
	num_volumes  = data_dimensions[3]

	resultant = np.ndarray(data_dimensions)

	# Iterate over all volumes
	for volume_number in range(0,num_volumes):

		# 1D convolution in x-direction
		kernel_x = gaussian_kernel_generator(x_max, fwhm, x_pix_dim)
		conv_1D_x = np.zeros([x_max, y_max, z_max])
		for y in range(0,y_max):
			for z in range(0,z_max):
				conv_1D_x[:,y,z] = convolve_1D(kernel_x, data[:,y,z,volume_number])

		# 1D convolution in y-direction
		kernel_y = gaussian_kernel_generator(y_max, fwhm, y_pix_dim)
		conv_1D_y = np.zeros([x_max, y_max, z_max])
		for x in range(0,x_max):
			for z in range(0,z_max):
				conv_1D_y[x,:,z] = convolve_1D(kernel_y, conv_1D_x[x,:,z])		


		# 1D convolution in z-direction
		kernel_z = gaussian_kernel_generator(z_max, fwhm, z_pix_dim)
		conv_1D_z = np.zeros([x_max, y_max, z_max])
		for x in range(0,x_max):
			for y in range(0,y_max):
				conv_1D_z[x,y,:] = convolve_1D(kernel_z, conv_1D_y[x,y,:])

		resultant[:,:,:,volume_number] = conv_1D_z

	return resultant

############################# MAIN ############################

# INITIAL INFORMATION EXTRACTION FROM ARGS
input_filename = sys.argv[sys.argv.index('-i')+1] # file name of input fMRI file
output_filename = sys.argv[sys.argv.index('-o')+1] # file name of output file

# load fMRI data and extract information from header
input_nii = nib.load(input_filename) 
working_data = input_nii.get_fdata() # create a working array that stores pre-processed data
input_dtype = working_data.dtype # store input data type
working_data = working_data.astype('float64') 

x_pix_dim = input_nii.header["pixdim"][1]
y_pix_dim = input_nii.header["pixdim"][2]
z_pix_dim = input_nii.header["pixdim"][3] 
TR = input_nii.header["pixdim"][4] ## CHECK THIS ONCE
num_slices = input_nii.header["dim"][3]


# IF SLICE TIME CORRECTION IS REQUIRED
if '-tc' in sys.argv:
	# extract argument data
	target_time  =  float(sys.argv[sys.argv.index('-tc')+1])/1000.0
	slice_time_acq_filename =  sys.argv[sys.argv.index('-tc')+2]
	
	# extract slice acquisition times from slice time acquisition file
	slice_time_file = open(slice_time_acq_filename, 'r')
	slice_times_data = slice_time_file.readlines()
	slice_times = [float(line.rstrip("\r\n"))/1000.0 for line in slice_times_data]
	slice_time_file.close()

	# check if slice time data is valid
	st_error = not (is_slice_time_valid(slice_times, target_time, TR, num_slices)) # this guy checks 1) slc_time = [0,tr], target time = [0,tr], lines in slc time acq = num slices
	if st_error:
		# print error if slice time data invalid
		with open(output_filename + '.txt', 'w') as file:
			file.write("SLICE TIME CORRECTION FAILURE")
		sys.exit()
	else:
		# perform slice time correction
		working_data = slice_time_correct(working_data, TR, slice_times, target_time) # this guy takes input nii file, st_acq file and t time. gives st corrected output
		# print success message to file
		with open(output_filename + '.txt', 'w') as file:
					file.write("SLICE TIME CORRECTION SUCCESS")

# IF TEMPORAL FILTERING IS REQURIED
if '-tf' in sys.argv:
	# parse arguments for temporal filtering
	cutoff_high_s = float(sys.argv[sys.argv.index('-tf')+1])
	cutoff_low_s = float(sys.argv[sys.argv.index('-tf')+2])
	# perform temporal filtering
	working_data = temporal_filtering(working_data, TR, cutoff_low_s, cutoff_high_s) # this guy applies temporal filering to each voxel time series

# IF SPATIAL SMOOTHING IS REQUIRED    
if '-sm' in sys.argv:
	# parse arguments for spatial smoothing
	fwhm = float(sys.argv[sys.argv.index('-sm')+1])
	# perform spatial smoothing
	working_data = spatial_smoothing(working_data, fwhm, x_pix_dim, y_pix_dim, z_pix_dim) # this guy applies gaussian smoothing with given fwhm to data



# GENERATING OUTPUT NIFTI IMAGE

working_data = working_data.astype(input_dtype) # convert data-type to match input file
output_file = nib.Nifti1Image(working_data, input_nii.affine, input_nii.header) # generate nifti image
nib.save(output_file, output_filename + ".nii.gz") # save output  file
