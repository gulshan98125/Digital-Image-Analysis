import reduce as rc
import expand as ec
import laplacian as lp
import cv2
import numpy as np

#this is gaussian kernel
kernel  =     [ 
			[1,4,6,4,1], \
			[4,16,24,16,4], \
			[6,24,36,24,6], \
			[4,16,24,16,4], \
			[1,4,6,4,1]
		  ]
#kernel_multiplier 
k = 1/256.0

divide_factor = 1

def addition(image1,expanded_image2):
	rows1,cols1,_ = image1.shape
	rows2,cols2,_ = expanded_image2.shape
	extra_rows = rows2-rows1
	extra_cols = cols2-cols1
	if(extra_rows>0 and extra_cols>0):
		new_expanded_image2 = np.array(expanded_image2[:-extra_rows,:-extra_cols])
	elif (extra_rows>0 and extra_cols==0):
		new_expanded_image2 = np.array(expanded_image2[:-extra_rows,:])
	elif (extra_rows==0 and extra_cols>0):
		new_expanded_image2 = np.array(expanded_image2[:,:-extra_cols])
	else:
		new_expanded_image2 = np.array(expanded_image2[:,:])
	added = image1+new_expanded_image2
	return added



def reconstruction(input_arr):
	input_arr = input_arr*divide_factor
	input_arr = input_arr.tolist()
	reconstruction_arr  = []
	working_image = input_arr[len(input_arr)-1]
	for i in xrange(len(input_arr)-2,-1,-1):
		
		expanded = ec.expand(working_image,kernel,k)
		# cv2.imshow(str(i),expanded)
		working_image = addition(input_arr[i],expanded)
		reconstruction_arr.append(working_image)
	return reconstruction_arr

# image_arr = []
# for i in xrange(lp.length_of_array):
# 	image_arr.append(cv2.imread('../output/laplacian/'+str(i)+'.png',1))

output_array = reconstruction(np.array(lp.laplacian_arr))


for i in xrange(len(lp.reduced_arr)):
	cv2.imwrite('../output/reduced/'+str(i)+'.png',lp.reduced_arr[i])

for i in xrange(len(lp.expanded_arr)):
	cv2.imwrite('../output/expanded/'+str(i)+'.png',lp.expanded_arr[i])

for i in xrange(len(output_array)):
	cv2.imwrite('../output/reconstruction/'+str(i)+'.png',output_array[i])