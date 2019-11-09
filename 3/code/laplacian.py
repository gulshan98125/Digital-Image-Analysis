import reduce as rc
import expand as ec
import cv2
import numpy as np

image = cv2.imread('../input/lena.png',1)
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

current_image = image
reduced_arr = [image]

while(current_image.shape[0]>20):
	current_image = rc.reduce(current_image,kernel,k)
	reduced_arr.append(current_image)

expanded_arr = []
for j in xrange(1,len(reduced_arr)):
	expanded_image = ec.expand(reduced_arr[j],kernel,k)
	expanded_arr.append(expanded_image)

def difference(G_image1,expanded_image2):
	rows1,cols1,_ = G_image1.shape
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
	# subtracted = cv2.subtract(G_image1,new_expanded_image2)
	subtracted = G_image1 - new_expanded_image2
	return subtracted



laplacian_arr = []
for i in xrange(len(expanded_arr)):
	output = difference(reduced_arr[i],expanded_arr[i])
	laplacian_arr.append(output/divide_factor)
laplacian_arr.append(reduced_arr[len(reduced_arr)-1]/divide_factor)

for k in range(len(laplacian_arr)):
	cv2.imwrite('../output/laplacian/'+str(k)+'.png',laplacian_arr[k])
length_of_array = len(laplacian_arr)