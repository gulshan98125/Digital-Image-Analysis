import reduce as rc
import expand as ec
import cv2
import numpy as np

image1 = cv2.imread('../input/apple3.jpg',1)
image2 = cv2.imread('../input/orange3.jpg',1)
#mask's white portion is obtained from image2

divide_factor = 1

kernel  =     [ 
			[1,4,6,4,1], \
			[4,16,24,16,4], \
			[6,24,36,24,6], \
			[4,16,24,16,4], \
			[1,4,6,4,1]
		  ]

blur_kernel = [ 
			[1,2,1], \
			[2,4,2], \
			[1,2,1],
		  ]
#kernel_multiplier 
k = 1/256.0

blur_mult = 1/16.0

def getBlur(i,j,kernel,k,image):
	sumB,sumG,sumR = 0,0,0
	for m in xrange(-1,2):
		for n in xrange(-1,2):
			kvalue = kernel[m+1][n+1]
			img_value = image[i][j]
			sumB += k*(kvalue)*(img_value[0])
			sumG += k*(kvalue)*(img_value[1])
			sumR += k*(kvalue)*(img_value[2])
			# print(sum)
	# return np.uint8(sumB),np.uint8(sumG),np.uint8(sumR)
	return int(sumB),int(sumG),int(sumR)

def blur_image(img,kernel,k,points_array):
	rows,cols,_ = img.shape
	image_padded = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_REPLICATE)
	newimg = img.copy()
	new_img_tolist = newimg.tolist()
	for i in xrange(len(points_array)):
			row = points_array[i][0]
			col = points_array[i][1]
			new_img_tolist[row][col] = getBlur(row,col,kernel,k,image_padded)
	return np.array(new_img_tolist)


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
	working_image = input_arr[len(input_arr)-1]
	reconstruction_arr  = [working_image]
	for i in xrange(len(input_arr)-2,-1,-1):
		
		expanded = ec.expand(working_image,kernel,k)
		# cv2.imshow(str(i),expanded)
		working_image = addition(input_arr[i],expanded)
		reconstruction_arr.append(working_image)
	return reconstruction_arr

#white portion = image1, which is added to image2
def applyMask(img1,img2,mask):
	newimg1 = img1*(mask/255)
	mask_inv = 255-mask
	newimg2 = img2*(mask_inv/255)
	return newimg1+newimg2

current_image1 = image1
reduced_arr1 = [image1]

while(current_image1.shape[0]>20):
	current_image1 = rc.reduce(current_image1,kernel,k)
	reduced_arr1.append(current_image1)

expanded_arr1 = []
for j in xrange(1,len(reduced_arr1)):
	expanded_image = ec.expand(reduced_arr1[j],kernel,k)
	expanded_arr1.append(expanded_image)

laplacian_arr1 = []
for i in xrange(len(expanded_arr1)):
	output = difference(reduced_arr1[i],expanded_arr1[i])
	laplacian_arr1.append(output/divide_factor)
laplacian_arr1.append(reduced_arr1[len(reduced_arr1)-1]/divide_factor)

current_image2 = image2
reduced_arr2 = [image2]

while(current_image2.shape[0]>20):
	current_image2 = rc.reduce(current_image2,kernel,k)
	reduced_arr2.append(current_image2)

expanded_arr2 = []
for j in xrange(1,len(reduced_arr2)):
	expanded_image = ec.expand(reduced_arr2[j],kernel,k)
	expanded_arr2.append(expanded_image)

laplacian_arr2 = []
for i in xrange(len(expanded_arr2)):
	output = difference(reduced_arr2[i],expanded_arr2[i])
	laplacian_arr2.append(output/divide_factor)
laplacian_arr2.append(reduced_arr2[len(reduced_arr2)-1]/divide_factor)

new_laplacian_arr = []

for i in xrange(len(laplacian_arr1)):
	img1 = laplacian_arr1[i]
	img2 = laplacian_arr2[i]
	r,c,_ = img1.shape
	# img3 = np.hstack((img1[:,:(int(0.5*c))],img2[:,int(0.5*c):]))
	mask = cv2.imread('../input/mask/mask'+str(i)+'.png',1)
	img3 = applyMask(img2,img1,mask)
	reduced_mask = rc.reduce(mask,kernel,k)
	# reduced_mask = cv2.resize(mask,(r/2+r%2,c/2+c%2))
	new_mask = cv2.inRange(reduced_mask, (127, 0, 0), (255,255,255))
	cv2.imwrite('../input/mask/mask'+str(i+1)+'.png',new_mask)
	new_laplacian_arr.append(img3)


output_array = reconstruction(np.array(new_laplacian_arr))
index = len(output_array)-1
for k in range(index+1):
	cv2.imwrite('../output/mosaic/'+str(k)+'.png',output_array[k])

first_mask = cv2.imread('../input/mask/mask0.png',1)
non_processed_join = applyMask(image2,image1,first_mask)
cv2.imwrite('../output/mosaic/actual_join.png',non_processed_join)


#blurring the edge part in the end image using canny edge from the mask
# temp = cv2.imread('../input/mask/mask0.png', 0)
# edges = cv2.Canny(temp, 100, 255)
# indices = np.where(edges != [0])
# coordinates = zip(indices[0], indices[1])

# final_image_unblurred = output_array[index]
# final_image_blurred = blur_image(final_image_unblurred,blur_kernel,blur_mult,coordinates)
# cv2.imwrite('../output/mosaic/final'+'.png',final_image_blurred)

# cv2.waitKey(0)