import numpy as np
import cv2

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

def getSum(i,j,kernel,k,image_padded,padding):
	sumB,sumG,sumR = 0,0,0
	for m in xrange(-2,3):
		for n in xrange(-2,3):
			kvalue = kernel[m+2][n+2]
			if(((i-m)%2 == 0) and ((j-n)%2 == 0) and ((i-m)>= 0) and ((j-n) >= 0)):
				img_value = image_padded[(i-m)/2+padding][(j-n)/2+padding]
				sumB += 4*k*(kvalue)*(img_value[0])
				sumG += 4*k*(kvalue)*(img_value[1])
				sumR += 4*k*(kvalue)*(img_value[2])
			# print(sum)
	# return np.uint8(sumB),np.uint8(sumG),np.uint8(sumR)
	return int(sumB),int(sumG),int(sumR)


def expand(input_image,kernel,k):
	
	kW = len(kernel[0])
	pad = (kW - 1) / 2
	image = input_image
	rows,cols,_ = image.shape
	# image_padded = np.array([[[0,0,0]]*(cols+4)]*(rows+4))
	# image_padded = np.array(image_padded).reshape((rows+4,cols+4))
	# image_padded = np.zeros((rows + 4, cols + 4))
	# image_padded[2:-2, 2:-2] = image
	# if(rows%2 != 0):
	# 	image = cv2.copyMakeBorder(image, 0,1,0,0, cv2.BORDER_REPLICATE)
	# if(cols%2 != 0):
	# 	image = cv2.copyMakeBorder(image, 0,0,0,1, cv2.BORDER_REPLICATE)

	rows_updated,cols_updated,_ = image.shape

	image_padded = cv2.copyMakeBorder(image, pad,pad,pad,pad, cv2.BORDER_REPLICATE)
	image_padded = image_padded.tolist()

	newImg = np.zeros((2*rows_updated, 2*cols_updated))
	newImg = newImg.tolist()
	for i in xrange(0,2*rows_updated):
		for j in xrange(0,2*cols_updated):
			sum = getSum(i,j,kernel,k,image_padded,pad)
			newImg[i][j] = [sum[0],sum[1],sum[2]]
			# print(sum[0],sum[1],sum[2])
	# print(np.array(newImg))
	return np.array(newImg)

def expand2(input_image,kernel,k,times):
	current_image = input_image
	for i in range(times):
		current_image = expand(current_image,kernel,k)

# counter = 0
# while(counter<4):
# 	image = cv2.imread('../reduced/'+str(counter)+'.jpg',1)
# 	current_image = image
# 	out_arr = [current_image]
# 	current_image = expand(current_image,kernel,k)
# 	out_arr.append(current_image)
# 	cv2.imwrite('../output/expanded/'+str(counter)+'.jpg',current_image)
# 	counter +=1
# out = expand(current_image,kernel,k)
# cv2.imwrite('expanded.jpg',out)