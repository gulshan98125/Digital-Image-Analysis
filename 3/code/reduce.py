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

#gives the value at (i,j) when kernel is applied there
def getSum(i,j,kernel,k,image_padded,padding):
	sumB,sumG,sumR = 0,0,0
	for m in xrange(-2,3):
		for n in xrange(-2,3):
			# if(2*i+m>=0 and 2*j+n>=0):
			kvalue = kernel[m+2][n+2]
			img_value = image_padded[2*i+m+padding][2*j+n+padding]
			sumB += k*(kvalue)*(img_value[0])
			sumG += k*(kvalue)*(img_value[1])
			sumR += k*(kvalue)*(img_value[2])
			# print(sum)
	return int(sumB),int(sumG),int(sumR)

#the main reduce function
def reduce(input_image,kernel,k):
	
	kW = len(kernel[0])
	pad = (kW - 1) / 2
	image = input_image
	rows,cols,_ = image.shape
	if(rows%2 != 0):
		image = cv2.copyMakeBorder(image, 0,1,0,0, cv2.BORDER_REPLICATE)
	if(cols%2 != 0):
		image = cv2.copyMakeBorder(image, 0,0,0,1, cv2.BORDER_REPLICATE)

	rows_updated,cols_updated,_ = image.shape

	image_padded = cv2.copyMakeBorder(image, pad,pad,pad,pad, cv2.BORDER_REPLICATE)
	image_padded = image_padded.tolist()

	newImg = np.zeros((rows_updated/2, cols_updated/2))
	newImg = newImg.tolist()
	for i in xrange(0,rows_updated/2):
		for j in xrange(0,cols_updated/2):
			sum = getSum(i,j,kernel,k,image_padded,pad)
			newImg[i][j] = [sum[0],sum[1],sum[2]]
	return np.array(newImg)



# image = cv2.imread('../input/dp3.jpg',1)
# current_image = image
# out_arr = [current_image]
# counter = 0
# while(current_image.shape[0]>40):
# 	current_image = reduce(current_image,kernel,k)
# 	out_arr.append(current_image)
# 	cv2.imwrite('../output/reduced/' + str(counter)+'.jpg',current_image)
# 	counter +=1