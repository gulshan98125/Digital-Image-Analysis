import struct
import cv2
import reduce as rc
import expand as ec
import numpy as np
import cPickle as pickle
import os
from bitstring import BitStream, BitArray

divide_factor = 1
laplacian_arr = []
kernel  =     [ 
			[1,4,6,4,1], \
			[4,16,24,16,4], \
			[6,24,36,24,6], \
			[4,16,24,16,4], \
			[1,4,6,4,1]
		  ]
#kernel_multiplier 
k = 1/256.0

f = open('../output/compression/num_images.custom', 'rb')
num_images_bytes = f.read(1)
num_images = struct.unpack("b",num_images_bytes)[0]
f.close()
os.unlink('../output/compression/num_images.custom')

def decompress(j):
	f1 = open('../output/compression/c'+str(j)+'.custom', 'rb')

	rows_bytes = f1.read(2)
	cols_bytes = f1.read(2)

	rows = struct.unpack("H",rows_bytes)[0]
	cols = struct.unpack("H",cols_bytes)[0]

	b_arr_bytes = f1.read(4)
	g_arr_bytes = f1.read(4)
	r_arr_bytes = f1.read(4)

	b_length = struct.unpack("i",b_arr_bytes)[0]
	g_length = struct.unpack("i",g_arr_bytes)[0]
	r_length = struct.unpack("i",r_arr_bytes)[0]

	blue_endbit_length = struct.unpack("b",f1.read(1))[0]
	green_endbit_length = struct.unpack("b",f1.read(1))[0]
	red_endbit_length = struct.unpack("b",f1.read(1))[0]

	blue_binary_string = ''
	green_binary_string = ''
	red_binary_string = ''

	for i in xrange(b_length):
		current_blue_byte = f1.read(1)
		blue_binary_string += BitArray(bytes=current_blue_byte).bin

	for i in xrange(g_length):
		current_green_byte = f1.read(1)
		green_binary_string += BitArray(bytes=current_green_byte).bin

	for i in xrange(r_length):
		current_red_byte = f1.read(1)
		red_binary_string += BitArray(bytes=current_red_byte).bin

	# while(current_blue_byte != ""):
	# 	blue_binary_string += BitArray(bytes=current_blue_byte).bin
	# 	current_blue_byte = f1.read(1)



	f1.close()

	# f2 = open('../output/compression/green.custom', 'rb')

	# green_endbit_length = struct.unpack("b",f2.read(1))[0]

	# current_green_byte = f2.read(1)
	# while(current_green_byte != ""):
	# 	green_binary_string += BitArray(bytes=current_green_byte).bin
	# 	current_green_byte = f2.read(1)
	# f2.close()

	# f3 = open('../output/compression/red.custom', 'rb')
	# red_endbit_length = struct.unpack("b",f3.read(1))[0]

	# current_red_byte = f3.read(1)
	# while(current_red_byte != ""):
	# 	red_binary_string += BitArray(bytes=current_red_byte).bin
	# 	current_red_byte = f3.read(1)

	# f3.close()

	blue_binary_string = blue_binary_string[:len(blue_binary_string)-8] +\
						 blue_binary_string[len(blue_binary_string)-blue_endbit_length:]
	green_binary_string = green_binary_string[:len(green_binary_string)-8] +\
						 green_binary_string[len(green_binary_string)-green_endbit_length:]
	red_binary_string = red_binary_string[:len(red_binary_string)-8] +\
						 red_binary_string[len(red_binary_string)-red_endbit_length:]

	print(len(blue_binary_string),len(green_binary_string),len(red_binary_string),rows,cols)

	# reverse_dict_blue = np.load('../output/compression/reverse_dict_blue.npy').item()
	# reverse_dict_green = np.load('../output/compression/reverse_dict_green.npy').item()
	# reverse_dict_red = np.load('../output/compression/reverse_dict_red.npy').item()
	with open('../output/compression/dict'+str(j)+'.custom', 'rb') as infile:
	    data = pickle.load(infile)
	reverse_dict_blue=data[0]
	reverse_dict_green=data[1]
	reverse_dict_red = data[2]
	blue_arr,green_arr,red_arr = [],[],[]


	temp1 = ''
	for i in xrange(len(blue_binary_string)):
		try:
		  reverse_dict_blue[temp1]
		  blue_arr.append(reverse_dict_blue[temp1])
		  temp1 = blue_binary_string[i]
		except:
			temp1+=blue_binary_string[i]
	blue_arr.append(reverse_dict_blue[temp1])

	temp2 = ''
	for i in xrange(len(green_binary_string)):
		try:
		  reverse_dict_green[temp2]
		  green_arr.append(reverse_dict_green[temp2])
		  temp2 = green_binary_string[i]
		except:
			temp2+=green_binary_string[i]
	green_arr.append(reverse_dict_green[temp2])

	temp3 = ''
	for i in xrange(len(red_binary_string)):
		try:
		  reverse_dict_red[temp3]
		  red_arr.append(reverse_dict_red[temp3])
		  temp3 = red_binary_string[i]
		except:
			temp3+=red_binary_string[i]
	red_arr.append(reverse_dict_red[temp3])

	print(len(blue_arr),len(green_arr), len(red_arr))
	img_unfolded = np.array(zip(blue_arr,green_arr,red_arr))
	img = img_unfolded.reshape((rows,cols,3))

	laplacian_arr.append(np.array(img))

	cv2.imwrite('../output/compression/out_'+str(j)+'.png',img)
	os.unlink('../output/compression/dict'+str(j)+'.custom')
	os.unlink('../output/compression/c'+str(j)+'.custom')

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

for j in xrange(num_images):
	decompress(j)

output_arr = reconstruction(np.array(laplacian_arr))
output_img = output_arr[len(output_arr)-1]
cv2.imwrite('../output/compression/uncompressed.png',output_img)