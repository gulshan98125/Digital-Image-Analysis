import huffman
import collections
import struct
import numpy as np
import cPickle as pickle
from bitstring import BitStream, BitArray

import laplacian as lp

lp_arr = lp.laplacian_arr

def bitstring_to_bytes(s): #currently not using
	if(s=='00000000' or s=='0000000' or s=='000000' or s=='00000' or s=='0000' or s=='000' or s=='00' or s=='0'):
		barr = bytearray(['\x00'])
		return bytes(barr)
	else:
		v = int(s, 2)
		b = bytearray()
		while v:
		    b.append(v & 0xff)
		    v >>= 8
		return bytes(b[::-1])

# reverse_dict_blue = {}
# reverse_dict_green = {}
# reverse_dict_red = {}
#a=byte
#back to bits = BitArray(bytes=a).bin , its a string output
#textwrap.wrap("123456789", 8), to convert to 8 sized string array

# for q in xrange(len(lp_arr)-1):
f = open('../output/compression/num_images.custom', 'wb')
num_images_bytes = struct.pack("b",len(lp_arr))
f.write(num_images_bytes)
f.close()

def compress(q,lp_arr,reverse_dict_blue,reverse_dict_green,reverse_dict_red):
	img = lp_arr[q]
	img_to_list = img.tolist()
	blue_arr,green_arr,red_arr = [],[],[]

	rows,cols,_ = img.shape
	for i in xrange(rows):
		for j in xrange(cols):
			blue_arr.append(img_to_list[i][j][0])
			green_arr.append(img_to_list[i][j][1])
			red_arr.append(img_to_list[i][j][2])

	blue_dict = collections.Counter(blue_arr).items()
	green_dict = collections.Counter(green_arr).items()
	red_dict = collections.Counter(red_arr).items()
	blue_huffman = huffman.codebook(blue_dict)
	green_huffman = huffman.codebook(green_dict)
	red_huffman = huffman.codebook(red_dict)

	blue_bit_string = ''
	green_bit_string = ''
	red_bit_string = ''

	for i in xrange(rows*cols):
			blue_bit_string += blue_huffman[blue_arr[i]]
			reverse_dict_blue[blue_huffman[blue_arr[i]]] =  blue_arr[i]

			green_bit_string += green_huffman[green_arr[i]]
			reverse_dict_green[green_huffman[green_arr[i]]] =  green_arr[i]

			red_bit_string += red_huffman[red_arr[i]]
			reverse_dict_red[red_huffman[red_arr[i]]] =  red_arr[i]

	blue_arr_8_spaced = []
	green_arr_8_spaced = []
	red_arr_8_spaced = []
	for i in xrange(0,len(blue_bit_string),8):
		blue_arr_8_spaced.append(blue_bit_string[i:min(i+8,len(blue_bit_string))])
	for i in xrange(0,len(green_bit_string),8):
		green_arr_8_spaced.append(green_bit_string[i:min(i+8,len(green_bit_string))])
	for i in xrange(0,len(red_bit_string),8):
		red_arr_8_spaced.append(red_bit_string[i:min(i+8,len(red_bit_string))])

	blue_endbit_length = len(blue_arr_8_spaced[len(blue_arr_8_spaced)-1])
	green_endbit_length = len(green_arr_8_spaced[len(green_arr_8_spaced)-1])
	red_endbit_length = len(red_arr_8_spaced[len(red_arr_8_spaced)-1])
	actual_size = ( len(blue_arr) + len(green_arr)+ len(red_arr) )/1024.
	compressed_size = ( len(blue_arr_8_spaced) + len(green_arr_8_spaced) + len(red_arr_8_spaced) )/1024.
	print("actual size , compressed size",actual_size, compressed_size )

	f1 = open('../output/compression/c'+str(q)+'.custom', 'wb')

	rows_bytes = struct.pack("H",rows) #writing rows and cols as
	cols_bytes = struct.pack("H",cols)
	f1.write(rows_bytes)
	# f1.write(rows_bytes[1])
	f1.write(cols_bytes)
	# f1.write(cols_bytes[1])

	#length of 8_space array
	b_arr_bytes = struct.pack("i",len(blue_arr_8_spaced))
	g_arr_bytes = struct.pack("i",len(green_arr_8_spaced))
	r_arr_bytes = struct.pack("i",len(red_arr_8_spaced))
	f1.write(b_arr_bytes)
	f1.write(g_arr_bytes)
	f1.write(r_arr_bytes)

	f1.write(struct.pack("b",blue_endbit_length))
	f1.write(struct.pack("b",green_endbit_length))
	f1.write(struct.pack("b",red_endbit_length))

	for i in xrange(len(blue_arr_8_spaced)):
		f1.write( chr(int(blue_arr_8_spaced[i], 2)))
	
	for i in xrange(len(green_arr_8_spaced)):
		f1.write( chr(int(green_arr_8_spaced[i], 2)))

	for i in xrange(len(red_arr_8_spaced)):
		f1.write( chr(int(red_arr_8_spaced[i], 2)))
	f1.close()

	dict = [reverse_dict_blue,reverse_dict_green,reverse_dict_red]
	with open('../output/compression/dict'+str(q)+'.custom', 'wb') as outfile:
	    pickle.dump(dict, outfile, pickle.HIGHEST_PROTOCOL)
	print('done')

for q in xrange(0,len(lp_arr)):
	compress(q,lp_arr,{},{},{})	






