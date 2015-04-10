import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../../../caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import draw_net
import matplotlib.pyplot as plt
import cv2
import os 
from PIL import Image

# arg 0 ==> exec
# arg 1 ==> caffe model
# arg 2 ==> npy file 
# arg 3 ==> gpu number (id) 
# Input image files are taken from the image folder 

if(len(sys.argv) != 4):
  print 'Check the arguments please -- arg 0 = {} arg 1 = {} arg 2 = {} arg 3 = {}'.format(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
  exit(0)
else:
  print 'Arguments are -- arg 0 = {} arg 1 = {} arg 2 = {} arg 3 = {}'.format(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
LOGO_NAMES= ['Addidas', 'Fedex', 'Cocacola', 'DHL' ,'Lufthansa', 'McDonalds' ,'Nike', 'No_Logo', 'Pepsi' ,'Sprite' ,'Starbucks']
MODEL_FILE = 'deploy_11_classes.prototxt'
#PRETRAINED = 'train_val_spp_11_classes_iter17_iter_20001.caffemodel'
PRETRAINED = sys.argv[1]
#CLASS_FILE= '../data/ilsvrc12/synset_words.txt'
IMAGE_FILE = 'input_images.txt'
OUTPUT_FILE = 'out_caffe.txt'
#OUTPUT_HTML_FILE = 'output_caffe_imgnet/out_caffe.html'
#caffe.set_phase_test()
caffe.set_device(int(sys.argv[3]))
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(sys.argv[2]),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(224, 224))

#class_file_ptr = open(CLASS_FILE)
output_file_ptr = open(OUTPUT_FILE,'w')
#output_html_file_ptr = open(OUTPUT_HTML_FILE,'w')
#print 'input file name = ',IMAGE_FILE
input_img_file_ptr = open(IMAGE_FILE)
Path = './images/'

num_of_img = 1
list_of_files = os.listdir(Path)

for j in range(0,len(list_of_files)):
        if not ((list_of_files[j].rfind('.py') == -1) and (list_of_files[j].rfind('.sh') == -1) and (list_of_files[j].rfind('.txt') == -1)):
           continue
        print 'List of files = ',list_of_files[j]
     
	im = Image.open(Path+'/'+list_of_files[j])
	print 'Image width = ',im.size[0]
	print 'Image height = ',im.size[1]
	hori_factor = 1
	vert_factor = 1
	hori_shift = im.size[0] / hori_factor
	vert_shift = im.size[1] / vert_factor   
        
        for y in range(0,vert_factor):
          for x in range (0,hori_factor):
             outfile = 'Cropped_' + str(y) + '_' + str(x)+'.jpg'
             #im = Image.open('fedex.jpg')
             point_x = x * hori_shift
             point_y = y * vert_shift
             #print 'point_x = {} point_y = {}'.format(point_x,point_y) 
             #print 'point_x shi = {} point_y shi = {}'.format(point_x + hori_shift,point_y + vert_shift) 
             region = im.crop((point_x,point_y,point_x + hori_shift,point_y + vert_shift))
             region.save(outfile,"JPEG")
             try:
	        print 'Try prediction'
	        input_image = caffe.io.load_image(outfile)
	     except IOError:
	        print 'IOError '
	        continue
             #plt.imshow(input_image)
             prediction = net.predict([input_image], oversample=False)
	     print '[{}] [{}] Arg = {}'.format(y,x,prediction[0])
	     print '[{}] [{}] Image {} INPUT IMAGE ==> {} and PREDICTED LOGO ==> {} with score {}'.format(y,x,num_of_img,list_of_files[j],LOGO_NAMES[prediction[0].argmax()],prediction[0].argmax())
        num_of_img += 1
#print 'Predicted logo in the input image ---> ',LOGO_NAMES[prediction[0].argmax()]

	input_image_for_disp = cv2.imread(Path+'/'+list_of_files[j],cv2.CV_LOAD_IMAGE_COLOR)
	if(input_image_for_disp == None):
		print 'No input image to display'
	else:
		cv2.namedWindow('Input Image')
	cv2.imshow('Input Image',input_image_for_disp) 
	cv2.waitKey(1000)
	input_img_file_ptr.close()
