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
     #INPUT_IMAGE_FILE = 'images/'
     #INPUT_IMAGE_FILE = INPUT_IMAGE_FILE + in_line[:-1]
     #print 'INPUT FILE TO CAFFE for prediction = ',INPUT_IMAGE_FILE
     print 'List of files = ',list_of_files[j]
     try:
         print 'Try prediction'
         input_image = caffe.io.load_image(Path+'/'+list_of_files[j])
     except IOError:
         print 'IOError '
         continue
     #plt.imshow(input_image)
     prediction = net.predict([input_image], oversample=False)
     #net.deprocess('data',net.blobs['data'].data[0])
     #print 'prediction shape:', prediction[0].shape
     plt.plot(prediction[0])
     #print 'predicted class:', prediction[0].argmax()
     print 'Arg = ',prediction[0]
     print 'Image {} INPUT IMAGE ==> {} and PREDICTED LOGO ==> {} with score {}'.format(num_of_img,list_of_files[j],LOGO_NAMES[prediction[0].argmax()],prediction[0].argmax())
     num_of_img += 1
     #print 'Predicted logo in the input image ---> ',LOGO_NAMES[prediction[0].argmax()]
     
     input_image_for_disp = cv2.imread(Path+'/'+list_of_files[j],cv2.CV_LOAD_IMAGE_COLOR)
     if(input_image_for_disp == None):
        print 'No input image to display'
     else:
        cv2.namedWindow('Input Image')
        cv2.imshow('Input Image',input_image_for_disp) 
        cv2.waitKey(1000)
 
#class_file_ptr.close()
#output_file_ptr.close()
input_img_file_ptr.close()
