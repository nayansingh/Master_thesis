import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../../../caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import draw_net
import matplotlib.pyplot as plt
import cv2
import numpy as np


if(len(sys.argv) != 5):
  print 'Check the arguments please -- arg 0 = {} arg 1 = {} arg 2 = {} arg 3 = {} arg 4 = {}'.format(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
  exit(0)
else:
  print 'Arguments are -- arg 0 = {} arg 1 = {} arg 2 = {} arg 3 = {} arg 4 = {}'.format(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
LOGO_NAMES= ['Addidas', 'Fedex', 'Cocacola', 'DHL' ,'Lufthansa', 'McDonalds' ,'Nike', 'No_Logo', 'Pepsi' ,'Sprite' ,'Starbucks']
MODEL_FILE = 'deploy_11_classes.prototxt'
#PRETRAINED = 'train_val_spp_11_classes_iter17_iter_20001.caffemodel'
PRETRAINED = sys.argv[1]
IMAGE_FILE = 'input_images.txt'
OUTPUT_FILE = 'predicted_output.txt'
#caffe.set_phase_test()
caffe.set_device(int(sys.argv[4]))
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
#input_img_file_ptr = open(IMAGE_FILE)
input_video_cap = cv2.VideoCapture(sys.argv[3])
input_video_cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
input_video_cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

num_of_img = 0
#for j,in_line in enumerate(input_img_file_ptr):
while (input_video_cap.isOpened()):
     #INPUT_IMAGE_FILE = 'images/'
     #INPUT_IMAGE_FILE = INPUT_IMAGE_FILE + in_line[:-1]
     #print 'INPUT FILE TO CAFFE for prediction = ',INPUT_IMAGE_FILE
     ret,INPUT_IMAGE_FILE = input_video_cap.read()
     num_of_img += 1
     #if (num_of_img % 30) != 0:
     #   continue 

     try:
         print 'Frame num = ',num_of_img
         frame_image = open('frame_img.jpg','w')
         cv2.imwrite('frame_image.jpg',INPUT_IMAGE_FILE)
         input_image = caffe.io.load_image('frame_image.jpg')
         frame_image.close()
 
     except IOError:
         print 'IOError '
         break
     ##plt.imshow(input_image)
     prediction = net.predict([input_image], oversample=False)
     #plt.plot(prediction[0])
     out_string = 'Frame num = '+ str(num_of_img) +' PREDICTED LOGO ==> '+ str(LOGO_NAMES[prediction[0].argmax()])+'\n'
     print 'out = ',out_string
     output_file_ptr.write(out_string) 
     
     #input_image_for_disp = cv2.imread(INPUT_IMAGE_FILE,cv2.CV_LOAD_IMAGE_COLOR)
     #if(input_image_for_disp == None):
     #   print 'No input image to display'
     #else:
     cv2.namedWindow('Input Video',cv2.WINDOW_AUTOSIZE)
     #out_image = 'display.jpg'
     #cv2.imwrite(out_image,INPUT_IMAGE_FILE)
     cv2.imshow('Input Video',INPUT_IMAGE_FILE) 
     k = cv2.waitKey(100)
     if k==27:
       break
  
output_file_ptr.close()
#input_img_file_ptr.close()
