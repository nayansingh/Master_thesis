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


# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
#def vis_square(data, padsize=1, padval=0):
#
#    fp = open('/home/vion_labs/Caffe/caffe-master/examples/dump.txt','w')
#    #data.dump('/dump.txt')
#
#    for i in range (0,1000):
#       print'Res = ',str(data[i])
#       fp.write(str(data[i]))
#
#    data -= data.min()
#    data /= data.max()
#
#    # force the number of filters to be square
#    n = int(np.ceil(np.sqrt(data.shape[0])))
#    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
#    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
#
#    # tile the filters into an image
#    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#
#    cv2.imshow("feature",data)


# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'deploy_11_classes.prototxt'
PRETRAINED = 'train_val_spp_11_classes_iter17_iter_20001.caffemodel'
#CLASS_FILE= '../data/ilsvrc12/synset_words.txt'
IMAGE_FILE = 'input_images.txt'
OUTPUT_FILE = 'out_caffe.txt'
#OUTPUT_HTML_FILE = 'output_caffe_imgnet/out_caffe.html'
#caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('deploy_11_classes.npy'),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(224, 224))

#class_file_ptr = open(CLASS_FILE)
output_file_ptr = open(OUTPUT_FILE,'w')
#output_html_file_ptr = open(OUTPUT_HTML_FILE,'w')
print 'input file name = ',IMAGE_FILE
input_img_file_ptr = open(IMAGE_FILE)

# accessing weights and biases

#net.deprocess(fc,net.blobs[fc].data[0])
#param = ['loss3/classifier']
#net.param[weights][0].data
#net.param[weights][1].data
#draw_net(net,'drawing.raw')
#params = ['loss3/classifier']
#Layer_name = 'conv1/7x7_s2'
#Layer_name = 'loss3/classifier'
#params = [Layer_name]
#fc_params = {name: (weight, biases)}
#fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
#for conv in params:
#      print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, fc_params[conv][0].shape, fc_params[conv][1].shape)
      #for wei_index in range(0,1000):
          #print ''
          #print 'Weight {} = {}'.format(wei_index,fc_params[conv][0].data)

for j,in_line in enumerate(input_img_file_ptr):
     INPUT_IMAGE_FILE = 'images/'
     INPUT_IMAGE_FILE = INPUT_IMAGE_FILE + in_line[:-1]
     print 'INPUT FILE TO CAFFE for prediction = ',INPUT_IMAGE_FILE
     try:
         print 'Try prediction'
         input_image = caffe.io.load_image(INPUT_IMAGE_FILE.rstrip('\0'))
     except IOError:
         print 'IOError '
         continue
     plt.imshow(input_image)
     prediction = net.predict([input_image], oversample=False)
     #net.deprocess('data',net.blobs['data'].data[0])
     print 'prediction shape:', prediction[0].shape
     plt.plot(prediction[0])
     print 'predicted class:', prediction[0].argmax()
     #net.Forward(input_blobs, output_blobs)
     #output_blob = net.blobs()[-1].data
     #print'ss = {}'.format([(k, v.data.shape) for k, v in net.blobs.items()])
     #feat1 = net.blobs['loss3/classifier'].data[0]
     #feat1 = net.blobs[Layer_name].data[0]
     #vis_square(feat1,padval=1)
     #cv2.waitKey()
     #break
     #np.set_printoptions(threshold='nan')
     #for i,line in enumerate(class_file_ptr):
     #  if i == prediction[0].argmax()-1:
     #     print 'The input image is predicted as : ', line
     #     i = 0
     #     out_str = 'The input image --> ' + in_line[:-1] + ' is predicted to be as --> ' + line
     #     output_file_ptr.write(out_str)
     #     output_html_file_ptr.write(line)
     #     class_file_ptr.seek(0,0)
     #     break

#class_file_ptr.close()
#output_file_ptr.close()
input_img_file_ptr.close()
