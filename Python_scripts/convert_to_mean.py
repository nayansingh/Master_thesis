caffe_root = '../../../../caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
print 'Arg 1 = ',sys.argv[0]
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
print 'Arg 2 = ',sys.argv[2]
np.save( sys.argv[2] , out )
