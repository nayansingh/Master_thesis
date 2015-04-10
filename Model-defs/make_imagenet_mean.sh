#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

/home/vion_labs/Caffe/caffe-master/build/tools/compute_image_mean  spp_net_train_lmdb/
    spp_net_mean.binaryproto

echo "Done."
