sudo ../../build/tools/compute_image_mean $1 $2

sudo ../../build/tools/caffe train --solver=$3
