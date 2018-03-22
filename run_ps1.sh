

python2 tools/test_net.py --gpu 0 \
  --gallery_def models/psdb/VGG16/test_gallery.prototxt \
  --probe_def models/psdb/VGG16/test_probe.prototxt \
  --net output/psdb_train/VGG16_iter_100000.caffemodel \
  --cfg experiments/cfgs/train.yml \
  --imdb psdb_test \
  --gallery_size=50
