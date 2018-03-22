#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test_gallery import test_gallery_image
from fast_rcnn.test_probe import test_net_on_probe_set
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from utils import unpickle
import caffe
import cv2
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
import cPickle

def load_targets(target_dir, fname):

    """font = cv2.FONT_HERSHEY_SIMPLEX

    matfname = osp.join(target_dir, fname)
    data = loadmat(matfname)
    query = data['query'].squeeze()

    print len(query)
    print type(query)

    box = []
    images, rois, labels = [], [], []
    for item in query:
        im_name = str(item['imname'][0,0][0])
        print im_name
        lbl = str(item['idname'][0,0][0])
        print lbl
        box = item['idlocate'][0,0][0]
        print box
        box[2:] += box[:2]
        print box


        images.append(osp.join(target_dir, im_name))
        rois.append(box)
        labels.append(lbl)

        #view_detection(target_dir, im_name, box, lbl)

        impath = osp.join(target_dir, im_name)
        targetIm = cv2.imread(impath)
        print targetIm.shape
        cv2.rectangle(targetIm, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(targetIm, lbl, (box[0], box[1] - 6), font, 0.5, (0, 255, 0), 1)
        cv2.imshow('target', targetIm)

        cv2.waitKey(-1)


    return images, rois, labels"""

    print fname
    fname = osp.join(target_dir,  fname)
    with open(fname, 'rb') as f:
        thisTarget = cPickle.load(f)


    impa, rois , label= [], [] , []
    im_name = thisTarget['imname']
    box = thisTarget['idlocate']
    lbl = thisTarget['idname']
    print label

    box[2]+= box[0]
    box[3]+= box[1]

    rois.append(box)
    label.append(lbl)
    impa.append(osp.join(target_dir, im_name))
    print label

    impath = os.path.join(target_dir, im_name)
    image = cv2.imread(impath)

    print image.shape
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('target', image)

    cv2.waitKey(-1)

    return impa, rois, label


def evaluate_img(result_dir, image, det, feat_g, labels):

    image = image[:, :, (2, 1, 0)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")

    img4face = image.copy()
    faceinBoxes = []
    keep_idx = []
    # remove the boxes without face inside it
    for idx,box in enumerate(det[:4]):
        #detect face only on upper half
        upperbody = int(box[1]+(box[3]-box[1])/2)
        box = [int(x) for x in box]
        # first dim:y, second dim:x
        roi_gray = gray[box[1]:upperbody, box[0]:box[2]]
        faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10), maxSize=(40,40), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

        # there should be one and only one face
        if len(faces)==1:
            facebox = [box[0]+faces[0,0], box[1]+faces[0,1], box[0]+faces[0,0]+faces[0,2], box[1]+faces[0,1]+faces[0,3]]
            faceinBoxes.append(facebox)
            keep_idx.append(idx)

    # if len(faceinBoxes) > 0:
    #    for (x1, y1, x2, y2) in faceinBoxes:
    #        cv2.rectangle(img4face, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # cv2.imshow('Face', img4face)


    # det = det[keep_idx]
    # feat_g = feat_g[keep_idx]

    # for not-normalised feature
    dthreshold = 900
    #dthreshold = 0.45  #[0.40 0.5]
    probe_feat = unpickle(osp.join(result_dir, 'probe_features.pkl'))

    num_probe = len(probe_feat)
    num_boxes = len(feat_g)

    bboxes = []
    names = []

    # normalize the gallery feature
    #feat_gn = feat_g/np.linalg.norm(feat_g,2)

    dis = np.zeros((num_probe,num_boxes))

    for i in xrange(num_probe):

        feat_p = probe_feat[i][np.newaxis, :]
        #feat_pn = feat_p / np.linalg.norm(feat_p, 2)
        #dis[i] = np.sum((feat_pn - feat_gn) ** 2, axis=1)
        dis[i] = np.sum((feat_p - feat_g) ** 2, axis=1)


    for i in xrange(num_probe):
        # row of dis
        rdis = dis[i,:]
        box_inds = np.argsort(rdis)
        rdis = rdis[box_inds]
        # print 'dis after sorting'
        # print rdis
        bbox = det[box_inds[0], :4]
        aspect = (bbox[3] - bbox[1])/(bbox[2] - bbox[0])
        bbox = [int(x) for x in bbox]

        #if (rdis[0] < dthreshold) and (aspect > 2 ):
            #each box only has one prediction according to the shortest distance
        prob_inds = np.argsort(dis[:, box_inds[0]])
        if i== prob_inds[0]:
            bboxes.append(bbox)
            names.append(labels[i])

    return bboxes, names


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
    #                     default=0, type=int)
    # parser.add_argument('--gallery_def',
    #                     help='prototxt file defining the gallery network',
    #                     default=None, type=str)
    # parser.add_argument('--probe_def',
    #                     help='prototxt file defining the probe network',
    #                     default=None, type=str)
    # parser.add_argument('--net', dest='caffemodel',
    #                     help='model to test',
    #                     default=None, type=str)
    # parser.add_argument('--cfg', dest='cfg_file',
    #                     help='optional config file', default=None, type=str)
    # parser.add_argument('--imdb', dest='imdb_name',
    #                     help='dataset to test',
    #                     default='psdb_test', type=str)

    parser.add_argument('--video_name',
                        help='video for search',
                        default=None, type=str)
    parser.add_argument('--video_set',
                        help='video folder for search',
                        default='sbwy', type=str)
    parser.add_argument('--target_file',
                        help='targets for search',
                        default=None, type=str)

    parser.add_argument('--wait', dest='wait',
                         help='wait until net file exists',
                         default=True, type=bool)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--feat_blob',
                        help='name of the feature blob to be extracted',
                        default='feat')



    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':


    flag=0

    args = parse_args()
    # print('Called with args:')
    # print(args)

    video_name = args.video_name
    assert(video_name)
    video_set = args.video_set
    assert(video_set)

    gpu_id = 0
    gallery_def = 'models/psdb/VGG16/test_gallery.prototxt'
    probe_def = 'models/psdb/VGG16/test_probe.prototxt'
    caffemodel = 'output/psdb_train/VGG16_iter_100000.caffemodel'
    cfg_file = 'experiments/cfgs/train.yml'
    imdb_name = 'psdb_test'

    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(set_cfgs)

    cfg.GPU_ID = gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not osp.exists(caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)


  
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)


    # Extract features for probe people

    data_dir = osp.join('/home/wangkx/person_search-master/data', video_set)
    target_dir = osp.join(data_dir, 'targets')

    target_fname = args.target_file
    probe_images, probe_rois, probe_labels = load_targets(target_dir, target_fname)

    # root_dir = imdb._root_dir
    # images_dir = imdb._data_path
    # probe_images, probe_rois, probe_labels = load_probe(root_dir, images_dir)
    net = caffe.Net(probe_def, caffemodel, caffe.TEST)
    net.name = osp.splitext(osp.basename(caffemodel))[0]

    #output_dir = get_output_dir(imdb, net)
    output_dir = osp.join('/home/wangkx/person_search-master/results/', video_set)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    test_net_on_probe_set(net, probe_images, probe_rois, args.feat_blob,
                          output_dir)


   # Detect and store re-id features for all the images in the test images pool
    net = caffe.Net(gallery_def, caffemodel, caffe.TEST)
    net.name = osp.splitext(osp.basename(caffemodel))[0]


    vid_dir = osp.join(data_dir, 'videos')
    #video_name = 'Line2Channel2.avi'
    #video_name = 'Line2Gate1.avi'
    #video_name = 'JWTD08001000_20140721.avi'


    vidpath = osp.join(vid_dir, video_name)
    print vidpath
    cap = cv2.VideoCapture(vidpath)

    frame_rate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # print 'Frame Rate:%d' %frame_rate   # 25 FPS
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

    height=int(height)
    width=int(width)

    nf_2sec = frame_rate*60*2

    vid_pred = []

    frame_count = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frame_idx = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # matplotlib.interactive(True)

    res_fname = video_name[:-4] + '_result.pkl'
    det_file = osp.join(output_dir, res_fname)

    while(cap.isOpened()):
        ret,frame = cap.read()
        if frame is None:
           print("Error! No frame read from the video.")
           break

        #plt.imshow(frame) 
        #plt.show()

        if flag==1:
            im = frame[b1:b2,a1:a2,:]

            exa=a1 
            exb=b1

            #plt.imshow(im) 
            #plt.show()

            det, feat_g = test_gallery_image(net, im, args.feat_blob)
            #print'args.feat_blob'
            #print args.feat_blob
		

        if flag==0:
            det, feat_g = test_gallery_image(net, frame, args.feat_blob)

        if len(det)>0:
            # Evaluate

            #print 'aaaaaaaaaaaaaaaa'

            if flag==0:
                bboxes, persons = evaluate_img(output_dir, frame, det, feat_g, probe_labels)

            if flag==1:
                bboxes, persons = evaluate_img(output_dir, im, det, feat_g, probe_labels)

            #print 'bboxes'
            #print bboxes


            if (len(bboxes)>0):
 
                if flag==0:
                    for i, (x1,y1,x2,y2) in enumerate(bboxes):

                        a1=int(x1*2-x2)
                        if a1<0:
                            a1=0
                        a2=int(x2*2-x1)
                        if a2>width:
                            a2=width
    
                        b1=y1-30
                        if b1<0:
                            b1=0
                        b2=y2+30
                        if b2>height:
                            b2=height


                        #print x1,y1,x2,y2
                        #print a1,b1,a2,b2


                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, persons[i], (x1, y1 - 6), font, 0.5, (0, 0, 255), 1)

                if flag==1:

                    for i, (x1,y1,x2,y2) in enumerate(bboxes):

                        a1=int((x1+exa)*2-(x2+exa))
                        if a1<0:
                            a1=0
                        a2=int((x2+exa)*2-(x1+exa))
                        if a2>width:
                            a2=width
    
                        b1=(y1+exb)-30
                        if b1<0:
                            b1=0
                        b2=(y2+exb)+30
                        if b2>height:
                            b2=height

                        #print x1,y1,x2,y2
                        #print a1,b1,a2,b2

                        cv2.rectangle(frame, (x1+exa, y1+exb), (x2+exa, y2+exb), (0, 0, 255), 2)

                        #print 'x1+exa,y1+exb,x2+exa,y2+exb'
                        #print x1+exa,y1+exb,x2+exa,y2+exb

                        cv2.putText(frame, persons[i], (x1+exa, y1+exb - 6), font, 0.5, (0, 0, 255), 1)


                time_stamps = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                frame_pred={'time_stamps':time_stamps,
                            'bboxes':bboxes,
                            'persons':persons
                        }
                vid_pred.append(frame_pred)

        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # save the search result every 2 mins

        if (frame_idx> nf_2sec) & (frame_idx%nf_2sec==0) & (len(vid_pred)>0):
            print 'Results saved!'
            with open(det_file, 'wb') as f:
                cPickle.dump(vid_pred, f, cPickle.HIGHEST_PROTOCOL)

        #frame_idx += 5
        #print 'Frame Index %s' %frame_idx
        #if (frame_idx <= frame_count):
          # cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_idx)
        #else:
         #  break
        flag=1


    cap.release()
    cv2.destroyAllWindows()

    with open(det_file, 'wb') as f:
        cPickle.dump(vid_pred, f, cPickle.HIGHEST_PROTOCOL)






