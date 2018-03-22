
# import numpy as np
# from math import *
# import sys
# import glob

import os
import os.path 
import cv2
import cPickle
import argparse
from scipy.io import loadmat, savemat

drag_start = None
setDone = False
pauseOn = False
sel = (0, 0, 0, 0)


def load_targets(target_dir, fname):

    matfname = osp.join(target_dir, fname)
    data = loadmat(matfname)
    query = data['query'].squeeze()

    # print len(query)
    # print type(query)

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

        impath = osp.join(target_dir, im_name)
        targetIm = cv2.imread(impath)
        print targetIm.shape
        cv2.rectangle(targetIm, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow('target', targetIm)

        cv2.waitKey(-1)

    return images, rois,labels


def onmouse(event, x, y, flags, param):
    global drag_start, sel, setDone
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0, 0, 0, 0
        cv2.imshow("Video", img)
    elif drag_start and (setDone == False):
        if (flags & cv2.EVENT_FLAG_LBUTTON) and (event == cv2.EVENT_MOUSEMOVE):
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            dsp = img.copy()
            cv2.rectangle(dsp, (sel[0], sel[1]), (sel[2], sel[3]), (0, 255, 255), 2)
            cv2.imshow("Video", dsp)
        elif event == cv2.EVENT_LBUTTONUP:
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            dsp = img.copy()
            cv2.rectangle(dsp, (sel[0], sel[1]), (sel[2], sel[3]), (0, 255, 255), 2)
            cv2.imshow("Video", dsp)
            setDone = True


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
                        default=None, type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    # print('Called with args:')
    # print(args)

    video_name = args.video_name
    assert(video_name)
    video_set = args.video_set
    assert(video_set)

    data_dir = os.path.join('/home/wangkx/person_search-master/data', video_set)
    vid_dir = os.path.join(data_dir, 'videos')
    print vid_dir
    target_dir = os.path.join(data_dir, 'targets')
    print target_dir
    
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    vidpath = os.path.join(vid_dir, video_name)
    cap = cv2.VideoCapture(vidpath)

    # if (cap.isOpened()):
    #     #start_time = 2*60*1000
    #     start_time = 35 * 60 * 1000
    #     cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_time)

    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

    print height, width


    cv2.namedWindow('Video')
    cv2.setMouseCallback("Video", onmouse)

    drag_start = None
    setDone = False

    ret, frame = cap.read()
    if frame is None:
        print "Video is not opened"
    else:
        img = frame.copy()
        cv2.imshow('Video', img)

    thisTarget = {}

    while True:
        if setDone:
            ch = raw_input("Save the target?(y/n):")
            if (ch == 'y') or (ch == 'Y'):
                target = raw_input("Input the target Name:")
                imgfname = os.path.join(target_dir, target + '.jpg')
                cv2.imwrite(imgfname, img)
                outfname = os.path.join(target_dir, target + '_bbox.pkl')
                bbox = [sel[0], sel[1], sel[2], sel[3]]
                cv2.namedWindow('target')
                targetIm = cv2.imread(imgfname)
                print targetIm.shape
                cv2.rectangle(targetIm, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.imshow('target', targetIm)

                thisTarget['imname'] = imgfname
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                print bbox
                thisTarget['idlocate'] = bbox
                thisTarget['idname'] = target
                thisTarget['ishard'] = []


                outfname = os.path.join(target_dir, 'TrackingTarget.pkl')

                with open(outfname, 'wb') as f:
                    cPickle.dump(thisTarget, f, cPickle.HIGHEST_PROTOCOL)

                cv2.waitKey(-1)
                break

            else:
                cv2.imshow('Video', img)
                drag_start = None
                setDone = False

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
