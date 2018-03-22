
# import numpy as np
# from math import *
# import sys
# import glob

import os
import os.path 
import cv2
import cPickle
import argparse

drag_start = None
setDone = False
pauseOn = False
sel = (0, 0, 0, 0)


def onmouse(event, x, y, flags, param):
    global drag_start, sel, setDone, pauseOn

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0,0,0,0
        cv2.imshow("Video", img)
    elif drag_start and (setDone == False):
        if  (flags & cv2.EVENT_FLAG_LBUTTON) and (event == cv2.EVENT_MOUSEMOVE):
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            dsp = img.copy()
            cv2.rectangle(dsp, (sel[0], sel[1]), (sel[2], sel[3]), (0,255,255), 2)
            cv2.imshow("Video", dsp)
        elif event == cv2.EVENT_LBUTTONUP:
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            dsp = img.copy()
            cv2.rectangle(dsp, (sel[0], sel[1]), (sel[2], sel[3]), (0, 255, 0), 2)
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
                        default='sbwy', type=str)

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


    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

    print height, width


    cv2.namedWindow('Video')
    cv2.setMouseCallback("Video", onmouse)

    drag_start = None
    setDone = False
    pauseOn = False


    while True: 

        if pauseOn == False:
            ret, frame = cap.read()
            if frame is None:
                break
            else:
                img = frame.copy()
                cv2.imshow('Video', img)

        elif setDone:
            ch = raw_input("Save the target?(y/n):")
            if (ch == 'y') or (ch == 'Y'):
                target = raw_input("Input the target Name:")

                imgfname = os.path.join(target_dir, target + '.jpg')
                cv2.imwrite(imgfname, img)

                outfname = os.path.join(target_dir, target + '_bbox.pkl')
                bbox = [sel[0], sel[1], sel[2], sel[3]]
                with open(outfname, 'wb') as f:
                    cPickle.dump(bbox, f, cPickle.HIGHEST_PROTOCOL)

            else:
                setDone = False


        k = cv2.waitKey(1) & 0xff
        if k == 0x20:
            pauseOn = not pauseOn
        elif k== 27:
            break
		

    cv2.destroyAllWindows()
