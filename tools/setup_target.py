

import os.path as osp
import cv2
import cPickle
import numpy as np
import argparse
import datetime
from scipy.io import loadmat, savemat



def set_targets(target_dir, fname):

    query = []

    while True:

        thisTarget = {}
        target = raw_input("Input the target Name or quit:")


        if target == 'quit':

            print 'target'
            print target

            break
        else:

            boxfname = osp.join(target_dir, target + '_bbox.pkl')

            with open(boxfname, 'rb') as f:
                box = cPickle.load(f)
                print box

            cv2.namedWindow('target')
            imgfname = target + '.jpg'

            impath = osp.join(target_dir, target + '.jpg')
            targetIm = cv2.imread(impath)
            print targetIm.shape
            cv2.rectangle(targetIm, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow('target', targetIm)

            thisTarget['imname'] = imgfname
            box[2] -= box[0]
            box[3] -= box[1]
            print box
            thisTarget['idlocate'] = box
            thisTarget['idname'] = target
            thisTarget['ishard'] = []

            query.append(thisTarget)
            print thisTarget
            print query

            cv2.waitKey(-1)
            print len(query)

    cv2.destroyAllWindows()

    data = {}
    data['query'] = query

    print data['query']

    matfname = osp.join(target_dir, fname)
    savemat(matfname, data)



def load_targets(target_dir, fname):

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

        impath = osp.join(target_dir, im_name)
        targetIm = cv2.imread(impath)
        print targetIm.shape
        cv2.rectangle(targetIm, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow('target', targetIm)

        cv2.waitKey(-1)

    return images, rois,labels

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Person Search network')

    parser.add_argument('--video_set',
                        help='video folder for search',
                        default='sbwy', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    # print('Called with args:')
    # print(args)


    video_set = args.video_set
    assert(video_set)

    data_dir = osp.join('/home/wangkx/person_search-master/data', video_set)

    target_dir = osp.join(data_dir, 'targets')

    now = datetime.datetime.now()
    fname = 'SearchTargets'+ now.strftime('%Y-%m-%d-%H:%M')+'.mat'

    set_targets(target_dir, fname)


    images, rois, labels = load_targets(target_dir, fname)