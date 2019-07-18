import cv2
import numpy as np
from matplotlib.path import Path

root = 'D:/Code/MultiCamOverlap/dataset/calibration/0421_37/'
image_root = root + 'cam'
matrix_save = root + 'information/ROI.npy'
global refPt
get_Corner = True
get_ROI = True
test_ROI = False

'''
    if you have ROI already,
    you can fill p1 ~ p4 and test_ROI=True to test.
    p1 = Path([(321, 685), (1605, 644), (1918, 731), (1914, 1075), (0, 1080), (0, 794)])
    p2 = Path([(2, 558), (795, 520), (1858, 677), (1691, 1077), (0, 1075)])
    p3 = Path([(0, 455), (792, 388), (1905, 738), (1391, 1077), (0, 1072)])
    p4 = Path([(51, 478), (462, 1074), (811, 1075), (1732, 658), (921, 484)])
'''


def get_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('[[%d, %d]], ' % (x, y), end='')
        refPt.append([x, y])
        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)


def get_coordinate2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, y)
        refPt.append((x, y))
        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)


if get_Corner:
    for icam in range(1, 5):
        image_file = image_root + str(icam) + '.jpg'
        img = cv2.imread(image_file)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', get_coordinate)
        refPt = []
        print('\nicam' + str(icam) + ' = ')
        while(len(refPt) < 16):
            cv2.imshow('image', img)
            cv2.resizeWindow('image', 1728, 972)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # 按q键退出
                break
        cv2.destroyAllWindows()

if get_ROI:
    p = []
    for icam in range(1, 5):
        image_file = image_root + str(icam) + '.jpg'
        img = cv2.imread(image_file)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', get_coordinate2)
        refPt = []
        print('------------------cam: ' + str(icam))
        while(1):
            cv2.imshow('image', img)
            cv2.resizeWindow('image', 1728, 972)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # 按q键退出
                break
        print(refPt)
        new_refPt = Path(refPt)
        p.append(new_refPt)
        cv2.destroyAllWindows()
    #print(p)
    np.save(matrix_save, p)


def inROI(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        p = [p1, p2, p3, p4]
        print(p[icam-1].contains_points([(x, y)])[0])


if test_ROI:
    for icam in range(1, 5):
        image_file = image_root + str(icam) + '.jpg'
        img = cv2.imread(image_file)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image',  1728, 972)
        cv2.setMouseCallback('image', inROI)
        refPt = []
        print('------------------cam: ' + str(icam))
        while(1):
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # 按q键退出
                break
        cv2.destroyAllWindows()
        