import cv2
import numpy as np
import os
from scipy.optimize import fsolve
'''
Use this to calucate intrinsics and extrinsics
and save the reslut in npy file
'''

root = 'D:/Code/MultiCamOverlap/dataset/calibration/0421_37/'
intrinsics_root = root + 'cam'
extrinsics_root = intrinsics_root
matrix_save = root + 'information/'
cam_num = 4
unit = 50


def get_intrinsics(file_dir, icam):
    w = 9
    h = 6
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    file_dir = file_dir + str(icam) + '/'
    count = 0
    for file_name in os.listdir(file_dir):
        img = cv2.imread(file_dir + file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
        # If found, add object points, image points (after refining them)
        if ret is True:
            count += 1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (h, w), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            print('count = ' + str(count))
    #   Calibration     find cameraMatrix and distCoeff
    retval, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    cv2.destroyAllWindows()
    return cmtx, dist


def undistortion(file_dir, icam, cmtx, dist):
    file_dir = file_dir + str(icam) + '/'
    for file_name in os.listdir(file_dir):
        img = cv2.imread(file_dir + file_name)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            cmtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, cmtx, dist, None, newcameramtx)
        cv2.imshow('test', dst)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def get_extrinsics(icam, cameraMatrix, distCoeffs, objp, corners):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    objpoints.append(objp)
    imgpoints.append(corners)
    objpoints2 = np.array(objpoints[0])
    imgpoints2 = np.array(imgpoints[0])
    retval, rvec, tvec = cv2.solvePnP(objpoints2, imgpoints2, cameraMatrix,
                                      distCoeffs)
    r = np.array(rvec)
    t = np.array(tvec)
    rvecs, jacobian = cv2.Rodrigues(r)
    Rt = np.hstack((rvecs, t))
    objpoints_test = objp
    r = Rt[:, 0:3]
    t = Rt[:, 3:]
    imagePoints, jacobian2 = cv2.projectPoints(objpoints_test, r, t,
                                               cameraMatrix, distCoeffs)
    img = cv2.imread(extrinsics_root + str(icam) + '.jpg')
    for i in range(0, len(imagePoints)):
        x = int(imagePoints[i][0, 0])
        y = int(imagePoints[i][0, 1])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    cv2.imwrite(extrinsics_root + str(icam) + '_temp.jpg', img)
    return r, t, Rt


def load_objp_corners():
    objp = np.load(matrix_save + 'objp.npy')
    corners = np.load(matrix_save + 'corners.npy')
    return objp, corners


def main():
    cal_intrinsics = False
    if cal_intrinsics:
        for icam in range(1, cam_num + 1):
            cmtx, dist = get_intrinsics(intrinsics_root, icam)
            undistortion(intrinsics_root, icam, cmtx, dist)
            np.savetxt(matrix_save + 'intrinsics' + str(icam) + '.txt', cmtx)
            np.savetxt(matrix_save + 'distCoeffs' + str(icam) + '.txt', dist)

    cmtx = np.loadtxt(matrix_save + 'intrinsics.txt')
    dist = np.loadtxt(matrix_save + 'distCoeffs.txt')

    objp, corners = load_objp_corners()
    for icam in range(1, cam_num + 1):
        r, t, Rt = get_extrinsics(icam, cmtx, dist, objp[icam - 1], corners[icam - 1])
        np.savetxt(matrix_save + 'Rt' + str(icam) + '.txt', Rt)

    Rt_all = []
    for i in range(1, cam_num + 1):
        Rt = np.loadtxt(matrix_save + 'Rt' + str(i) + '.txt')
        Rt_all.append(Rt)


if __name__ == '__main__':
    main()
