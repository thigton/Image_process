'''Script will take in checkerboard images and provide a camera matrix to calibrate .ARW images.'''
import os
import pickle
import numpy as np
import cv2
import raw_img



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    REL_IMGS_DIR = './Data/190606_cam_cali/' # File path relative to the script
    FILE_EXT = '.ARW'
    GET_CAM_MTX = 1

    if GET_CAM_MTX == 1:
        # termination CRITERIA
        CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(12,5,0)
        OBJP = np.zeros((10*15, 3), np.float32)
        OBJP[:, :2] = np.mgrid[0:15, 0:10].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        OBJPOINTS = [] # 3d point in real world space
        IMGPOINTS = [] # 2d points in image plane.
        # Get list of file names
        FILE_IDS = raw_img.get_image_fid(REL_IMGS_DIR, FILE_EXT)
        FILENAMES = FILE_IDS[FILE_EXT]


        for fname in FILENAMES:
            print(f'Processing {fname}')
            obj = raw_img.raw_img(REL_IMGS_DIR, fname, FILE_EXT)
            IMG = obj.raw_image

            gray = cv2.cvtColor(IMG, cv2.COLOR_RGB2GRAY)

            # gray = img
            # Find the chess board corners
            RET, corners = cv2.findChessboardCorners(gray, (15, 10), None)
            print(RET)
            # If found, add object points, image points (after refining them)
            if RET:
                OBJPOINTS.append(OBJP)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                IMGPOINTS.append(corners2)

                # Draw and display the corners
                IMG = cv2.drawChessboardCorners(IMG, (15, 10), corners2, RET)
                imS = cv2.resize(IMG, (1620, 1080))
                cv2.imshow('img', imS)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        RET, MTX, DIST, RVECS, TVECS = cv2.calibrateCamera(OBJPOINTS, IMGPOINTS, gray.shape[::-1],
                                                           None, None)

        with open(f'{REL_IMGS_DIR[:7]}cam_mtx.pickle', 'wb') as pickle_out:
            pickle.dump((RET, MTX, DIST, RVECS, TVECS), pickle_out)
        exit()
    else:
        pass




    # OBJ = raw_img.raw_img('./Data/190328/', 'DSC01087', FILE_EXT)
    # IMG = OBJ.raw_red
    # H, W = IMG.shape[:2]
    # NEWCAMERAMTX, ROI = cv2.getOptimalNewCameraMatrix(MTX, DIST, (W, H), 0, (W, H))

    # DST = cv2.undistort(IMG, MTX, DIST, None, NEWCAMERAMTX)
    # # crop the image
    # X, Y, W, H = ROI
    # DST = DST[Y:Y+H, X:X+W]
    # cv2.imwrite('./Data/190606_cam_cali/calibresult8.png', DST)
