""""
Script to calibrate camera.
Press 'c' to capture a still frame
Press 's' to save image for calibration
Press 'esc' to exit image selection phase
"""
# import os.path
import glob
import numpy as np
import cv2 as cv
import v4l2
import arducam_mipicamera as arducam

# 7,9 for own pictures, 6,7 for opencv test pictures
WIDTH = 7
HEIGHT = 9
#square_size = 21.5/1000

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)

objp = np.zeros((WIDTH*HEIGHT, 3), np.float32)
objp[:, 0:2] = np.mgrid[0:HEIGHT, 0:WIDTH].T.reshape(-1, 2)
#objp *= square_size

objpoints = []
imgpoints = []


def align_up(size, align):
    """Returns the closest value greater than 'size' that is dividable by 'align'."""
    return (size + align - 1) & ~(align - 1)


def set_controls(camera):
    """Sets the control variables for the camera."""
    camera.software_auto_exposure(enable=True)
    camera.set_control(v4l2.V4L2_CID_VFLIP, 1)
    camera.set_control(v4l2.V4L2_CID_HFLIP, 1)
    camera.set_control(v4l2.V4L2_CID_GAIN, 255)
    gain = camera.get_control(v4l2.V4L2_CID_GAIN)
    print("Gain = {gain}".format(gain=gain))


def open_frame(img, header, pos_x, pos_y):
    cv.namedWindow(header, cv.WINDOW_NORMAL)
    cv.resizeWindow(header, 534, 360)
    cv.imshow(header, img)
    cv.moveWindow(header, pos_x, pos_y)


if __name__ == "__main__":
    cam = arducam.mipi_camera()
    cam.init_camera()
    fmt = cam.set_resolution(1600, 1080)
    print('Current resolution = {}'.format(fmt))
    set_controls(cam)
    count = 1
    new_frame = False

    while True:
        frame = cam.capture(encoding='raw')
        f_height = int(align_up(fmt[1], 16))
        f_width = int(align_up(fmt[0], 32))
        image = frame.as_array.reshape(f_height, f_width)
        open_frame(image, 'Live feed', 0, 0)
        key = cv.waitKey(1)
        if key == 27:  # esc key exits loop
            print('"esc" key pressed')
            break
        if key == 99:   # 'c' captures and displays still frame
            print('"c" key pressed')
            open_frame(image, 'Captured frame', 534, 0)
            frame_cap = cam.capture(encoding='jpeg')
            new_frame = True
        if key == 115:  # 's' saves still frame
            print('"s" key pressed')
            if new_frame:
                frame_cap.as_array.tofile('clbr{}.jpg'.format(count))
                print('Frame saved as: clbr{}'.format(count))
                count += 1
                new_frame = False
            elif not new_frame:
                print('No new capture available')

    del frame
    cv.destroyWindow('Live feed')
    cam.close_camera()
    images = glob.glob('*.jpg')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (HEIGHT, WIDTH), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv.drawChessboardCorners(img, (HEIGHT, WIDTH), corners2, ret)
            open_frame(img, 'Drawn corners', 0, 0)
            cv.waitKey(500)
        elif not ret:
            print('No corners found in {}'.format(fname))

    cv.destroyWindow('Drawn corners')

    print('Calibrating with {} selected pictures...'.format(len(images)))
    calibrate_ret, matrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print('Calibration results =')
    print('Reprojection error')
    print(calibrate_ret)
    print('Camera Matrix')
    print(matrix)
    print('Distortion Coefficients')
    print(dist)

    # Write data in file
    filename = 'calib_data_RPi.npz'
    print('Writing calibration data in file: {}'.format(filename))
    np.savez(filename, calib_matrix=matrix, dist_coeff=dist)

    #print("Printing loaded data")
    #data = np.load('calib_data.npz')
    # print(data['calib_matrix'])
    # print(data['dist_coeff'])
