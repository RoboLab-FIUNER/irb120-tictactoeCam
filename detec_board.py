#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Copyright (C) 2018 FI-UNER Robotic Group
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@mail: robotica@ingenieria.uner.edu.ar
"""
import cv2
import numpy as np
import helper as H
# import socket
# import select
# import time


class Calib:

    def __init__(self):
        self.win_names = ('Raw', 'Canny', 'Dilation', 'Calibration')
        self.win_size = (640, 480)
        self.camera = None
        self.cam_width = None
        self.cam_height = None
        self.T1 = np.float32([[1, 0, -4], [0, 1, -3]])
        self.T2 = np.float32([[1, 0, 5], [0, 1, 4]])
        self.maxT = None
        self.minT = None
        self.k_size = None
        self.d_iter = None
        self.e_iter = None
        self.roi = None

    # Jetson onboard camera
    def open_onboard_camera(self):
        self.camera = cv2.VideoCapture(
            "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
        self.cam_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.camera

    # Open an external usb camera /dev/videoX
    def open_camera_device(self, device_number):
        self.camera = cv2.VideoCapture(device_number)
        self.cam_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.camera

    def create_windows(self):
        win_size = self.win_size
        for name in self.win_names:
            cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
            cv2.resizeWindow(name, win_size[0], win_size[1])

        cv2.createTrackbar('max_canny',  self.win_names[1], 100, 255, lambda *args: None)
        cv2.createTrackbar('min_canny',  self.win_names[1], 50, 255, lambda *args: None)
        cv2.createTrackbar('kernel_size',  self.win_names[2], 1, 5, lambda *args: None)
        cv2.createTrackbar('dilate_iter',  self.win_names[2], 3, 5, lambda *args: None)
        cv2.createTrackbar('erode_iter',  self.win_names[2], 2, 5, lambda *args: None)

    def checkCamera(self):
        if not self.camera.isOpened():
            print("Error opening video stream or file")
            self.camera.release()
            cv2.destroyAllWindows()
            return False
        else:
            return True

    def show(self):
        while True:
            _, frame = self.camera.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def get_roi(self):
        while True:
            self.maxT = cv2.getTrackbarPos('max_canny', self.win_names[1])
            self.minT = cv2.getTrackbarPos('min_canny', self.win_names[1])
            self.k_size = cv2.getTrackbarPos('kernel_size', self.win_names[2])
            self.d_iter = cv2.getTrackbarPos('dilate_iter', self.win_names[2])
            self.e_iter = cv2.getTrackbarPos('erode_iter', self.win_names[2])

            ret, frame = self.camera.read()

            if not ret:
                continue

            raw_frame = frame.copy()
            src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            src_gray = cv2.blur(src_gray, (3, 3))
            canny_output = cv2.Canny(src_gray, self.minT, self.maxT)

            if k_size == 0:
                wrapped = canny_output
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.k_size, self.k_size))
                dilation = cv2.dilate(canny_output, self.kernel, iterations=self.d_iter)
                closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, self.kernel)
                erode = cv2.erode(closed, kernel, iterations=self.e_iter)
                wrapped = cv2.warpAffine(erode, self.T1, (self.cam_width, self.cam_height))

            board = H.findBiggestContour(wrapped)

            if board is False:
                cv2.imshow(self.win_names[1], frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue

            cv2.drawContours(frame, board, -1, (0, 0, 255), 1)
            rect = cv2.boundingRect(board[0])
            x, y, w, h = H.wrap_digit(rect, 5, False)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.roi = np.array((y, y + h, x, x + w))
            cv2.imshow(self.win_names[0], raw_frame)
            cv2.imshow(self.win_names[1], canny_output)
            cv2.imshow(self.win_names[2], wrapped)
            cv2.imshow(self.win_names[3], frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
        return self.roi

    def run(self):
        self.create_windows()
        self.checkCamera()
        self.get_roi()
        # self.show()
#
# def calib(cam):
#     cap = cv2.VideoCapture(cam)
#
#     win_name = ['Canny', 'Dilation', 'Calibration']
#     for name in win_name:
#         cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
#         cv2.resizeWindow(name,640,480)
#
#
#     cv2.createTrackbar('max_canny', win_name[0], 100, 255, lambda *args: None)
#     cv2.createTrackbar('min_canny', win_name[0], 50, 255, lambda *args: None)
#
#
# #     Check if camera opened successfully
#     if (cap.isOpened() == False):
#         print("Error opening video stream or file")
#         cap.release()
#         cv2.destroyAllWindows()
#         return None
#
#     while True:
#         ret, frame0 = cap.read()
#         if ret:
#             rows,cols,ch = frame0.shape
#             T1 = np.float32([[1,0,-4],[0,1,-3]])
#             T2 = np.float32([[1,0,5],[0,1,4]])
#             break
#     while True:
#         maxT=cv2.getTrackbarPos('canny_max_thresh', win_name[0])
#         minT=cv2.getTrackbarPos('canny_min_thresh', win_name[0])
#
#         ret, frame = cap.read()
#         src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         src_gray = cv2.blur(src_gray, (3,3))
#         canny_output = cv2.Canny(src_gray, minT, maxT )
#         kernel = np.ones((2,2),np.uint8)
#         dilation = cv2.dilate(canny_output,kernel,iterations = 3)
#             dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
#         dilation = cv2.erode(dilation,kernel,iterations = 3)
#         dilation = cv2.warpAffine(dilation,T1,(cols,rows))
#         board = H.findBiggestContour(dilation)
#         if board is False:
#             cv2.imshow(source_window,canny_output)
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break
#             continue
#         cv2.drawContours(frame, board, -1, (0,0,255), 1)
#         rect = cv2.boundingRect(board[0])
#         x,y,w,h = H.wrap_digit(rect, 5, False)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
#         roi = (y, y+h, x, x+w)
#         cv2.imshow(win_name[0],canny_output)
#         cv2.imshow(win_name[1],dilation)
#         cv2.imshow(win_name[2],frame)
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return np.array(roi)


def extracCells(roi,cam=0):
    cap = cv2.VideoCapture(cam)

    thresh = 25 # initial threshold
    source_window = 'Extraction'
    cv2.namedWindow(source_window, cv2.WINDOW_GUI_EXPANDED)


#     Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        cap.release()
        cv2.destroyAllWindows()
        return None

    while True:
        ret, frame0 = cap.read()
        if ret:
            rows,cols,ch = frame0.shape
            T1 = np.float32([[1,0,-4],[0,1,-3]])
            T2 = np.float32([[1,0,5],[0,1,4]])
            break

    while True:
        ret, frame = cap.read()
        copy = frame.copy()
#        copy = cv2.warpAffine(copy,T2,(cols,rows))
        copy = copy[roi[0]:roi[1],roi[2]:roi[3]]

        frame = cv2.warpAffine(frame,T1,(cols,rows))
        frame = frame[roi[0]:roi[1],roi[2]:roi[3]]

        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.blur(src_gray, (3,3))
        canny_output = cv2.Canny(src_gray, thresh, thresh * 2)

        kernel = np.ones((2,2),np.uint8)
        dilation = cv2.dilate(canny_output,kernel,iterations = 3)
        dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        dilation = cv2.erode(dilation,kernel,iterations = 3)

        celdas = H.findCells(dilation)
        if celdas is False:
            cv2.imshow(source_window,copy)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue
        celdas, bbox = H.sort_contours(celdas,"top-to-bottom")
        celdas, bbox = H.sort_contours(celdas)

        # Display the resulting frame
        cnt = []
        for i in range(len(celdas)):
            cv2.drawContours(copy, celdas, i, (0,255,0), 1)
            M = cv2.moments(celdas[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #cv2.circle(copy, (cX, cY), 3, (0, 255, 0), -1)
            cnt.append((cX,cY))
            cv2.putText(copy, str(i),(cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(100,200,50),2,cv2.LINE_AA)
#            cv2.waitKey()

        cv2.imshow(source_window,copy)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    celdas = np.array(celdas)
    celdas = np.transpose(celdas.reshape((3,3))).flatten()
    return celdas


def cutCells(roi, celdas, comm, cam=0):
    cap = cv2.VideoCapture(cam)

    thresh = 25 # initial threshold
    source_window = 'Complete'
    cv2.namedWindow(source_window, cv2.WINDOW_GUI_EXPANDED)


#     Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        cv2.destroyAllWindows()
        cap.release()
        return None
    # Read until video is completed
    while True:
        ret, frame0 = cap.read()
        if ret:
            rows,cols,ch = frame0.shape
            break
    T1 = np.float32([[1,0,-4],[0,1,-3]])
    T2 = np.float32([[1,0,2],[0,1,2]])
    cc = []
    for i in celdas:
        cell = cv2.boundingRect(i)
        x,y,w,h = H.wrap_digit(cell, 0, False)
#        cv2.rectangle(copy, (x,y), (x+w, y+h), (0, 0, 255), 1)
        cc.append((y, y+h, x, x+w))


#        cv2.rectangle(copy, (x,y), (x+w, y+h), (0, 255, 0), 2)
#        roi = copy[y:y+h, x:x+w]
#        cv2.drawContours(roi, celdas, -1, (0,0,0), 3)
#
#
#
#
#
#
#    # Convert image to gray and blur it
##    cc = np.array(cc)
##    cc = cc.reshape((3,3,4))
    while True:

        ret, frame = cap.read()
#        copy2 = cv2.warpAffine(copy2,T2,(cols,rows))

#        copy2 = copy2[roi[0]:roi[1],roi[2]:roi[3]]
        frame = frame[roi[0]:roi[1],roi[2]:roi[3]]
        frame = cv2.warpAffine(frame,T2,(cols,rows))

#        copy = frame.copy()

        a1 = detect(frame[cc[0][0]:cc[0][1],cc[0][2]:cc[0][3]])
        a2 = detect(frame[cc[1][0]:cc[1][1],cc[1][2]:cc[1][3]])
        a3 = detect(frame[cc[2][0]:cc[2][1],cc[2][2]:cc[2][3]])

        b1 = detect(frame[cc[3][0]:cc[3][1],cc[3][2]:cc[3][3]])
        b2 = detect(frame[cc[4][0]:cc[4][1],cc[4][2]:cc[4][3]])
        b3 = detect(frame[cc[5][0]:cc[5][1],cc[5][2]:cc[5][3]])

        c1 = detect(frame[cc[6][0]:cc[6][1],cc[6][2]:cc[6][3]])
        c2 = detect(frame[cc[7][0]:cc[7][1],cc[7][2]:cc[7][3]])
        c3 = detect(frame[cc[8][0]:cc[8][1],cc[8][2]:cc[8][3]])
#        for c in row1:
#            c = detect(c)

        row1 = np.array([c1,b2,a3])
        row2 = np.array([b1,c2,b3])
        row3 = np.array([a1,a2,c3])

        detected = np.concatenate((row1[:,1],row2[:,1],row3[:,1]),axis=0)
        detected = np.reshape(detected,(3,3))

        row1 = [cv2.resize(x[0],(90,90)) for x in row1]
        row1 = np.concatenate(row1, axis=1)

        row2 = [cv2.resize(x[0],(90,90)) for x in row2]
        row2 = np.concatenate(row2, axis=1)

        row3 = [cv2.resize(x[0],(90,90)) for x in row3]
        row3 = np.concatenate(row3, axis=1)

        complete = np.concatenate((row1,row2,row3),axis=0)

        cv2.imshow(source_window,complete)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
#        time.sleep(0.5)


        try:
            data_r = conn.recv(1024)
            if data_r == b'a':
                conn.sendall(detected.astype('int8'))
            elif not data_r:
                print('server closing')
                break
        except:
            pass
#           print(detected)

    cap.release()
    cv2.destroyAllWindows()


#
def detect(img):
    ficha = 0
    frame = img
    thresh = 25 # initial thresh.old
#    source_window = 'Complete'0
#    cv2.namedWindow(source_window, cv2.WINDOW_GUI_EXPANDED)
    rows,cols,ch = frame.shape
    T1 = np.float32([[1,0,-4],[0,1,-3]])
    T2 = np.float32([[1,0,5],[0,1,4]])
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3,3))
    canny_output = cv2.Canny(src_gray, thresh, thresh * 2)
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(canny_output,kernel,iterations = 3)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
#    dilation = cv2.erode(dilation,kernel,iterations = 2)
    dilation = cv2.warpAffine(dilation,T1,(cols,rows))
    cnt = H.findContour(dilation)
    cv2.drawContours(frame, cnt, -1, (0,255,0), 1)
    circles = cv2.HoughCircles(src_gray, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 35)
#    print (len(circles.shape))
    if len(circles.shape) == 3:
        a, b, c = circles.shape
        for i in range(b):
            cv2.circle(frame, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
        return frame, 2
    patternX = cv2.cvtColor(cv2.imread('X.png'), cv2.COLOR_BGR2GRAY)
    patternX = cv2.resize(patternX,(90,90))
    patternX = H.findBiggestContour2(patternX)
    threshold = 1.5
    for c in cnt:
        res = cv2.matchShapes(patternX, c, 1, 0.0)
        if res <= threshold:
            rect = cv2.boundingRect(c)
            x,y,w,h = H.wrap_digit(rect, 1, False)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 180, 200), 2)
            return frame,1
    return frame,0


#    res = cv2.matchTemplate(dilation,template,cv2.TM_CCOEFF_NORMED)

#

#    while True:
#        cv2.imshow('dilate', dilation)
#        if cv2.waitKey(5) & 0xFF == ord('q'):
#            break




          # Break the loop
# When everything done, release the video capture object
if __name__ == '__main__':
    calibration = Calib()
    camera = calibration.open_camera_device(1)
    calibration.run()
    print(calibration.cam_height)
    print(calibration.cam_width)
    # calibration.show()
    # cam = cv2.VideoCapture(1)
    # while True:
    #     _, frame = cam.read()
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(5)
    #
    # cv2.destroyAllWindows()
    # cam.release()


#    pitch = calib(cam)
 #   np.save('calib_board.npy',pitch)
 #    pitch = np.load('calib_board.npy')
 #    Cells = extracCells(pitch,cam)
#    Cells = np.save('calib_cells.npy',Cells)
##    Cells = np.load('calib_cells.npy')
#    HOST = ''                 # Symbolic name meaning all available interfaces
#    PORT = 50007              # Arbitrary non-privileged port
##    socket.setdefaulttimeout(0.5)
#    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    s.bind((HOST, PORT))
#    s.listen(1)
#    conn, addr = s.accept()
#    conn.settimeout(0.5)
#    print( 'Connected by', addr)
#    cutCells(pitch, Cells, conn, cam)
#%%
#    cv2.destroyAllWindows()
