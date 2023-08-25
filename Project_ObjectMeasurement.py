import cv2
import numpy as np
import Module_UtilsForObjectMeasurement as utils

########################
webcam = False
path = 'card_img.jpg'
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 640)
cap.set(4, 480)
scale = 2
wP = 270 * scale
hP = 360 * scale
########################

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    img, conts = (utils.getContours
                  (img, showCanny=False, minArea=50000, filter=4))

    if len(conts) != 0:
        biggest = conts[0][2]
        imgWarp = utils.warpImg(img, biggest, wP, hP)
        cv2.imshow("A4", imgWarp)
        imgContours2, conts2 = (utils.getContours
                                (imgWarp, showCanny=False, minArea=2000, filter=4,
                                 cThr=[50, 50], draw=True))
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = utils.reorder(obj[2])
                nW = round((utils.findDis(nPoints[0][0] // (scale*2), nPoints[1][0] // (scale*2)) / 10), 1)
                nH = round((utils.findDis(nPoints[0][0] // (scale*2), nPoints[2][0] // (scale*2)) / 10), 1)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 0), 2, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 0), 2, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 80, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 0, 0), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x + ((x*2)+ x//2), y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 0, 0), 2)
        cv2.imshow("Object", imgContours2)

    img = cv2.resize(img, (600, 800), None, 1, 1)
    cv2.imshow("Original", img)
    cv2.waitKey(1)
