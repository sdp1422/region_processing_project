import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename


def showImage():
    filename = askopenfilename()

    stream = open(filename.encode("utf-8"), "rb")

    bytes = bytearray(stream.read())

    numpyarray = np.asarray(bytes, dtype=np.uint8)

    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('binary image', threshold)

    # Opening
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    # Labeling
    ret, markers = cv2.connectedComponents(threshold)
    cnt = np.amax(markers)
    print('number of labels = ', cnt)

    # display markers
    markers = markers * (254 / cnt)
    markers = markers.astype(np.uint8)
    cv2.imshow('labeled image', markers)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


showImage()


def peripheralHoleBoundaryTracking(mode, memImage, cr, cc, pixel, label):
    ndir, pdir = 0
    r = cc
    c = cc
    d = []
    flag = False

    while True:
        d[0] = memImage[r][c + 1]
        d[1] = memImage[r + 1][c + 1]
        d[2] = memImage[r + 1][c]
        d[3] = memImage[r + 1][c - 1]
        d[4] = memImage[r][c - 1]
        d[5] = memImage[r - 1][c - 1]
        d[6] = memImage[r - 1][c]
        d[7] = memImage[r - 1][c + 1]

        if (not d[0]) and (not d[1]) and (not d[2]) and (not d[3]) and (not d[4]) and (not d[5]) and (not d[6]) and (
                not d[7]):
            break
        ndir = pdir - 3
        if ndir == -1:
            ndir = 7
        elif ndir == -2:
            ndir = 6
        elif ndir == -3:
            ndir = 5
        while True:
            if (d[ndir] == pixel) or (d[ndir] == label):
                flag = False
                if pdir == 1:
                    if ndir == 5:
                        flag = True
                        break
                elif pdir == 2:
                    if ndir == 5 or ndir == 6:
                        flag = True
                        break
                elif pdir == 3:
                    if ndir == 5 or ndir == 6 or ndir == 7:
                        flag = True
                        break
                elif pdir == 4:
                    if ndir == 0 or ndir == 5 or ndir == 6 or ndir == 7:
                        flag = True
                        break
                elif pdir == 5:
                    if ndir != 2 and ndir != 3 and ndir != 4:
                        flag = True
                        break
                elif pdir == 6:
                    if ndir != 3 and ndir != 4:
                        flag = True
                        break
                elif pdir == 7:
                    if ndir == 4:
                        flag = True
                        break

                if flag:
                    memImage[r][c] = label
                    pdir = ndir
                break
            else:
                ndir = ndir + 1
                if ndir > 7:
                    ndir = 0

        if ndir == 0:
            c = c + 1
            break
        elif ndir == 1:
            r = r + 1
            c = c + 1
            break
        elif ndir == 2:
            r = r + 1
            break
        elif ndir == 3:
            r = r + 1
            c = c - 1
            break
        elif ndir == 4:
            c = c - 1
            break
        elif ndir == 5:
            r = r - 1
            c = c - 1
            break
        elif ndir == 6:
            r = r - 1
            break
        elif ndir == 7:
            r = r - 1
            c = c + 1
            break
        if (r == cr) and (c == cc):
            break

# def onRegionLabeling():
