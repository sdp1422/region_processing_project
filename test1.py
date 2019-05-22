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
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('binary image', threshold)

    # Opening
    kernel = np.ones((3,3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    # Labeling
    ret, markers = cv2.connectedComponents(threshold)
    # cnt = np.amax(markers)
    # print('number of labels = ', cnt)
    #
    # # display markers
    # markers = markers * (254/cnt)
    # markers = markers.astype(np.uint8)

    ################################
    ###### 라벨링 색깔입히기 #######
    labels = markers
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    ################################

    # cv2.imshow('labeled image', markers)
    cv2.imshow('labeled image', labeled_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

showImage()
