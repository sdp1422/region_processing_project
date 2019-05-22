import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

def showImage():
    filename = askopenfilename()
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Thresholding
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('binary image', threshold)
    # Opening
    kernel = np.ones((3,3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    # Labeling
    ret, markers = cv2.connectedComponents(threshold)
    cnt = np.amax(markers)
    print('number of labels = ', cnt)
    # display markers
    markers = markers * (254/cnt)
    markers = markers.astype(np.uint8)
    cv2.imshow('labeled image', markers)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

showImage()
