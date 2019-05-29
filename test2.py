import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename


def showImage():
    filename = askopenfilename()

    # 한글 경로의 이미지 파일 읽어오기 위한 코드 수정
    stream = open(filename.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

    cv2.imshow('original image', img)

    maxY = img.shape[0]
    maxX = img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    # OTSU 알고리즘를 이용한 전역 이치화 처리
    # ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('binary image', threshold)

    # Opening
    # 모폴로지 연산 적용
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    # # Labeling
    # # 라벨링
    # ret, markers = cv2.connectedComponents(threshold)
    # cnt = np.amax(markers)
    # print('number of labels = ', cnt)
    #
    # # display markers
    # markers = markers * (254 / cnt)
    # markers = markers.astype(np.uint8)
    # cv2.imshow('labeled image', markers)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    onRegionLabeling(maxX=maxX, maxY=maxY, memImage=threshold)


def peripheralHoleBoundaryTracking(mode, memImage, cr, cc, pixel, label):
    ndir = 0
    pdir = 0
    r = cc
    c = cc
    d = [0 for i in range(8)]
    flag = False

    while True:
        d[0] = memImage.item(r, c + 1)
        d[1] = memImage.item(r + 1, c + 1)
        d[2] = memImage.item(r + 1, c)
        d[3] = memImage.item(r + 1, c - 1)
        d[4] = memImage.item(r, c - 1)
        d[5] = memImage.item(r - 1, c - 1)
        d[6] = memImage.item(r - 1, c)
        d[7] = memImage.item(r - 1, c + 1)


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

                # start - switch(pdir) statement for python
                if pdir == 1:
                    if ndir == 5:
                        flag = True; break
                elif pdir == 2:
                    if ndir == 5 or ndir == 6:
                        flag = True; break
                elif pdir == 3:
                    if ndir == 5 or ndir == 6 or ndir == 7:
                        flag = True; break
                elif pdir == 4:
                    if ndir == 0 or ndir == 5 or ndir == 6 or ndir == 7:
                        flag = True; break
                elif pdir == 5:
                    if ndir != 2 and ndir != 3 and ndir != 4:
                        flag = True; break
                elif pdir == 6:
                    if ndir != 3 and ndir != 4:
                        flag = True; break
                elif pdir == 7:
                    if ndir != 4:
                        flag = True; break
                # end - switch(pdir) statement for python

                if flag:
                    memImage.itemset((r, c), label)
                pdir = ndir
                break
            # end - if statement
            else:
                ndir += 1
                if ndir > 7:
                    ndir = 0
            # end - (it - else) statement
        # end - while loop

        # start - switch(ndir) statement for python
        if ndir == 0:
            c += 1; break
        elif ndir == 1:
            r += 1; c += 1; break
        elif ndir == 2:
            r += 1; break
        elif ndir == 3:
            r += 1; c -= 1; break
        elif ndir == 4:
            c -= 1; break
        elif ndir == 5:
            r -= 1; c -= 1; break
        elif ndir == 6:
            r -= 1; break
        elif ndir == 7:
            r -= 1; c += 1; break
        # end - switch(ndir) statement for python

        if (r == cr) and (c == cc):
            break


def onRegionLabeling(maxX, maxY, memImage):
    pixValue = 0
    label = 0

    for y in range(0, maxY):
        for x in range(0, maxX):
            c = 0
            if x == 0 or y == 0 or x == (maxX - 1) or y == (maxY - 1):
                c = 0
            else:
                c = memImage.item(y, x)
                print(c)
                if c == 0:
                    c = 0
                else:
                    c = -c
                memImage.itemset((y, x), c)
        # end - for x range
    # end - for y range

    for y in range(1, maxY - 1):
        for x in range(1, maxX - 1):
            pixValue = memImage.item(y, x)

            if memImage.item(y, x) < 0:
                if (memImage.item(y, x - 1) <= 0) and (memImage.item(y - 1, x - 1) <= 0):
                    label += 1
                    memImage.itemset((y, x), label)
                    peripheralHoleBoundaryTracking(1, memImage, y, x, pixValue, label)
                    print('1')
                elif memImage.item(y, x - 1) > 0:
                    memImage.itemset((y, x), memImage.item(y, x-1))
                    print('2')
                elif (memImage.item(y, x - 1) <= 0) and (memImage.item(y - 1, x - 1) > 0):
                    memImage.itemset((y, x), memImage.item(y - 1, x - 1))
                    print('흐으으으으음')
                    peripheralHoleBoundaryTracking(2, memImage, y, x, pixValue, memImage.item(y - 1, x - 1))
                    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        # end - for x range
    # end - for y range

    print('label = ' + str(label))

    for y in range(0, maxY):
        for x in range(0, maxX):
            c = memImage.item(y, x) * (255 / (label + 1))
            if c == 0:
                c = 255
            # 이 부분에 색 구분을 픽셀로 지정하는 코드가 들어가야 함
            memImage.itemset((y, x), c)
        # end - for x range
    # end - for y range
    cv2.imshow('labeled image', memImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    showImage()


main()
