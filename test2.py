import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

'''

# Region Labeling Project

## 논문 제목
- Fast Region Labeling with Boundary Tracking

## 논문 저자
- Hironobu Takahashi / Fumiaki Tomita

## 작성자
- 한국교통대학교 소프트웨어학과 1444009 박상돈
- Github : [https://github.com/sdp1422/region_processing_project](https://github.com/sdp1422/region_processing_project)

## 최종 구현 일자
- 2019/05/30

## 구현 목적
- 4학년 1학기 영상정보처리 2번째 레포트 과제
- 논문을 읽고 이해한 후, Region Labeling 알고리즘을 구현하고, 간단한 영상을 이치화 시킨 후, 레이블링 된 결과를 캡쳐해서 보고서를 작성한 후 제출하시오.

## 참고 사항
- 적응 이치화 적용
- 논문 해석 오역에 따른 잘못된 코드 구현이 있을 수 있습니다.

'''


def showImage():
    filename = askopenfilename()

    # 한글 경로의 이미지 파일 읽어오기 위한 코드로 수정
    stream = open(filename.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

    # 원본 이미지 띄우기
    cv2.imshow('original image', img)

    # 이미지 높이, 너비
    maxY = img.shape[0]
    maxX = img.shape[1]

    # 이미지 변환 : 컬러 -> 흑백
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    # 원래 코드
    # OTSU의 이진화
    # OTSU 알고리즘과 이진화를 통한 전역 이치화 처리
    # threshold() 함수 사용
    # 극단적으로 이분화 됨

    # 수정 코드 : 적응 이치화로 수정
    # adaptiveThreshold() 함수 사용
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # 이진화된 이미지 출력
    cv2.imshow('binary image', threshold)

    # Opening
    # 모폴로지 연산 적용
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    onRegionLabeling(maxX=maxX, maxY=maxY, source=threshold)


# 영역 레이블링시 경계 탐색(8방향)
def peripheralHoleBoundaryTracking(mode, memImage, cr, cc, pixel, label):
    pdir = 0        # 이전 탐색 방향
    ndir = 0        # 다음 탐색 방향
    r = cr          # row 좌표
    c = cc          # column 좌표
    d = [0 for i in range(8)]
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

        # 마스크 내의 탐색시작 방향 설정
        ndir = pdir - 3
        if ndir == -1:
            ndir = 7
        elif ndir == -2:
            ndir = 6
        elif ndir == -3:
            ndir = 5

        # 마스크 내의 탐색을 시계방향으로 수행
        while True:

            if (d[ndir] == pixel) or (d[ndir] == label):
                flag = False

                # start - switch (pdir) statement for python
                # pdir == 0 상황인 경우의 코드 추가
                if pdir == 0:
                    if ndir != 5 and ndir != 6:
                        flag = True; break
                elif pdir == 1:
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
                    memImage[r][c] = label
                pdir = ndir
                break
            # end - if statement
            # 다음 탐색 방향 설정
            else:
                ndir += 1
                if ndir > 7:
                    ndir = 0
                    # 이미지에 따라 무한 반복되는 상황(while True)이 연출되어 break 구문 추가
                    break
            # end - (it - else) statement
        # end - while loop

        # 위치 이동
        # start - switch (ndir) statement for python
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
        # end - switch (ndir) statement for python

        if (r == cr) and (c == cc):
            break


# 이진 영상에 대해 레이블링 수행
def onRegionLabeling(maxX, maxY, source):
    pixValue = 0
    label = 0

    # 원본 이미지와 같은 크기의 2차원 리스트 생성
    memImage = [[0 for x in range(0, maxX)] for y in range(0, maxY)]

    # 원본 이미지를 memImage에 복사(음수로 만듬, 가장자리는 0으로 변환)
    # 논문에 써있듯이 이미지의 픽셀 값을 0 또는 음수('R')로 초기화
    for y in range(maxY):
        for x in range(maxX):
            c = 0
            if x == 0 or y == 0 or x == (maxX - 1) or y == (maxY - 1):
                c = 0
            else:
                c = source.item((y, x))
                c = -c
            memImage[y][x] = c

    # 영역 레이블링 수행
    for y in range(1, maxY - 1):
        for x in range(1, maxX - 1):
            pixValue = memImage[y][x]

            if memImage[y][x] < 0:
                if (memImage[y][x - 1] <= 0) and (memImage[y - 1][x - 1] <= 0):
                    label += 1
                    memImage[y][x] = label
                    peripheralHoleBoundaryTracking(1, memImage, y, x, pixValue, label)
                elif memImage[y][x - 1] > 0:
                    memImage[y][x] = memImage[y][x - 1]
                elif (memImage[y][x - 1] <= 0) and (memImage[y - 1][x - 1] > 0):
                    memImage[y - 1][x] = memImage[y - 1][x - 1]
                    peripheralHoleBoundaryTracking(2, memImage, y, x, pixValue, memImage[y - 1][x - 1])
        # end - for x range
    # end - for y range

    # 라벨링 개수 출력 및 확인
    print('label = ' + str(label))

    # 레이블링된 각 영역을 적절한 색상으로 표현
    for y in range(0, maxY):
        for x in range(0, maxX):
            # 레이블의 수에 따라 밝기 값을 균등 분할
            c = memImage[y][x] * (255 / (label + 1))
            if c == 0:
                c = 255

            # 각 픽셀에 BGR 값을 대입
            memImage[y][x] = [c, c, c]
        # end - for x range
    # end - for y range

    # memImage를 ndarray 타입으로 변경
    memImage_ndarray = np.asarray(memImage, dtype=np.uint8)

    # matplot 라이브러리를 이용한 결과 화면 출력
    plt.imshow(memImage_ndarray, interpolation='nearest')
    plt.show()

    # 'labeled image' 윈도우 창으로 결과 화면 출력
    cv2.imshow('labeled image', memImage_ndarray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    showImage()


main()
