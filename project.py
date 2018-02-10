import os
import os.path
import cv2
from skimage import img_as_ubyte
from sklearn.datasets import fetch_mldata
from skimage.morphology import skeletonize
from sklearn.neighbors import KNeighborsClassifier
from skimage.measure import label, regionprops  # implementacija connected-components labelling postupka
# da možemo da dobavimo osobine svakog regiona, za regionprops

import numpy as np
import matplotlib.pyplot as plt

DIR = 'C:\\Users\prole\Desktop\predaja'
file = open(DIR+'\\out.txt', 'w')
file.write('RA 160/2014 Tanja Provci\nfile\tsum\n')


def prepare_train_data(data):
    for i in range(0, len(data)):
        number = data[i].reshape(28, 28)
        closing = cv2.morphologyEx(cv2.inRange(number, 150, 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        labeled = label(closing)
        regions = regionprops(labeled)
        length = len(regions)

        if length <= 1:
            bbox = regions[0].bbox
        elif length > 1:
            max_width = 0
            max_height = 0
            for region in regions:
                t_bbox = region.bbox
                t_width = t_bbox[3] - t_bbox[1]
                t_height = t_bbox[2] - t_bbox[0]
                if(max_width < t_width and max_height < t_height):
                    max_height = t_height
                    max_width = t_width
                    bbox = t_bbox

        img = np.zeros((28, 28))
        x = 0
        x1 = bbox[0]
        x2 = bbox[2]
        y1 = bbox[1]
        y2 = bbox[3]

        for row in range(x1, x2):
            y = 0
            for col in range(y1, y2):
                img[x, y] = number[row, col]
                y += 1
            x += 1
        data[i] = img.reshape(1, 784)


def linear_equation(x, y):
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    return a*x + b*y + c            # linearna jednacina


def get_number_image(bbox, img):
    min_row = bbox[0]
    min_col = bbox[1]
    img_number = np.zeros((28, 28))
    top_range = 28

    for x in range(0, top_range):
        for y in range(0, top_range):
            img_number[x, y] = img[min_row+x-1, min_col+y-1]
    return img_number


def add_number(numbers, currentNum, bbox, frame_num):
    x_axis = bbox[1]
    y_axis = bbox[0]

    for number in reversed(numbers):
        number_in_list = number[0]
        number_x = number[1]
        number_y = number[2]
        number_frame = number[3]
        if number_in_list == currentNum and number_x <= x_axis + 5 and number_y <= y_axis + 5 and number_frame + 2 <= frame_num:
            if abs(frame_num - number_frame) >= 250:         # da ne zameni broj iz liste ukoliko se poklapa sve sem razlike frameova
                break
            numbers.remove(number)
            numbers.append((currentNum, x_axis, y_axis, frame_num))
            return False

    numbers.append((currentNum, x_axis, y_axis, frame_num))


def line_intersection(bbox):
    tol = 4
    tops = linear_equation(bbox[1], bbox[0]) and linear_equation(bbox[3] + tol, bbox[0])
    bottoms = linear_equation(bbox[1], bbox[2] + tol) and linear_equation(bbox[3] + tol, bbox[2] + tol)

    cen = linear_equation((bbox[1] + bbox[3]) / 2, (bbox[0] + (bbox[2]) / 2))
    bottom_end = linear_equation(bbox[3], bbox[2])
    beg = linear_equation(bbox[1], bbox[0])

    # svi koji nisu ni u kom slucaju
    if tops > 0 and bottoms > 0:
        return False

    elif tops < 0 and bottoms < 0:
        return False

    elif bbox[2] + tol < y2 or bbox[3] + tol < x1 or bbox[1] > x2:
        return False

    else:

        if bottom_end >= 0 and frame_num >= 1188:  # Ne racunati poslednje frameove jer donji deo regiona predje liniju, ali ne broj vizuelno
            return False

        elif bottom_end <= 0:                        # Ako donji deo regiona predje liniju, racuna se kao da je broj presao
            return False

        else:
            if bottom_end <= 0 and frame_num == 0:  # Da detektuje onog koji je u 0-om frameu na liniji
                return True
            else:
                return True


def add_and_write(recognized_numbers, video_number):
    sum_of_numbers = 0
    print('Dodati brojevi:')

    for number in recognized_numbers:
        sum_of_numbers += number[0]
        print(number[0])            # samo broj printaj

    print('Suma: '+ str(sum_of_numbers)+'\n')
    file.write('video-' + str(video_number) + '.avi\t ' + str(sum_of_numbers) + '\n')


def find_line(img):
    line_th = cv2.inRange(gray, 10, 55)
    erosion = cv2.erode(line_th, np.ones((2, 2), np.uint8), iterations=1)
    skeleton = skeletonize(erosion / 255.0)
    cv_skeleton = img_as_ubyte(skeleton)
    lines = cv2.HoughLinesP(cv_skeleton, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines


if os.path.exists(os.path.join(DIR, 'mnistPrepared') + '.npy'):
    train = np.load(os.path.join(DIR, 'mnistPrepared') + '.npy')
else:
    train = fetch_mldata('MNIST original').data
    prepare_train_data(train)
    np.save(os.path.join(DIR, 'mnistPrepared'), train)

train_labels = fetch_mldata('MNIST original').target
knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(train, train_labels)

DIR += '\\Videos2'
videos = 10

for video_number in range(0, videos):

    video_names = 'video-' + str(video_number) + '.avi'
    print(video_names)
    video_path = os.path.join(DIR, video_names)
    cap = cv2.VideoCapture(video_path)

    recognized_numbers = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()     # Capture frame-by-frame
        if frame_num % 2 != 0:
            frame_num += 1
            continue
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_num == 0:
            lines = find_line(gray)
            x1, y1, x2, y2 = lines[0][0]

        closing = cv2.morphologyEx(cv2.inRange(gray, 120, 255), cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))         # img, cv.MORPH_CLOSE, kernel
        gray_labeled = label(closing)

        regions = regionprops(gray_labeled)
        # plt.imshow(gray_labeled, 'gray')
        # plt.show()

        for region in regions:
            bbox = region.bbox

            # ako je nize od 9, nije slovo
            if (bbox[2]-bbox[0]) <= 9 or not line_intersection(bbox):
                continue

            img_number = get_number_image(bbox, gray)
            number = int(knn.predict(img_number.reshape(1, 784)))
            plt.imshow(img_number, 'gray')
            # plt.show()
            plt.imshow(gray, 'gray')
           # plt.show()
            if not add_number(recognized_numbers, number, bbox, frame_num):
                continue
        frame_num += 1

    # print('ukupno frameova ' + str(frameNum))

    add_and_write(recognized_numbers, video_number)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
file.close()
