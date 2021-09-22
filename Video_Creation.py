import cv2
import os
from os.path import isfile, join

pathIn = './data/'
pathOut = 'Video1.mp4'
fps = 15
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# Sorting the images
files.sort(key=lambda x: x[5:-4])
files.sort()

for i in range(len(files)):
    filename = pathIn + files[i]
    # Reading each file
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    # Inserting the frames into an image array
    frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for i in range(len(frame_array)):

    # Writing to a image array
    out.write(frame_array[i])

out.release()