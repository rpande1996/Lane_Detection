## Lane_Detection
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
---

## Overview

The dataset for problem-2 has 2 different video sequences which have different camera parameters for undistorting the images and are
provided separately. The first folder consists of images that you need to stitch to get the video, whereas the second folder has
a video. The respective camera parameters have been included for your reference.
In this project we aim to do simple Lane Detection to mimic Lane Departure Warning systems used in Self Driving Cars. You are
provided with two video sequences, taken from a self-driving car. Your task will be to design an algorithm to detect lanes on the
road, as well as estimate the road curvature to predict car turns.

## Softwares

* Recommended IDE: PyCharm 2021.2

## Libraries

* Numpy 1.21.2
* OpenCV 3.4.8.29

## Programming Languages

* Python 3.8.12

## License 

```
MIT License

Copyright (c) 2021 Rajan Pande

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```
## Bugs

* Improper lane detection at shadowed regions

## Demo

- [Video 1:](https://youtu.be/v1_M-7o2NDo)

![ezgif com-gif-maker](https://github.com/rpande1996/Lane_Detection/blob/main/media/gif/video1.gif)

- [Video 2:](https://youtu.be/S2twHfM-lGk)

![ezgif com-gif-maker](https://github.com/rpande1996/Lane_Detection/blob/main/media/gif/video2.gif)

## Build

```
git clone https://github.com/rpande1996/Lane_Detection
cd Lane_Detection/src
```

For Video 1:

Uncomment Lines: 6, 7, 12, 21, 53, 54, 62, 65, 87, 90, 94, 95, 134 and 135
```
python Video_Creation.py
python Lane_Detection.py
```

For Video 2:

Uncomment Lines: 4, 5, 10, 20, 33, 34, 51, 58, 61, 64, 86, 89, 92, 93, 132 and 133
```
python Lane_Detection.py
```
