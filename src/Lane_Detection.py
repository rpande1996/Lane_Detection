import cv2
import numpy as np

#K = np.array([[1154.22732, 0, 671.627794], [0, 1148.18221, 386.046312], [0, 0, 1]])
#D = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
K = np.array([[903.7596, 0, 695.7519], [0, 901.9653, 224.2509], [0, 0, 1]])
D = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])


#cap = cv2.VideoCapture("../media/input/Video2.mp4")
cap = cv2.VideoCapture("../media/input/Video1.mp4")

w = int(cap.get(3))
h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fps = 29.97
fps = 15
dim = (w, h)

#out = cv2.VideoWriter('../media/output/OutputLaneDetection2.mp4', fourcc, fps, dim)
out = cv2.VideoWriter('../media/output/OutputLaneDetection1.mp4', fourcc, fps, dim)

font = cv2.FONT_HERSHEY_SIMPLEX


def thresh(img):
    global K, D
    img = cv2.undistort(img, K, D, None, K)

    k = np.ones((5, 5), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    #m1 = cv2.inRange(hsv, np.array([10, 110, 18]), np.array([50, 200, 250]))
    m2 = cv2.inRange(hsv, np.array([0, 180, 0]), np.array([255, 255, 255]))
    m1 = cv2.inRange(hsv, np.array([0, 180, 0]), np.array([255, 255, 255]))

    # Remove noise with white color limits
    m1 = cv2.dilate(m1, k, iterations=1)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, k)
    m1 = cv2.erode(m1, k, iterations=1)

    # remove noise with yellow color limits
    m2 = cv2.dilate(m2, k, iterations=1)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k)
    m2 = cv2.erode(m2, k, iterations=1)

    # Combine both the masks
    mask = cv2.bitwise_or(m1, m2)

    # Get the points for the homography
    #ref = np.array([[0, 0], [400, 0], [400, 400], [0, 400]])
    #fimg = np.array([[598, 465], [721, 465], [1263, 700], [22, 700]])
    ref = np.array([[0, 0], [550, 0], [550, 550], [0, 550]])
    fimg = np.array([[530, 275], [780, 275], [1050, 420], [120, 420]])
    return ref, fimg, mask, m1

def lane_detection(fimg, ref, mask, img, m1):
    # Calculate the homography matrix
    H, _ = cv2.findHomography(fimg, ref, cv2.RANSAC, 3.0)

    #R1 = cv2.warpPerspective(mask, H, (400, 400))
    R1 = cv2.warpPerspective(m1, H, (600, 600))

    #R2 = cv2.warpPerspective(img, H, (400, 400))
    R2 = cv2.warpPerspective(img, H, (600, 600))

    # Find Histogram
    crop_lim = R1.shape[0] // 2
    hist = np.sum(R1[crop_lim:, :], axis=0)
    mp = np.int(hist.shape[0] / 2)
    sl = np.argmax(hist[:mp])
    sr = np.argmax(hist[mp:]) + mp

    xleft = []
    yleft = []
    xright = []
    yright = []
    st1 = sl
    st2 = sr

    # Plotting lane lines using calculated histogram
    for i in range(0, 100):
        s1 = st1
        s2 = st2
        xleft.append(st1)
        #yleft.append(400 - (2) * i)
        yleft.append(600 - (2) * i)
        xright.append(st2)
        #yright.append(400 - (2) * i)
        yright.append(600 - (2) * i)

        #hist2 = np.sum(R1[400 - (2) * i - 2:400 - (2) * i, st2 - 30:st2 + 30], axis=0)
        #hist1 = np.sum(R1[400 - (2) * i - 2:400 - (2) * i, st1 - 30:st1 + 30], axis=0)
        hist1 = np.sum(R1[600 - (2) * i - 2:600 - (2) * i, st1 - 30:st1 + 30], axis=0)
        hist2 = np.sum(R1[600 - (2) * i - 2:600 - (2) * i, st2 - 30:st2 + 30], axis=0)

        if len(hist1) != 0:
            actual1 = np.argmax(hist1) + st1 - 30
            st1 = actual1
            if np.argmax(hist1) == 0:
                st1 = s1
        if len(hist2) != 0:
            actual2 = np.argmax(hist2) + st2 - 30
            st2 = actual2
            if np.argmax(hist2) == 0:
                st2 = s2

        leftpoints = np.array([np.transpose(np.vstack([xleft, yleft]))])
        rightpoints = np.array([np.transpose(np.vstack([xright, yright]))])

        cv2.line(R2, (xleft[len(xleft) - 1], yleft[len(yleft) - 1]), (xright[len(xright) - 1], yright[len(yright) - 1]), [255, 0, 0], 10)

        cv2.polylines(R2, np.int32([leftpoints]), 20, (255, 0, 0))
        cv2.polylines(R2, np.int32([rightpoints]), 20, (255, 0, 0))

    # Plotting the edges of the lane
    for i in range(len(xleft)):
        cv2.circle(R2, (xleft[i], yleft[i]), 3, [0, 69, 255], -1)
        cv2.circle(R2, (xright[i], yright[i]), 3, [0, 69, 255], -1)

    return xleft, xright, yleft, yright, img, H, R2

def direction(xleft, xright, yleft, yright, img):
    global font

    # Calculate the mid points of the lanes
    mid = np.array(
        [(xleft[len(xleft) - 1], yleft[len(yleft) - 1]), (xright[len(xright) - 1], yright[len(yright) - 1]), (xleft[len(xleft) - 2], yleft[len(yleft) - 2]),
         (xright[len(xright) - 2], yright[len(yright) - 2])], np.int32)
    midline = np.mean(mid) / 2

    #center_max = 107
    #center_min = 77
    center_max = 170
    center_min = 140

    # Finding the direction of lanes
    if midline > center_max:
        cv2.putText(img, 'Lane Direction: Right', (50, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
    elif midline < center_min:
        cv2.putText(img, 'Lane Direction: Left', (50, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(img, 'Lane Direction: Straight', (50, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
    return img

def warp(homo, img, dup):
    # Creating a warp using the homography matrix, original image and mask
    coor = np.indices((dup.shape[1], dup.shape[0]))
    coor = coor.reshape(2, -1)
    coor = np.vstack((coor, np.ones(coor.shape[1])))
    xt, yt = coor[0], coor[1]
    warp_coor = homo @ coor
    x, y, z = warp_coor[0, :] / warp_coor[2, :], warp_coor[1, :] / warp_coor[2, :], warp_coor[2, :] / warp_coor[2, :]
    xt, yt = xt.astype(int), yt.astype(int)
    x, y = x.astype(int), y.astype(int)

    if x.all() >= 0 and x.all() < 1392 and y.all() >= 0 and y.all() < 512:
        img[y, x] = dup[yt, xt]
    return img

while (cap.isOpened()):
    _, frame = cap.read()
    if frame is not None:

        # Get the masks
        ref_image, img_frame, mask, m1 = thresh(frame)

        # Get the points making the lane.
        xl, xr, yl, yr, frame, H, res1 = lane_detection(img_frame, ref_image, mask, frame, m1)

        # Get the direction of the lane
        frame = direction(xl, xr, yl, yr, frame)

        # Map the lane back to the main image
        frame = warp(np.linalg.inv(H), frame, res1)

        cv2.imshow("Lane Detection", frame)
        out.write(frame)

        k = cv2.waitKey(1) & 0xFF == ord('q')
        if k == 27:
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()