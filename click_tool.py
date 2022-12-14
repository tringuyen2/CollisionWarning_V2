# importing the module
import cv2
import numpy as np

file_depth = 'image-distance/12,4.txt'
file_distance = 'assets/12,4.txt'
# reading the image
img = cv2.imread('image-distance/12,4.png', 1)
f = open(file_depth, 'r')
lines = f.readlines()

depth = []
for line in lines:
    depth.append(line.split(' '))
# line1 = lines[0].split(' ')
print(depth[0][0])

f = open(file_distance, 'r')
lines = f.readlines()

distance = []
for line in lines:
    distance.append(line.split(' '))
# line1 = lines[0].split(' ')
print(distance[0][0])

h, w, _ = img.shape

h1 = h//2
w1 = w//2

img = cv2.resize(img, (w1, h1))
# depth = cv2.resize(img_depth, (w1, h1))
  
# function to display the coordinates of
# of the points clicked on the image

# x1 = 0
# y1 = 0
# x2 = 0
# y2 = 0

boxes = []

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        
        x1 = x
        y1 = y
        sbox = [x, y]
        boxes.append(sbox)
        # displaying the coordinates
        # on the image window
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' +
        #             str(y), (x,y), font,
        #             1, (255, 0, 0), 2)
        print(f"depth: {depth[y*2][x*2]}")
        print(f"distance: {distance[y*2][x*2]}")

        cv2.imshow('image', img)
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        ebox = [x, y]
        boxes.append(ebox)
        print(boxes)
        # x2 = x
        # y2 = y

        crop = distance[boxes[-2][1]*2:boxes[-1][1]*2][boxes[-2][0]*2:boxes[-1][0]*2]

        # print(f"x1={x1}, x2={x2}")
        # displaying the coordinates
        # on the image window
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # b = img[y, x, 0]
        # g = img[y, x, 1]
        # r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' +
        #             str(g) + ',' + str(r),
        #             (x,y), font, 1,
        #             (255, 255, 0), 2)

        # if x1 != 0 and x2 != 0:
        # d = np.median(np.array(crop))
        # print(f"D: {d}")
        cv2.imshow('image', img)
 
# # driver function
# if __name__=="__main__":
 
   

# displaying the image
cv2.imshow('image', img)

# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)

# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()