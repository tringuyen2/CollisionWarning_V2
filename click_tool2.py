# importing the module
import cv2
import numpy as np

file_depth = 'assets/test.txt'
# reading the image
img = cv2.imread('./test.png', 1)
f = open(file_depth, 'r')
lines = f.readlines()

depth = []
for line in lines:
    depth.append(line.split(' '))
# line1 = lines[0].split(' ')
print(depth[0][0])


h, w, _ = img.shape

h1 = h//2
w1 = w//2

img = cv2.resize(img, (w1, h1))

boxes = []

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        
        print(f"depth: {depth[y*2][x*2]}")

        cv2.imshow('image', img)
 

 
# # driver function
if __name__=="__main__":
 
   

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()