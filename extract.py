import cv2
import numpy as np
from imutils.perspective import four_point_transform
from classifier import ocr

# (9*9) grid
grid = np.zeros((9, 9), dtype=int)

## searching and getting the puzzle
def find_puzzle(img, draw_contours=False):
    '''Finds the biggest object in the image and returns its 4 corners (to crop it)'''

    # Preprocessing:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Get contours:
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extracting the image of what we think might be a sudoku:
    topbottom_edges = (0, img.shape[0]-1)
    leftright_edges = (0, img.shape[1]-1)

    # NOTE change this to 0?
    # NOTE in my webcam contours[0] is always the whole image, so i just ignore it
    if len(contours) > 1:
        conts = sorted(contours, key=cv2.contourArea, reverse=True)

        # Loops through the found objects
        # for something with at least 4 corners and kinda big (>10_000 pixels)
        # NOTE change the 10000 if different webcam
        for cnt in conts:

            epsilon = 0.04*cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(cnt) > 3:
                # Gets the 4 corners of the object (assume it's a square)
                topleft =       min(cnt, key=lambda x: x[0,0]+x[0,1])
                bottomright =   max(cnt, key=lambda x: x[0,0]+x[0,1])
                topright =      max(cnt, key=lambda x: x[0,0]-x[0,1])
                bottomleft =    min(cnt, key=lambda x: x[0,0]-x[0,1])
                corners = (topleft, topright, bottomleft, bottomright)

                # Sometimes it finds 'objects' which are just parts of the screen
                # Ignore those
                badobject = False
                for corner in corners:
                    if corner[0][0] in leftright_edges or corner[0][1] in topbottom_edges:
                        badobject = True

                if badobject is True:
                    continue

            else:
                # If it has less than 4 corners its not a sudoku
                return edges, None


            # NOTE edit this for different webcams, I found at least size 10k is good
            if cv2.contourArea(cnt) > 10000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if draw_contours is True:
                    cv2.drawContours(edges, [box], 0, (0,255,0), 2)

                # Returns the 4 corners of an object with 4+ corners and area of >10k
                # perspective transform
                cnt = np.array([corners[0], corners[1], corners[2], corners[3]]).reshape(4, 2)
                cropped = four_point_transform(img, cnt)
                return edges, cropped

            else:
                return edges, None
    return edges, None


## extracting (OCR) digits from the image
# returns a matrix of digits
def extract_digits(puzzle, debug=False):
    y = round(puzzle.shape[0] / 9)
    x = round(puzzle.shape[1] / 9)
    coords = []
    for i in range(9):
        row = []
        for j in range(9):
            startX = j*x
            startY = i*y
            endX = (j+1)*x
            endY = (i+1)*y
            row.append((startX, startY, endX, endY))
            cell = puzzle[startY:endY, startX:endX]
            digit = ocr(cell, debug)
            grid[i][j] = digit
        coords.append(row)
    
# return puzzle grid
    print("\n################ Sudoku puzzle ###################\n")
    # print(grid)
    return grid, coords


def extract_puzzle(imgPath, debug=False):
    im = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    if im is not None:
        _, cropped = find_puzzle(im)
        # if puzzle found
        if cropped is not None:
            grid, coords = extract_digits(cropped, debug)
            return cropped, [coords, grid]
        else:
            print("Puzzle not found!!!")
            return None
    else:
        print("No image found!!!")
        return None

    

