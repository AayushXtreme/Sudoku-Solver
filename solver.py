## puzzle grid detection using opencv
from numpy.lib.shape_base import dsplit
from extract import extract_puzzle
from sudoku import Sudoku
import argparse
import cv2

parser = argparse.ArgumentParser(description="Sudoku solver using image")
parser.add_argument('--path', '-p', help="Path of image containing puzzle")
parser.add_argument('--debug', '-d', action='store_true', help="helps in debugging")
parser.add_argument('--visualize', '-v', action='store_true', help="visualize puzzle")

args = parser.parse_args()

# grid mask 
def solver(imgpath, debug=False, visualize=False):
    img, solved = extract_puzzle(imgpath, debug)
    orig = img.copy()
    coords = solved[0]
    grid = solved[1]
    print("[INFO] OCR'd Sudoku board:")
    puzzle = Sudoku(3, 3, board=grid.tolist())
    # solve the Sudoku puzzle
    print("[INFO] solving Sudoku puzzle...")
    solution = puzzle.solve()

    # visualize the solved grid
    if visualize:
        # loop over the cell locations and board
        for (cellRow, boardRow) in zip(coords, solution.board):
            # loop over individual cell in the row
            for (box, digit) in zip(cellRow, boardRow):
                # unpack the cell coordinates
                startX, startY, endX, endY = box
                # compute the coordinates of where the digit will be drawn
                # on the output puzzle image
                textX = int((endX - startX) * 0.33)
                textY = int((endY - startY) * -0.2)
                textX += startX
                textY += endY
                # draw the result digit on the Sudoku puzzle image
                cv2.putText(img, str(digit), (textX, textY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
        # show the output image
        cv2.imshow("sudoku puzzle", orig)
        cv2.imshow("sudoku result", img)
        cv2.waitKey(0)
    else:
        # print command line solution
        puzzle.show()
        solution.show_full()

    return solution.board

    
# driver code
if __name__ == '__main__':
    grid = solver(args.path, args.debug, args.visualize)


