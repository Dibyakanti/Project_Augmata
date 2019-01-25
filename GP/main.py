#Project Augmata - Robotics Club
#25-01-19
#Import libraries
import cv2

#Main function
def main():
	time_img = cv2.imread('clk1.png')
	cap = cv2.VideoCapture(0)
	while True:
		#Get camera data
		_, frame = cap.read()
		row, col, _ = time_img.shape
		frame[0:row, 0:col] = cv2.addWeighted(time_img, 0.8, frame[0:row, 0:col], 1, 1)
		cv2.imshow("wad", time_img)
		cv2.imshow("Input", frame)
		key = cv2.waitKey(5) & 0xFF
		if key == 27: break
	cap.release()
	cv2.destroyAllWindows()

#Run
if __name__ == "__main__":
    main()
