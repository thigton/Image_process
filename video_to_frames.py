import cv2
import os

os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location


video_loc = './Data/190516_Flow_meter_cali/'

if not os.path.exists(video_loc + 'Video_3/frames'):
    os.mkdir(video_loc + 'Video_3/frames')
# Opens the Video file
cap= cv2.VideoCapture(video_loc + 'Video_3/00000.MTS')

count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('./Data/190516_Flow_meter_cali/Video_3/frames/'+str(count)+'.png',frame)
        count += 10
        cap.set(1, count)
    else:
        cap.release()
        break

cv2.destroyAllWindows()



