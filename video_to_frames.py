import cv2
import os
import math as m

def video_to_frames(path, fname,video_fps = 50, image_ext = '.jpg', spacing = 0.5, start_time = 0, **kwargs):
    '''Will save jpg frames of video file in same folder as the video
    path - str location of video file relative to cwd
    fname - str video file name
    spacing - int, spacing between captures - seconds
    start_time - int, time in seconds to start on
    '''

    if not os.path.isfile(path + fname):
        print('video file not found')
        return FileNotFoundError
    # Opens the Video file
    cap= cv2.VideoCapture(path + fname)

    spacing_frame = m.floor(spacing * video_fps)

    start_frame = start_time * video_fps # assuming 50fps

    if 'end_time' in kwargs:
        end_frame = kwargs['end_time'] * 50
    else:
        end_frame = 3600 * 50 # this is an hour
    count = start_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'{path}{count:06d}{image_ext}',frame)
            count += spacing_frame
            if count > end_frame:
                break
            cap.set(1, count)
        else:
            cap.release()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':

    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    video_loc = './Data/190516_Flow_meter_cali/Video_3/'