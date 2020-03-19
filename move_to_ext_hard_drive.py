'''Script will copy a set of data to the the external hard drive
 and then delete the photos from the folders on the computer.'''
import shutil
import glob
import os
import time
import pprint


DIRS_TO_MOVE_TO_EXT_DRIVE = []
DATA_LOC = '/home/tdh17/Documents/BOX/PhD/02 Research/Window/Experiment/Data/'
DRIVE = '/media/tdh17/Elements/Window/Data/'

for DIR in DIRS_TO_MOVE_TO_EXT_DRIVE:
    print(f'Updating folder: {DIR}')
    try:
        # copy all files to the external harddrive
        shutil.copytree(f'{DATA_LOC}{DIR}', f'{DRIVE}{DIR}')

        #delete all images from the computer to save space
        # Main exp images
        files = os.listdir(f'{DATA_LOC}{DIR}')
        for fid in files:
            if fid.endswith(('.ARW', '.JPG')):
                os.remove(f'{DATA_LOC}{DIR}/{fid}')
        #Background images
        files = os.listdir(f'{DATA_LOC}{DIR}/BG')
        for fid in files:
            if fid.endswith(('.ARW', '.JPG')):
                os.remove(f'{DATA_LOC}{DIR}/BG/{fid}')
    except FileExistsError:
        # If you've done the original transfer just update the analysis folder
        shutil.rmtree(f'{DRIVE}{DIR}/analysis/')
        shutil.copytree(f'{DATA_LOC}{DIR}/analysis', f'{DRIVE}{DIR}/analysis')


