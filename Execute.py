
import os
import RAW_img
import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------#


if __name__ == '__main__':

    # Chose Parameters
    rel_imgs_dir = './Data/190328_3/' # File path relative to the script
    file_ext = '.JPG' # JPG is working ARW gets error but this might be because I corrupted all the data using git control


    # Get list of file names
    file_ids = RAW_img.get_image_fid(rel_imgs_dir, file_ext)
    filenames = file_ids[file_ext]


    # Get background images
    BG_ids = RAW_img.get_image_fid(rel_imgs_dir + 'BG/', file_ext)
    BG_filenames = BG_ids[file_ext]

    ## Check histogram of background image
    #img_b = RAW_img.Raw_img(rel_imgs_dir + 'BG/', BG_filenames[1], file_ext)
    #metadata = img_b.get_metadata()
    #img_b.save_histogram(metadata, crop = False)
    


    #crop background imgs and get crop_pos  
    (BG, crop_pos) = RAW_img.prep_background_imgs([RAW_img.Raw_img(rel_imgs_dir + 'BG/', f, file_ext) for f in BG_filenames])
 
    
    #----------------------------------------------------------------

    # OPTIONS [0 = NO. 1 = YES]
    SAVE = 1
    DENSITY_PROFILES = 0

    count = 0
    for f in filenames: # Analyse in reverse so that the crop images are of the steady state
        # Image preprocessing ========================

         # import image
        img = RAW_img.Raw_img(rel_imgs_dir, f, file_ext)

        if count == 0:
            metadata = img.get_metadata()
            img.get_time()
            t0 = img.time # intial time
        img.get_time(t0)

        
        # realign

        # undistort

         #crop image
        img.crop_img(crop_pos)

        # black offset
        #img.black_offset()

        #normalise images
        img.normalise(BG) 

        # convert pixels to a real life scale

        # dye calibration

        # Image analysis ============================

        # split into vertical strips to analyse
        if count == 0:
            if not os.path.isfile(img.img_loc + 'analysis_crop_area.csv'): # if csv file doesn't exist
                img1 = RAW_img.Raw_img(rel_imgs_dir, filenames[-1], file_ext) # import the last image for the crop
                analysis_area = img1.choose_crop() # global crop area
                del(img1)
                
                RAW_img.save_crop(img, analysis_area, purpose = 'analysis') # save crop coordinates
            else:
                analysis_area = RAW_img.read_crop(img, purpose = 'analysis') # else read in crop coordinates

            
            # Transform global cordinates of analysis area to suit next method
            analysis_area['x1'] -= crop_pos['x1']
            analysis_area['y1'] -= crop_pos['y1']

            door_level = RAW_img.door_level(img, analysis_area) # define the door level
        

        
        if count ==  len(filenames)-1:
            img.define_analysis_strips(analysis_area, 1, display=True) # define the analysis strips and save an image
        else:
            img.define_analysis_strips(analysis_area, 1)
        # get 1d density distribution
        if DENSITY_PROFILES == 1:
            img.one_d_density(door_level, n = 10 )
        # print(density_profiles.head())
        # print(density_profiles.shape)
        
        # get the interface position
        

    #    if count % 10 == 0:
        #    img.save_histogram(metadata)

        # save cropped red image
        if SAVE == 1:
            img.disp_img(disp = False, crop= True, save = True, channel = 'red')

        # housekeeping
        print( str(count+1) + ' of ' + str(len(filenames)) + ' images processed')
        count += 1
  

