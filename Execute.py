
import os
import RAW_img
import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------#


if __name__ == '__main__':

    # Chose Parameters
    rel_imgs_dir = './Data/190329/' # File path relative to the script
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
    #exit()

    #crop background imgs and get crop_pos  
    (BG, crop_pos) = RAW_img.prep_background_imgs([RAW_img.Raw_img(rel_imgs_dir + 'BG/', f, file_ext) for f in BG_filenames])
 
    #----------------------------------------------------------------

    # OPTIONS [0 = NO. 1 = YES]
    SAVE = 0
    DENSITY_PROFILES = 1

    count = 0
    for f in filenames: # Analyse in reverse so that the crop images are of the steady state
        # Image preprocessing ========================

         # import image
        img = RAW_img.Raw_img(rel_imgs_dir, f, file_ext)
        img.get_experiment_conditions()

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

        # split into a door strip and a box strip
        if count == 0:
            # if csv file doesn't exist
            if not os.path.isfile(img.img_loc + 'analysis_crop_area.csv'): 
                # import the last image for the crop
                img1 = RAW_img.Raw_img(rel_imgs_dir, filenames[-1], file_ext) 
                 # global crop area
                analysis_area = img1.choose_crop()
                # Transform global cordinates of analysis area to coordinates on cropped image
                analysis_area['x1'] -= crop_pos['x1']
                analysis_area['y1'] -= crop_pos['y1']
                del(img1)
                 # save crop coordinates
                RAW_img.save_dict(rel_imgs_dir, analysis_area, csv_name = 'analysis_crop_area')
            else:
                # else read in crop coordinates
                analysis_area = RAW_img.read_dict(rel_imgs_dir, csv_name = 'analysis_crop_area') 

            # define the door level , box top and bottom returns a dict
            if not os.path.isfile(img.img_loc + 'box_dims.csv'): 
                box_dims = RAW_img.box_dims(img, analysis_area) 

                RAW_img.save_dict(rel_imgs_dir, box_dims, csv_name = 'box_dims')
            else:
                box_dims = RAW_img.read_dict(rel_imgs_dir, csv_name = 'box_dims')

        # Define crop 
        if count ==  len(filenames)-1:
            # Save analysis area for last image
            img.define_analysis_strips(analysis_area, box_dims, door_strip_width = 200, save = True)
        else:
            img.define_analysis_strips(analysis_area, box_dims,  door_strip_width = 200)
        # get 1d density distribution

        if DENSITY_PROFILES == 1:
            # get one d density profiles
            img.one_d_density(box_dims)
 
            if count == 0: 
                # make dataframes on first image
                df_rho = pd.DataFrame(img.rho)
            else:
                # Add to dataframe
                df_rho = pd.concat([df_rho,img.rho], axis = 1)


        # get the interface position
        

    #    if count % 10 == 0:
        #    img.save_histogram(metadata)

        # save cropped red image
        if SAVE == 1:
            img.disp_img(disp = False, crop= True, save = True, channel = 'red')

        # housekeeping
        print( str(count+1) + ' of ' + str(len(filenames)) + ' images processed')
        count += 1
  
    # Write dataframes to csv
    if not os.path.isfile(rel_imgs_dir+ 'analysis/rho.csv'):
        os.remove(rel_imgs_dir+ 'analysis/rho.csv')
    df_rho.to_csv(rel_imgs_dir + 'analysis/rho.csv', sep = ',', index = True)
    

 

