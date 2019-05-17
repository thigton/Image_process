# idea of the file is to test on an image which is in the middle of the experiemtn
# so that you 'should' havse some stable data
import os
import RAW_img
import PLOT
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import itertools

if __name__ == '__main__':
    #pylint: disable=no-member

    # OPTIONS [1 = Create New Dataframe. 0 = Load in existing Dataframe]
    DENSITY_PROFILES = 1
    INTERFACE_HEIGHT = 1
    interface_height_methods = ['grad2']
    # interface_height_methods_to_plot = 'grad'
    # [1 = YES , 0 = NO]



    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    theory_df = PLOT.import_theory() # import the theory steady state dataframe

    
    data_loc = ['190328']# , '190329','190405','190405_2', '190405_3'] #'190328'
    for data in data_loc:
        
        rel_imgs_dir = './Data/' + data + '/' # File path relative to the script
        file_ext = '.JPG' # JPG is working ARW gets error but this might be because I corrupted all the data using git control


        # Get list of file names
        file_ids = RAW_img.get_image_fid(rel_imgs_dir, file_ext)
        filenames = file_ids[file_ext]


        # Get background images
        BG_ids = RAW_img.get_image_fid(rel_imgs_dir + 'BG/', file_ext)
        BG_filenames = BG_ids[file_ext]

        #crop background imgs and get crop_pos  
        (BG, crop_pos) = RAW_img.prep_background_imgs([RAW_img.Raw_img(rel_imgs_dir + 'BG/', f, file_ext) for f in BG_filenames])


        #----------------------------------------------------------------
        for count, f in enumerate(filenames[int((len(filenames)-1)/2):]): # Analyse in reverse so that the crop images are of the steady state
            # Image preprocessing ========================

             # import image
            img = RAW_img.Raw_img(rel_imgs_dir, f, file_ext)
            img.get_experiment_conditions()
            img.convert_centre_pixel_coordinate(crop_pos)

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
            if file_ext == '.ARW':
                img.black_offset(metadata)



            #normalise images
            img.normalise(BG) 

            # convert pixels to a real life scale

            # dye calibration

            # Image analysis ============================


            # split into a door strip and a box strip
            if count == 0:
                # define the door level , box top and bottom returns a dict
                try:
                    with open(rel_imgs_dir + 'box_dims.pickle', 'rb') as pickle_in:
                        box_dims = pickle.load(pickle_in)

                except FileNotFoundError as e:
                    box_dims = RAW_img.box_dims(img) 
                    with open(rel_imgs_dir + 'box_dims.pickle', 'wb') as pickle_out:
                        pickle.dump(box_dims, pickle_out)
                    

                try:
                    with open(rel_imgs_dir + 'analysis_area.pickle', 'rb') as pickle_in:
                        analysis_area = pickle.load(pickle_in)

                except FileNotFoundError as e:
                        img1 = RAW_img.Raw_img(rel_imgs_dir, filenames[-1], file_ext) 
                        img1.crop_img(crop_pos)
                        print('Choose analysis area... \n Ensure the top and bottom are within the depth of the box')
                        analysis_area = img1.choose_crop()
                        with open(rel_imgs_dir + 'analysis_area.pickle', 'wb') as pickle_out:
                            pickle.dump(analysis_area, pickle_out)                 

                # get the scales of analysis area in dimensionless form for both the front and back of the box.
                # Door level, vertical and horizontal scale, camera centre.
                try:
                    with open(rel_imgs_dir + 'analysis/scales.pickle', 'rb') as pickle_in:
                        scales = pickle.load(pickle_in)
                except (FileNotFoundError) as e:
                    scales = RAW_img.make_dimensionless(img,box_dims,analysis_area)
                    with open(rel_imgs_dir + 'analysis/scales.pickle', 'wb') as pickle_out:
                        pickle.dump(scales, pickle_out) 

                # unpack the tuple
                centre = scales[0]
                door_scale = scales[1]
                vertical_scale = scales[2]
                horizontal_scale = scales[3]   
            # Define crop 
            if count ==  len(filenames)-1:
                # Save analysis area for last image
                img.define_analysis_strips(analysis_area, vertical_scale, save = True, door_strip_width = 200)
            else:
                img.define_analysis_strips(analysis_area, vertical_scale,  door_strip_width = 200)

            
            # get 1d density distributioos.chdir
            if DENSITY_PROFILES == 1:
               
                # get one d density profiles
                img.one_d_density(vertical_scale)
                if count == 0: 
                    # make dict on first image
                    density = {}
                    for scale in ['front','back']:
                        density[scale] = pd.DataFrame(getattr(img, scale + '_rho'))

                else:
                    # Add to dataframe
                    for scale in ['front','back']:
                        density[scale] = pd.concat([ density[scale], getattr(img, scale + '_rho') ], axis = 1)
                # once the data has been passed into the df, remove the top level time from attribute for later plotting bugs
                for scale in ['front','back']:
                    setattr(img, scale + '_rho', getattr(img, scale +'_rho')[img.time])
            else: # load in pickle
                try:
                    if count == 0:
                        with open(rel_imgs_dir + 'analysis/density.pickle', 'rb') as pickle_in:
                            density = pickle.load(pickle_in)

                    for scale in ['front', 'back']:
                        # get the right 4 columns but it is missing top level of index
                        setattr(img, scale + '_rho', pd.DataFrame(density[scale][img.time]))

                except FileNotFoundError as e:
                    print('Pickle files don''t exist, need to create by changing DENSITY PROFILES = 1')
                    
            







            # get the interface position
           
            img.interface_height(vertical_scale,centre, methods = interface_height_methods, thres_val = 0.85, rolling_mean = 75, median_filter = 19)
               
               
         # housekeeping
            print( str(count+1) + ' of ' + str((len(filenames)-1)/2) + ' images processed in folder: ' + data + 
            ' - folder ' + str(data_loc.index(data) +1) + ' of ' + str(len(data_loc)) )