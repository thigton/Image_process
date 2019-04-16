
import os
import RAW_img
import PLOT
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#-------------------------------------------#


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    theory_df = PLOT.import_theory() # import the theory steady state dataframe

    # Chose Parameters
    data_loc = ['190328']#, '190328_3' ,'190329','190405','190405_2', '190405_3']
    for data in data_loc:
        
        rel_imgs_dir = './Data/' + data + '/' # File path relative to the script
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
        INTERFACE_HEIGHT = 1
        PLOT_DATA = 0

        count = 0
        for f in filenames: # Analyse in reverse so that the crop images are of the steady state
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
            #img.black_offset()

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
                img.define_analysis_strips(analysis_area, vertical_scale, door_strip_width = 200, save = True)
            else:
                img.define_analysis_strips(analysis_area, vertical_scale,  door_strip_width = 200)
            
            # get 1d density distribution

            if DENSITY_PROFILES == 1:
                # get one d density profiles
                img.one_d_density(vertical_scale)
                
                if count == 0: 
                    # make dataframes on first image
                    df_front_rho = pd.DataFrame(img.front_rho)
                    df_back_rho = pd.DataFrame(img.back_rho)
                else:
                    # Add to dataframe
                    df_front_rho = pd.concat([df_front_rho,img.front_rho], axis = 1)
                    df_back_rho = pd.concat([df_back_rho,img.back_rho], axis = 1)

            # get the interface position
            if INTERFACE_HEIGHT == 1:
                # img.interface_height(method = 'threshold', thres_val = 0.85)
                img.interface_height(method = 'grad')
                exit()
                if count == 0:
                    # make dataframe on first image
                    try:
                        df_front_interface = pd.DataFrame(img.front_interface)
                        df_back_interface = pd.DataFrame(img.back_interface)
                    except AttributeError as e:
                        print('img.interface doesn''t exist, check that eveything works on the .interface_height method')
                else:
                    df_front_interface = pd.concat([df_front_interface, img.front_interface], axis = 1)
                    df_back_interface = pd.concat([df_back_interface, img.back_interface], axis = 1)
            
            if PLOT_DATA == 1:
                RAW_img.plot_density(img, door_scale, theory_df)
        #    if count % 10 == 0:
            #    img.save_histogram(metadata)
            # save cropped red image
            if SAVE == 1:
                img.disp_img(disp = False, crop= True, save = True, channel = 'red')

            # housekeeping
            print( str(count+1) + ' of ' + str(len(filenames)) + ' images processed in folder: ' + data + 
            '- folder ' + str(data_loc.index(data) +1) + ' of ' + str(len(data_loc)) )
            count += 1
    
        # Write dataframes to pickle
        for df, df_str in zip([df_front_rho, df_back_rho,df_front_interface,df_back_interface], ['front_rho','back_rho','front_interface','back_interface']):
            fname = rel_imgs_dir + 'analysis/' + df_str + '.pickle'
            with open(fname, 'wb') as pickle_out:
                pickle.dump(df, pickle_out) 


 

