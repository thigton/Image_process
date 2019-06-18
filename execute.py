'''Script is the first script to run after standard experiment
    Will create:
    Defined crop zones from whole image to box area to analysis area
    Analysis data due to horzontally averaged vertical density profiles
    Interface tracking'''
import os
import pickle
import time
import pandas as pd
import raw_img
import plot
import matplotlib.pyplot as plt
#-------------------------------------------#

if __name__ == '__main__':
    #pylint: disable=no-member

    # OPTIONS [1 = Create New Dataframe. 0 = Load in existing Dataframe]
    DENSITY_PROFILES = 1
    INTERFACE_HEIGHT = 1
    INTERFACE_HEIGHT_METHODS = ['threshold', 'grad', 'grad2']
    INTERFACE_HEIGHT_METHODS_TO_PLOT = 'grad2'
    FILE_EXT = '.ARW'
    # [1 = YES , 0 = NO]
    SAVE = 0
    PLOT_DATA = 1
    TIME = 0

    if TIME == 1:
        TIC = time.time()
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    THEORY_DF = plot.import_theory() # import the theory steady state dataframe
    DATA_LOC = ['190328_3'] #'190328'
    for data in DATA_LOC:
        rel_imgs_dir = './Data/' + data + '/' # File path relative to the script

        # load in camera matrix
        with open(f'{rel_imgs_dir[:7]}cam_mtx.pickle', 'rb') as pickle_in:
            camera_params = pickle.load(pickle_in)

        # Get list of file names
        file_ids = raw_img.get_image_fid(rel_imgs_dir, FILE_EXT)
        filenames = file_ids[FILE_EXT]
        # Get background images
        BG_ids = raw_img.get_image_fid(rel_imgs_dir + 'BG/', FILE_EXT)
        BG_filenames = BG_ids[FILE_EXT]

        # Check histogram of background image
        # img_b = raw_img.raw_img(rel_imgs_dir + 'BG/', BG_filenames[1], FILE_EXT)
        # metadata = img_b.get_metadata()
        # img_b.save_histogram(metadata, crop = False)
        # exit()
        #crop background imgs and get crop_pos
        (BG, crop_pos, box_dims) = raw_img.prep_background_imgs(
            [raw_img.raw_img(rel_imgs_dir + 'BG/',
                             f, FILE_EXT) for f in BG_filenames], camera_params)

        #----------------------------------------------------------------


        if TIME == 1:
            print(str(time.time()-TIC) + 'sec for background images')
            TIC = time.time()
        # Analyse in reverse so that the crop images are of the steady state
        for count, f in enumerate(filenames):
            # Image preprocessing ========================

             # import image
            img = raw_img.raw_img(rel_imgs_dir, f, FILE_EXT)
            img.get_experiment_conditions()
            img.convert_centre_pixel_coordinate(crop_pos)

            if count == 0:
                metadata = img.get_metadata()
                img.get_time()
                t0 = img.time # intial time
            img.get_time(t0)
            if FILE_EXT == '.ARW':
                img.undistort(camera_params)
                img.black_offset(metadata['BlackLevel'], method=0)
                # realign

            img.crop_img(crop_pos)
            img.normalise(BG)

            # Image analysis ============================

            if TIME == 1:
                print(str(time.time()-TIC) + 'sec to normalise image')
                TIC = time.time()

            # split into a door strip and a box strip
            if count == 0:
                
                try:
                    with open(rel_imgs_dir + FILE_EXT[1:] +
                              '_analysis_area.pickle', 'rb') as pickle_in:
                        analysis_area = pickle.load(pickle_in)

                except FileNotFoundError:
                    img1 = raw_img.raw_img(rel_imgs_dir, filenames[-1], FILE_EXT)
                    img1.crop_img(crop_pos)
                    print('''Choose analysis area... \n
                          Ensure the top and bottom are within the depth of the box''')
                    analysis_area = img1.choose_crop()
                    with open(rel_imgs_dir + FILE_EXT[1:] +
                              '_analysis_area.pickle', 'wb') as pickle_out:
                        pickle.dump(analysis_area, pickle_out)
                    del img1
                # get the scales of analysis area in dimensionless form
                # for both the front and back of the box.
                # Door level, vertical and horizontal scale, camera centre.
                try:
                    with open(rel_imgs_dir + 'analysis/' + FILE_EXT[1:] +
                              '_scales.pickle', 'rb') as pickle_in:
                        scales = pickle.load(pickle_in)
                except FileNotFoundError:
                    scales = raw_img.make_dimensionless(img, box_dims, analysis_area)
                    with open(rel_imgs_dir + 'analysis/' + FILE_EXT[1:] +
                              '_scales.pickle', 'wb') as pickle_out:
                        pickle.dump(scales, pickle_out)

                # unpack the tuple
                centre = scales[0]
                door_scale = scales[1]
                vertical_scale = scales[2]
                horizontal_scale = scales[3]
                if TIME == 1:
                    print(str(time.time()-TIC) + 'sec to load in pickles')
                    TIC = time.time()

            # Define crop
            if count == len(filenames)-1:
                # Save analysis area for last image
                img.define_analysis_strips(analysis_area, vertical_scale,
                                           save=True, door_strip_width=100)
            else:
                img.define_analysis_strips(analysis_area, vertical_scale, door_strip_width=100)

            if TIME == 1:
                print(str(time.time()-TIC) + 'sec to define analysis strips')
                TIC = time.time()
            # get 1d density distributions
            if DENSITY_PROFILES == 1:
                # get one d density profiles
                img.one_d_density(vertical_scale)
                if count == 0:
                    # make dict on first image
                    density = {}
                    for scale in ['front', 'back']:
                        density[scale] = pd.DataFrame(getattr(img, scale + '_rho'))

                else:
                    # Add to dataframe
                    for scale in ['front', 'back']:
                        density[scale] = pd.concat([density[scale],
                                                    getattr(img, scale + '_rho')], axis=1)
                # once the data has been passed into the df,
                # remove the top level time from attribute for later plotting bugs
                for scale in ['front', 'back']:
                    setattr(img, scale + '_rho', getattr(img, scale +'_rho')[img.time])
            else: # load in pickle
                try:
                    if count == 0:
                        with open(rel_imgs_dir + 'analysis/' + FILE_EXT[1:]
                                  + '_density.pickle', 'rb') as pickle_in:
                            density = pickle.load(pickle_in)

                    for scale in ['front', 'back']:
                        # get the right 4 columns but it is missing top level of index
                        setattr(img, scale + '_rho', pd.DataFrame(density[scale][img.time]))

                except FileNotFoundError:
                    print('''Pickle files don''t exist,
                          need to create by changing DENSITY PROFILES = 1''')
            if TIME == 1:
                print(str(time.time()-TIC) + 'sec to analyse the density')
                TIC = time.time()

            # get the interface position
            if INTERFACE_HEIGHT == 1:
                img.interface_height(vertical_scale, centre,
                                     methods=INTERFACE_HEIGHT_METHODS, thres_val=0.85,
                                     rolling_mean=75, median_filter=19)
                if count == 0:
                    try:
                        interface_height = {}
                        for scale in ['front', 'back']:
                            interface_height[scale] = pd.DataFrame(getattr(img,
                                                                           scale + '_interface'))
                    except AttributeError:
                        print('''img.interface doesn''t exist,
                              check that eveything works on the .interface_height method''')
                else:
                    for scale in ['front', 'back']:
                        interface_height[scale] = pd.concat([interface_height[scale],
                                                             getattr(img, scale + '_interface')],
                                                            axis=1)
                # once the data has been passed into the df,
                # remove the top level time from attribute for later plotting bugs
                for scale in ['front', 'back']:
                    setattr(img, scale + '_interface', getattr(img, scale +'_interface')[img.time])

            else: # load in pickle

                try:
                    if count == 0:
                        with open(rel_imgs_dir + 'analysis/' + FILE_EXT[1:] +
                                  '_interface_height.pickle', 'rb') as pickle_in:
                            interface_height = pickle.load(pickle_in)

                    for scale in ['front', 'back']:
                        setattr(img, scale + '_interface', interface_height[scale][img.time])

                except FileNotFoundError:
                    print('''Pickle files don''t exist,
                          need to create by changing INTERFACE_HEIGHT = 1''')


            if TIME == 1:
                print(str(time.time()-TIC) + 'sec to analyse the interface height')
                TIC = time.time()

            if PLOT_DATA == 1:
                if (os.path.isfile(rel_imgs_dir + 'analysis/single_density_profiles')) and (count == 0):
                    os.unlink(rel_imgs_dir + 'analysis/single_density_profiles')
                raw_img.plot_density(img, door_scale,
                                     THEORY_DF, interface=INTERFACE_HEIGHT_METHODS_TO_PLOT)

            if TIME == 1:
                print(str(time.time()-TIC) + 'sec to plot the data')
                TIC = time.time()

            # save cropped red image
            if SAVE == 1:
                img.disp_img(disp=False, crop=True, save=True, channel='red')

            # housekeeping
            print(str(count+1) + ' of ' + str(len(filenames)) +
                  ' images processed in folder: ' + data +
                  ' - folder ' + str(DATA_LOC.index(data) +1) + ' of ' + str(len(DATA_LOC)))
        # Write dataframes to pickle
        for df, df_str in zip([density, interface_height], ['density', 'interface_height']):
            fname = rel_imgs_dir + 'analysis/' + FILE_EXT[1:] + '_' + df_str + '.pickle'
            with open(fname, 'wb') as pickle_out:
                pickle.dump(df, pickle_out)
