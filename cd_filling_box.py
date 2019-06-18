'''Script will create stills from a vidoe and then convert then find dh/dt '''
import os
from math import sqrt
from math import pi
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import raw_img
from plume_prelim import plume_time_ave
from video_to_frames import video_to_frames


# --------------------------------------#

def plot_draining_box_density(df, door, video_loc):

    # max_grad = get_interface_max_gradient(df) 
    ax = plt.axes()
    tick_spacing = 40
    colors = plt.pcolor(df, cmap='coolwarm')
    plt.colorbar(colors, ax=ax, orientation='vertical', label=r'$-\ln(\frac{I}{I_0})$')
    ylabs = ["%.2f"%item for item in df.index.values[0::tick_spacing] ]
    xlabs = ["%.0f"%item for item in df.columns.values[0::tick_spacing] ]
    plt.yticks(np.arange(0.5, len(df.index), tick_spacing), ylabs)
    plt.xticks(np.arange(0.5, len(df.columns), tick_spacing), xlabs, rotation='vertical')
    door_idx = min(range(df.shape[0]), key=lambda i: abs(df.index[i]- door))
    plt.plot([0, len(df.columns)], [door_idx, door_idx], color='r', label='door')
    # plt.plot(max_grad.index.values / time_between_img, max_grad, color = 'g', label = 'max grad' )
    plt.xlabel('Time (s)')
    plt.ylabel('h/H')
    plt.legend(loc='upper left')
    plt.savefig(f'{video_loc}analysis/density.png', dpi=300)
    plt.close()

def ss_density_check(data, save_loc):
    rho_mean = data['back'].xs(key=('box','mean'), axis=1, level=[1, 2])
    columns = rho_mean.columns.values[::15]

    for col in columns:
        plt.plot(rho_mean[col], rho_mean.index, label=str(col))
    plt.ylabel('h/H')
    plt.xlabel('$A$')
    plt.legend()
    plt.savefig(f'{save_loc}density_profiles.png', dpi=300)
    plt.close()



if __name__ == '__main__':

    CAPTURE_FRAMES = 0
    DENSITY_PROFILES = 0
    INTERFACE_HEIGHT = 0
    CHECK_SS = 0


    S = 0.45 * 0.3 # bot horizontal cross sectional area m^2

    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    video_loc = './Data/190617_5/'
    file_ext = '.jpg'
    fps = 50
    time_between_img = 0.5 # seconds
    plume_absorbance_thres = (0.0, 0.15)
    # Not needed but saves changing the prep_background imgs func
    with open(f'{video_loc[:7]}cam_mtx.pickle', 'rb') as pickle_in:
        camera_params = pickle.load(pickle_in)
    if CAPTURE_FRAMES == 1:
        # create jpgs frames
        video_to_frames(video_loc, '00000.MTS', image_ext=file_ext, 
                        video_fps=fps, spacing=time_between_img, start_time=15)
    
    # Get list of file names
    file_ids = raw_img.get_image_fid(video_loc, file_ext)
    FNAMES = file_ids[file_ext]

    # Get background images
    BG_IDS = raw_img.get_image_fid(video_loc + 'BG/', file_ext)
    BG_FNAMES = BG_IDS[file_ext]

    (BG, CROP_POS, box_dims) = raw_img.prep_background_imgs(
            [raw_img.raw_img(video_loc + 'BG/',
                             f, file_ext) for f in BG_FNAMES], camera_params)
    for count, f in enumerate(FNAMES):
        # Preprocess the image
        #########################################
        img = raw_img.raw_img(video_loc, f, file_ext)
        img.get_experiment_conditions(get_g_ss = True)
        if count == 0:
            plume_q = (img.plume_q*1e-06) / 60 # m^3s^-1
            plume_g = (img.sol_denisty - 0.999) * 9.81 / 0.999 # ms^-2
            g_ss = (img.rho_ss - 0.999) * 9.81 / 0.999 # ms^-2
        img.convert_centre_pixel_coordinate(CROP_POS)
        img.crop_img(CROP_POS)
        img.normalise(BG)
        img.time = float(img.filename) / fps


        # Define items in images (analysis area / scales)
         ##########################################
        try:
            with open(video_loc + file_ext[1:] + '_analysis_area.pickle', 'rb') as pickle_in:
                analysis_area = pickle.load(pickle_in)
        except FileNotFoundError:
            img.crop_img(CROP_POS)
            print('''Choose analysis area...
                  Ensure the top and bottom are within the depth of the box''')
            analysis_area = img.choose_crop()
            with open(video_loc + file_ext[1:] + '_analysis_area.pickle', 'wb') as pickle_out:
                pickle.dump(analysis_area, pickle_out)
        try:
            with open(video_loc + 'analysis/' + file_ext[1:] + '_scales.pickle', 'rb') as pickle_in:
                scales = pickle.load(pickle_in)
        except (FileNotFoundError) as e:
            scales = raw_img.make_dimensionless(img, box_dims, analysis_area)
            with open(video_loc + 'analysis/' + file_ext[1:] +
                      '_scales.pickle', 'wb') as pickle_out:
                pickle.dump(scales, pickle_out)
        # unpack the tuple
        centre = scales[0]
        door_scale = scales[1]
        vertical_scale = scales[2]
        horizontal_scale = scales[3]
         
         # Define crop
        if count ==  len(FNAMES)-1:
            # Save analysis area for last image
            img.define_analysis_strips(analysis_area, vertical_scale,
                                       save=True, door_strip_width=0)
        else:
            img.define_analysis_strips(analysis_area, vertical_scale, door_strip_width=0)
        # Get Density Data
        ##########################################
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
                                                getattr(img, scale + '_rho')],
                                               axis=1)
            # once the data has been passed into the df,
            # remove the top level time from attribute for later plotting bugs
            for scale in ['front', 'back']:
                setattr(img, scale + '_rho', getattr(img, scale +'_rho')[img.time])
        else: # load in pickle
            try:
                with open(video_loc + 'analysis/' + file_ext[1:] +
                          '_density.pickle', 'rb') as pickle_in:
                    density = pickle.load(pickle_in)
            except FileNotFoundError:
                print('Pickle files don''t exist, need to create by changing DENSITY PROFILES = 1')
        # Get Interface_height_data
         ##########################################
        if INTERFACE_HEIGHT == 1:
            img.interface_height(vertical_scale, centre,
                                 methods=['grad', 'grad2'],
                                 rolling_mean=20, median_filter=19)
            if count == 0:
                try:
                    interface_height = {}
                    for scale in ['front', 'back']:
                        interface_height[scale] = pd.DataFrame(getattr(img, scale + '_interface'))
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
                    with open(video_loc + 'analysis/' + file_ext[1:] +
                              '_interface.pickle', 'rb') as pickle_in:
                        interface_height = pickle.load(pickle_in)
                for scale in ['front', 'back']:
                    setattr(img, scale + '_interface', interface_height[scale][img.time])
            except FileNotFoundError:
                print('Pickle files don''t exist, need to create by changing INTERFACE_HEIGHT = 1')
        if (count % 25 == 0):
            raw_img.grad_2_plot(img, vertical_scale, rolling_mean=20)
        # If there is no preprocessing to do.  break to analysis section
        if (DENSITY_PROFILES == 0) and (INTERFACE_HEIGHT == 0):
            break
        print(str(count+1) + ' of ' + str(len(FNAMES)) + ' images processed in folder')


    # Write dataframes to pickle
    if DENSITY_PROFILES == 1:
        FNAME = video_loc + 'analysis/' + file_ext[1:] + '_density.pickle'
        with open(FNAME, 'wb') as pickle_out:
            pickle.dump(density, pickle_out)
    if INTERFACE_HEIGHT == 1:
        FNAME = video_loc + 'analysis/' + file_ext[1:] + '_interface.pickle'
        with open(FNAME, 'wb') as pickle_out:
            pickle.dump(interface_height, pickle_out)



    if CHECK_SS == 1:
        video_loc_ss = video_loc + 'ss_check/'
        # Get list of file names
        file_ids = raw_img.get_image_fid(video_loc_ss, file_ext)
        FNAMES = file_ids[file_ext]
        for count, f in enumerate(FNAMES):
            # Preprocess the image
            #########################################
            img = raw_img.raw_img(video_loc_ss, f, file_ext)
            # img.get_experiment_conditions(get_g_ss = True)
            # if count == 0:
            #     plume_q = (img.plume_q*1e-06) / 60 # m^3s^-1
            #     plume_g = (img.sol_denisty - 0.999) * 9.81 / 0.999 # ms^-2
            #     g_ss = (img.rho_ss - 0.999) * 9.81 / 0.999 # ms^-2
            img.time = float(img.filename) / fps
            img.convert_centre_pixel_coordinate(CROP_POS)
            img.crop_img(CROP_POS)
            img.normalise(BG)

            img.define_analysis_strips(analysis_area, vertical_scale, door_strip_width=0)
            
            # get one d density profiles
            img.one_d_density(vertical_scale)
            if count == 0:
                # make dict on first image
                ss_check = {}
                for scale in ['front', 'back']:
                    ss_check[scale] = pd.DataFrame(getattr(img, scale + '_rho'))
            else:
                # Add to dataframe
                for scale in ['front', 'back']:
                    ss_check[scale] = pd.concat([ss_check[scale],
                                                getattr(img, scale + '_rho')],
                                               axis=1)
            # once the data has been passed into the df,
            # remove the top level time from attribute for later plotting bugs
            for scale in ['front', 'back']:
                setattr(img, scale + '_rho', getattr(img, scale +'_rho')[img.time])

        FNAME = video_loc_ss + file_ext[1:] + '_density_ss_check.pickle'
        with open(FNAME, 'wb') as pickle_out:
            pickle.dump(ss_check, pickle_out)
        ss_density_check(ss_check, video_loc_ss)
   
    else: # load in pickle
        video_loc_ss = video_loc + 'ss_check/'
        try:
            with open(video_loc_ss + file_ext[1:] + '_density_ss_check.pickle', 'rb') as pickle_in:
                ss_check = pickle.load(pickle_in)
            
        except FileNotFoundError:
            print('Pickle files don''t exist, need to create by changing DENSITY PROFILES = 1')
        ss_density_check(ss_check, video_loc_ss)



    # # Analysis Section
    # ##########################################

    DATA = density['front'].xs(key=('box', 'mean'), axis=1, level=[1, 2])
    # # this will flip the DATAframe and therefore hopefully the plot
    # DATA_FLIP = DATA.sort_index(axis=0, ascending=True)
    # # drop rows which are higher than the door
    # DATA_FLIP_DROP = DATA_FLIP[DATA_FLIP.index.values < door_scale.loc['door', 'front']]
    DATA_DROP = DATA[DATA.index.values < door_scale.loc['door', 'front']]

    plot_draining_box_density(DATA_DROP, door_scale.loc['door', 'front'], video_loc)
   
   
    # 1. Create an array of the interface height. Thnk we will have to use a threshold for this.

    # 2. Calulate dh/dt
    
    # 3. depending on experiment conditions either caculate the discharge coefficient directly 
    # or use a already calulated value for the side opening.
   
   
   
   
    