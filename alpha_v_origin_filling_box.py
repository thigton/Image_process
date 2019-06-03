'''Script will create stills from a vidoe and then convert then find dh/dt '''
from video_to_frames import video_to_frames
import os
import RAW_img
import PLOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import FormatStrFormatter

# --------------------------------------#

def plot_filling_box_density(df, door, thres, video_loc):

    ax = plt.axes()
    tick_spacing = 40
    colors = plt.pcolor(df, cmap = 'rainbow')
    plt.colorbar(colors,ax = ax, orientation = 'vertical')
    ylabs = ["%.2f"%item for item in df.index.values[0::tick_spacing] ]
    xlabs = ["%.0f"%item for item in df.columns.values[0::tick_spacing] ]
    plt.yticks(np.arange(0.5, len(df.index), tick_spacing), ylabs)
    plt.xticks(np.arange(0.5, len(df.columns), tick_spacing), xlabs, rotation = 'vertical' )
    CS = plt.contour(df, [0.15, 0.2, 0.25, 0.3, 0.35], color = 'white')
    plt.clabel(CS, CS.levels[::2], inline=True)
    door_idx =  min( range(df.shape[0]) , key = lambda i : abs(df.index[i]- door) )
    plt.plot([0, len(df.columns)], [door_idx,door_idx], color = 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('h/H')
    plt.savefig(f'{video_loc}density.png', dpi = 300)
    plt.close()
    





if __name__ == '__main__':

    S = 0.45 * 0.3 # box horizontal cross sectional area m^2

    DENSITY_PROFILES = 1

    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    videos = ['190521', '190521_2','190521_3','190521_4']
    
    file_ext = '.jpg'
    fps = 50
    for video in videos:
        video_loc = f'./Data/{video}/'
        # create jpgs frames
        video_to_frames(video_loc, '00000.MTS',image_ext = file_ext, video_fps = fps, spacing = 10, start_time = 0)

        # Get list of file names
        file_ids = RAW_img.get_image_fid(video_loc, file_ext)
        fnames = list(map(int, file_ids[file_ext]))
        fnames.sort()
        fnames = list(map(str, fnames))


        # Get background images
        BG_ids = RAW_img.get_image_fid(video_loc + 'BG/', file_ext)
        BG_fnames = list(map(int, BG_ids[file_ext]))
        BG_fnames.sort()
        BG_fnames = list(map(str, BG_fnames))
        (BG, crop_pos) = RAW_img.prep_background_imgs([RAW_img.Raw_img(video_loc + 'BG/', f, file_ext) for f in BG_fnames])

        for count, f in enumerate(fnames):
            img = RAW_img.Raw_img(video_loc, f, file_ext)
            img.get_experiment_conditions()
            if count == 0:
                plume_q = (img.plume_q*1e-06) / 60 # m^3s^-1
                plume_g = (img.sol_denisty - 0.998) * 9.81 / 0.998
            img.convert_centre_pixel_coordinate(crop_pos)
            img.crop_img(crop_pos)
            img.normalise(BG) 
            img.time = float(img.filename) / fps

            try:
                with open(video_loc + file_ext[1:] + '_box_dims.pickle', 'rb') as pickle_in:
                    box_dims = pickle.load(pickle_in)

            except FileNotFoundError as e:
                box_dims = RAW_img.box_dims(img) 
                with open(video_loc + file_ext[1:] + '_box_dims.pickle', 'wb') as pickle_out:
                    pickle.dump(box_dims, pickle_out)

            try:
                with open(video_loc + file_ext[1:] + '_analysis_area.pickle', 'rb') as pickle_in:
                    analysis_area = pickle.load(pickle_in)
    
            except FileNotFoundError as e:
                    img1 = RAW_img.Raw_img(video_loc, fnames[-1], file_ext) 
                    img1.crop_img(crop_pos)
                    print('Choose analysis area... \n Ensure the top and bottom are within the depth of the box')
                    analysis_area = img1.choose_crop()
                    with open(video_loc + file_ext[1:] + '_analysis_area.pickle', 'wb') as pickle_out:
                        pickle.dump(analysis_area, pickle_out)           

            try:
                with open(video_loc + 'analysis/' + file_ext[1:] + '_scales.pickle', 'rb') as pickle_in:
                    scales = pickle.load(pickle_in)
            except (FileNotFoundError) as e:
                scales = RAW_img.make_dimensionless(img,box_dims,analysis_area)
                with open(video_loc + 'analysis/' + file_ext[1:] + '_scales.pickle', 'wb') as pickle_out:
                    pickle.dump(scales, pickle_out)       

            # unpack the tuple
            centre = scales[0]
            door_scale = scales[1]
            vertical_scale = scales[2]
            horizontal_scale = scales[3]   


            # Define crop 
            if count ==  len(fnames)-1:
                # Save analysis area for last image
                img.define_analysis_strips(analysis_area, vertical_scale, save = True, door_strip_width = 0)
            else:
                img.define_analysis_strips(analysis_area, vertical_scale,  door_strip_width = 0)

            # get 1d density distributions
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
                        with open(video_loc + 'analysis/' + file_ext[1:] + '_density.pickle', 'rb') as pickle_in:
                            density = pickle.load(pickle_in)
                    for scale in ['front', 'back']:
                        # get the right 4 columns but it is missing top level of index
                        setattr(img, scale + '_rho', pd.DataFrame(density[scale][img.time]))
                    break
                except FileNotFoundError as e:
                    print('Pickle files don''t exist, need to create by changing DENSITY PROFILES = 1')

            # if INTERFACE_HEIGHT == 1:
            #     img.interface_height(vertical_scale,centre, methods = ['threshold'], thres_val = 0.85)
            #     if count == 0:
            #         try:
            #             interface_height = {}
            #             for scale in ['front','back']:
            #                 interface_height[scale] = pd.DataFrame(getattr(img, scale + '_interface'))
            #         except AttributeError as e:
            #             print('img.interface doesn''t exist, check that eveything works on the .interface_height method')
            #     else:
            #         for scale in ['front','back']:
            #             interface_height[scale] = pd.concat([ interface_height[scale], getattr(img, scale + '_interface') ], axis = 1)
            #     # once the data has been passed into the df, remove the top level time from attribute for later plotting bugs
            #     for scale in ['front','back']:
            #         setattr(img, scale + '_interface', getattr(img, scale +'_interface')[img.time])

            # else: # load    print(thres_line)

            #         try:    exit()
            #             if count == 0:
            #                 with open(video_loc + 'analysis/' + file_ext[1:] + '_interface_height.pickle', 'rb') as pickle_in:
            #                     interface_height = pickle.load(pickle_in)

            #             for scale in ['front','back']:
            #                 setattr(img, scale + '_interface', interface_height[scale][img.time])

            #         except FileNotFoundError as e:
            #             print('Pickle files don''t exist, need to create by changing INTERFACE_HEIGHT = 1')

            print( str(count+1) + ' of ' + str(len(fnames)) + ' images processed in folder')

        # Write dataframes to pickle
        if DENSITY_PROFILES == 1:
            fname = video_loc + 'analysis/' + file_ext[1:] + '_density.pickle'
            with open(fname, 'wb') as pickle_out:        
                pickle.dump(density, pickle_out)        


        data = density['front'].xs(key = ('box','mean'),axis = 1, level = [1,2])
        # this will flip the dataframe and therefore hopefully the plot
        data_flip = data.sort_index(axis = 0, ascending = True)
        # drop rows which are higher than the door
        drop_lst = data_flip.index.values > door_scale.loc['door','front']
        data_flip_drop = data_flip[data_flip.index.values < door_scale.loc['door','front'] ] 

        threshold = 0.3

        # for col in data_flip_drop.columns:

        plot_filling_box_density(data_flip_drop, door_scale.loc['door','front'], thres = threshold, video_loc = video_loc)