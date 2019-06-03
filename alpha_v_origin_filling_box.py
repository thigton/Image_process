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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import pi

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
    




def convert_vertices_to_df_index(y, dff):
    r = int(round(y))
    return 1 - dff.index[r]

def get_threshold_contour(df, thres, const):
    '''Function will interface height**(-2/3) against time and then do a linear regression on that'''
    # Create contour plot and get contour vertices
    cs = plt.contour(df, [thres])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    t = v[:,0]
    # Trim vertices to remove waves at start
    # t[t<50] = None

    y = v[:,1]
    # y[t<50] = None

    y = list(map(lambda k : convert_vertices_to_df_index(k,df),y))
    # exit()

    # close contour plot
    plt.close()
    return (t, y)

def plot_lin_regress(x,y,y_pred, details):
    '''Plot a single linear regression'''
    ent_coeff, v_origin, rms = details
    v_origin *= 0.3
    plt.scatter(x[::20],y[::20], marker = '+', label = 'data')
    plt.plot(x,y_pred, label = 'lin_reg')
    plt.xlabel('$time (s)$')
    plt.ylabel('$(Distance from nozzle/m)^{-2/3}$')
    plt.title(f'alpha: {ent_coeff:.4f} \n virtual origin: {v_origin:.2f}m \n RMS: {rms:.4f}')
    plt.show()

def get_ent_coeff(m,S,F):
    gamma = 1.07
    cm = 2*(gamma**2 + 1)**(1/3) * 3**(5/3) * 5**(-4/3) * pi**(2/3) * F**(1/3) / S
    alpha = (m / cm)**(0.75)
    return alpha

def ent_coeff_and_v_origin(df, thres, const):
    S, plume_F = const
    t, y = get_threshold_contour(df,thres,const)
    # Linear regression for a variety of virtual_origins
    t = np.array(t).reshape((-1, 1))
    v_origins = np.linspace(0,0.3,90)
    rms = []
    ent_coeff = []
    for v_origin in v_origins:
        y_full = list(map(lambda k : ((k + v_origin) * 0.3)**(-0.6666), y))
        lm = LinearRegression()
        try:
            lm.fit(t, np.array(y_full))
        except ValueError as e:
            print(e)
            print(t)
            print(sum(np.isnan(t)))
            print(sum(np.isnan(y_full)))
            exit()
        y_pred = lm.predict(t)
        m = lm.coef_
        ent_coeff.append(get_ent_coeff(m, S, plume_F)[0])
        rms.append(sqrt(mean_squared_error(y_full, y_pred)))
        details = (ent_coeff[-1], v_origin, rms[-1])
        plot_lin_regress(t, y_full, y_pred, details)

    rms = np.array(rms)
    ent_coeff = np.array(ent_coeff)
    # Save the plot
    # plt.plot(v_origins,rms, label = 'rms')
    # plt.plot(v_origins, ent_coeff, marker = '^', label = '$alpha_G - pred$')
    # plt.legend()
    # plt.show()
    # plt.close()
    # Find the minimum rms value for this threshold
    rms_min = np.amin(rms)
    ent_coeff_min = ent_coeff[np.where(rms == np.amin(rms))]
    v_origins_min = v_origins[rms == rms_min] 
    
    return (rms_min, ent_coeff_min, v_origins_min)
    

if __name__ == '__main__':

    S = 0.45 * 0.3 # bot horizontal cross sectional area m^2
    CAPTURE_FRAMES = 0
    DENSITY_PROFILES = 0

    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    video_loc = './Data/190521_3/'
    file_ext = '.jpg'
    fps = 50
    if CAPTURE_FRAMES == 1:
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
            plume_g = (img.sol_denisty - 0.998) * 9.81 / 0.998 # ms^-2
            plume_F = plume_g * plume_q # m^4s^-3
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
                with open(video_loc + file_ext[1:] + '_box_dims.pickle', 'rb') as pickle_in:
                    box_dims = pickle.load(pickle_in)

            except FileNotFoundError as e:
                print('Pickle files don''t exist, need to create by changing DENSITY PROFILES = 1')


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
    data_drop = data[data.index.values < door_scale.loc['door','front'] ] 
    
    # plot_filling_box_density(data_drop, door_scale.loc['door','front'])
    # exit()
    rms = []
    ent_coeff = []
    v_origins = []
    thresholds = np.linspace(0.1,0.5,50)
    for thres in thresholds :
        rms_min, ent_coeff_min, v_origins_min = ent_coeff_and_v_origin(data_drop, thres, const = (S, plume_F))
        rms.append(rms_min)
        ent_coeff.append(ent_coeff_min)
        v_origins.append(v_origins_min)

    plt.plot(thresholds,rms, label = 'rms')
    plt.plot(thresholds,ent_coeff, label = 'alpha')
    plt.plot(thresholds,v_origins, label = '$z_v$')
    plt.legend()
    plt.xlabel('Absorbance Threshold')
    plt.ylabel('RMS, entrainment coefficient, virtual origin')
    
    plt.show()
