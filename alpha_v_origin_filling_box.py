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
from savepdf_tex import savepdf_tex
import itertools
import matplotlib.style as style
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------#

def plot_filling_box_density(df, door, thres, video_loc):

    # max_grad = get_interface_max_gradient(df) 
    fig = plt.figure()
    ax = plt.axes()
    tick_spacing = 40
    colors = plt.pcolor(df, cmap='coolwarm', rasterized=True)
    plt.colorbar(colors, ax=ax, orientation='vertical', label=r'\$-\ln(\frac{I}{I_0})\$')
    ylabs = ["%.2f"%item for item in df.index.values[0::tick_spacing] ]
    xlabs = ["%.0f"%item for item in df.columns.values[0::tick_spacing] ]
    plt.yticks(np.arange(0.5, len(df.index), tick_spacing), ylabs)
    plt.xticks(np.arange(0.5, len(df.columns), tick_spacing), xlabs, rotation='vertical')
    CS = plt.contour(df, thres, colors='w')
    plt.clabel(CS, CS.levels[::1], inline=True)
    # door_idx = min(range(df.shape[0]), key=lambda i: abs(df.index[i]- door))
    # plt.plot([0, len(df.columns)], [door_idx, door_idx], color='r', label='door')
    # plt.plot(max_grad.index.values / time_between_img, max_grad, color = 'g', label = 'max grad' )
    plt.xlabel('Time (s)')
    plt.ylabel(r'\$h/H\$')
    # plt.legend(loc='upper left')
    savepdf_tex(fig, '/home/tdh17/Documents/BOX/PhD/03 Writing/03_Thesis/figs_using/', r'{}_ent_coeff_filling_box'.format(video_loc[7:-1]))
    plt.savefig(f'{video_loc}analysis/density.png', dpi=300)
    plt.close()


def convert_vertices_to_df_index(y, dff):
    r = int(round(y))
    return 1.0 - dff.index[r]
def convert_vertices_to_df_columns(t, dff):
    r = int(round(t))
    return dff.columns[r]

def get_threshold_contour(df, thres, const):
    '''Function will interface height**(-2/3) against time
    and then do a linear regression on that'''
    # Create contour plot and get contour vertices
    cs = plt.contour(df, [thres])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    t = v[:, 0]
    y = v[:, 1]
    y = np.fromiter((convert_vertices_to_df_index(k,df) for k in y), float)
    t = np.fromiter((convert_vertices_to_df_columns(k,df) for k in t), float)
    # y = np.array((map(lambda k : convert_vertices_to_df_index(k,df),y)))
    # t = list(map(lambda k : convert_vertices_to_df_columns(k, df),t))
    # Trim vertices to remove waves at start
    y = y[t > 25]
    t = t[t > 25]
    # close contour plot
    plt.close()
    return (t, y)

def plot_lin_regress(x, y, y_pred, details, video_loc):
    '''Plot a single linear regression'''
    
    ent_coeff, v_origin, rms = details
    fig = plt.figure()
    plt.scatter(x[::20], y[::20], marker='+', label='data', color='k')
    plt.plot(x, y_pred, label='lin_reg', ls='--', color='k')
    plt.xlabel(r'\$time (s)\$')
    plt.ylabel(r'\$h^{-2/3} (m^{-2/3})=\$')
    plt.grid(True)
    plt.text(50,5,r"\$ \alpha_{G}=\$"+ ' {:.4f} \n'.format(ent_coeff)+
                  r"\$z_v/H=\$"+ " {:.4f} \n".format(v_origin)+
                  r"\$E_{RMS}=\$"+ " {:.4f}".format(rms))
    plt.savefig(f'{video_loc}analysis/lin_regress_rms_min.png')
    savepdf_tex(fig, '/home/tdh17/Documents/BOX/PhD/03 Writing/03_Thesis/figs_using/', r'{}_ent_coeff_lin_regress'.format(video_loc[7:-1]))
    plt.close()

def get_ent_coeff(m, S, F):
    gamma = 1.07
    cm = 2.0**2.0 * (gamma**2.0 + 1.0)**(0.3333) * 3.0**(0.6666) * \
         5.0**(-1.3333) * pi**(0.6666) * F**(0.3333) / S
    alpha = (m / cm)**(0.75)
    return alpha

def ent_coeff_and_v_origin(df, thres, v_origin, time, y, **kwargs):
    # Linear regression for a variety of virtual_origins
    time = np.array(time).reshape((-1, 1))
    y_full = list(map(lambda k: ((k + v_origin) * 0.3)**(-0.6666), y))
    lm = LinearRegression()
    try:
        lm.fit(time, np.array(y_full))
    except ValueError as e:
        print(e)
        plt.plot(time, y_full)
        plt.show()
        exit()
    y_pred = lm.predict(time)
    m = lm.coef_
    ent_coeff = get_ent_coeff(m, S, plume_F)[0]
    rms = sqrt(mean_squared_error(y_full, y_pred))

    if 'plot_regression' in kwargs:
        try:
            details = (ent_coeff, v_origin, rms)
            plot_lin_regress(time, y_full, y_pred, details, kwargs['video_loc'])
        except KeyError:
            pass

    return (rms, ent_coeff)

def plot_2d_data(rms_df, ent_coeff_df, video_loc, **kwargs):
    '''Function will plot a slice, fixing either the v origin or the threshold value'''
    fig = plt.figure()
    plt.ylabel(r'\$ \alpha_{G}\$ and \$E_{rms}\$')
    
    if 'thres' and 'v_origin' in kwargs:
        print('only define the threshold or the virtual origin, not both')
    
    elif 'thres' in kwargs:
        rms = rms_df[kwargs['thres']]
        ent_coeff = ent_coeff_df[kwargs['thres']]
        x = rms_df.index
        plt.xlabel(r'\$ z_v/H \$')
        plt.text(0.08, 0.095, r'\$ -\ln\big(\frac{I}{I_0}\big)=\$'+ '{:0.3f}'.format(kwargs['thres']))
    elif 'v_origin' in kwargs:
        rms = rms_df.loc[kwargs['v_origin']]
        ent_coeff = ent_coeff_df.loc[kwargs['v_origin']]
        x = rms_df.columns
        plt.xlabel(r'\$ -\ln\frac{I}{I_0}, threshold value \$')
    
    # Save the plot
    plt.plot(x, rms, color='r', label='rms')
    plt.plot(x, ent_coeff, color='orange', label='$alpha_G - pred$')
    # plt.legend()
    plt.grid(True)
    plt.savefig(f'{video_loc}analysis/best_v_origin.png')
    if 'thres' in kwargs:
        thres_str = str(kwargs['thres'])[:5].replace('.','_')
        savepdf_tex(fig, '/home/tdh17/Documents/BOX/PhD/03 Writing/03_Thesis/figs_using/', '{}_thres_{}_ent_coeff'.format(video_loc[7:-1], thres_str))
    if 'v_origin' in kwargs:
        v_origin_str = str(kwargs['v_origin'])[:5].replace('.','_')
        savepdf_tex(fig, '/home/tdh17/Documents/BOX/PhD/03 Writing/03_Thesis/figs_using/', '{}_v_origin_{}_ent_coeff'.format(video_loc[7:-1], v_origin_str))
    plt.close()

def plot_min_each_thres(rms_df, ent_coeff_df, min_vals, video_loc):
    '''Plots the results for the minimum RMS for each threshold value chosen'''
    fig = plt.figure()
    rms_df = rms_df.astype('float64')
    ent_coeff_df = ent_coeff_df.astype('float64')
    v_origin_min = rms_df.idxmin(axis=0)
    thres_vals = v_origin_min.index
    rms_min = rms_df.min(axis=0)
    ent_coeff_min = pd.Series([ent_coeff_df.loc[z_v, thres] for (z_v, thres) in zip(v_origin_min, thres_vals)])
    plt.plot(thres_vals, rms_min, label='rms', color='r')
    plt.plot(thres_vals, ent_coeff_min, label='alpha', color='orange')
    plt.plot(thres_vals, v_origin_min, label='$z_v$', color='k')
    plt.grid(True)
    # plt.legend()
    plt.xlabel(r'\$ -\ln\big(\frac{I}{I_0}\big)\$' + '  threshold value')
    plt.ylabel(r'\$\alpha_{G}\$'+ ', ' + r'\$E_{rms}\$'+ ', ' + r'\$z_v\$')
    plt.xlim([thres_vals[0], thres_vals[-1]])
    plt.savefig(f'{video_loc}analysis/overall_graph.png')
    savepdf_tex(fig, '/home/tdh17/Documents/BOX/PhD/03 Writing/03_Thesis/figs_using/', r'{}_ent_coeff_min_vals'.format(video_loc[7:-1]))
    plt.close()

def plot_surface(df, video_loc, ent_coeff=True):
    '''will create a 3d surface plot of the entire space for either entrainment coefficient of the RMS'''
    fig = plt.figure(dpi = 150)
    ax1 = fig.add_subplot(1,1,1, projection='3d')
    X, Y = np.meshgrid(df.columns, df.index)
    ax1.plot_surface(X, Y, df, cmap= 'coolwarm',
                     linewidth=0, antialiased=False, rasterized=True)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # style.use('ggplot')
    CAPTURE_FRAMES = 0
    DENSITY_PROFILES = 0
    INTERFACE_HEIGHT = 0
    PLUME = 0
    GEN_ENT_COEFFS = 0
    PLOTTING = 1

    S = 0.45 * 0.3 # bot horizontal cross sectional area m^2

    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    video_loc = './Data/190521_2/'
    file_ext = '.jpg'
    fps = 50
    time_between_img = 0.5 # seconds
    plume_absorbance_thres = (0.0, 0.15)
    if CAPTURE_FRAMES == 1:
        # create jpgs frames
        video_to_frames(video_loc, '00000.MTS', image_ext=file_ext, 
                        video_fps=fps, spacing=time_between_img, start_time=0)

    # Get list of file names
    file_ids = raw_img.get_image_fid(video_loc, file_ext)
    FNAMES = file_ids[file_ext[1:]]

    # load in camera matrix
    with open(f'{video_loc[:7]}cam_mtx.pickle', 'rb') as pickle_in:
        camera_params = pickle.load(pickle_in)

    # Get background images
    BG_IDS = raw_img.get_image_fid(video_loc + 'BG/', file_ext)
    BG_FNAMES = BG_IDS[file_ext[1:]]

    (BG, CROP_POS, box_dims) = raw_img.prep_background_imgs(
            [raw_img.raw_img(video_loc + 'BG/',
                             f, file_ext) for f in BG_FNAMES], camera_params)
    for count, f in enumerate(FNAMES):
        # Preprocess the image
        ##########################################

        img = raw_img.raw_img(video_loc, f, file_ext)
        img.get_experiment_conditions()
        if count == 0:
            plume_q = (img.plume_q*1e-06) / 60 # m^3s^-1
            plume_g = (img.sol_denisty - 0.998) * 9.81 / 0.998 # ms^-2
            plume_F = plume_g * plume_q # m^4s^-3
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
            img1 = raw_img.raw_img(video_loc, FNAMES[-1], file_ext)
            img1.crop_img(CROP_POS)
            print('''Choose analysis area... \n
                  Ensure the top and bottom are within the depth of the box''')
            analysis_area = img1.choose_crop()
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



        # Define the same for the plume area
        ##########################################

        try:
            with open(video_loc + file_ext[1:] + '_plume_area.pickle', 'rb') as pickle_in:
                plume_area = pickle.load(pickle_in)
        except FileNotFoundError as e:
            img1 = raw_img.raw_img(video_loc, FNAMES[-1], file_ext)
            img1.crop_img(CROP_POS)
            # img1.normalise(BG) 
            print('Choose plume area...')
            plume_area = img1.choose_crop()
            with open(video_loc + file_ext[1:] + '_plume_area.pickle', 'wb') as pickle_out:
                pickle.dump(plume_area, pickle_out)                                   
        # get the scales of analysis area in dimensionless form
        # for both the front and back of the box.
        # Door level, vertical and horizontal scale, camera centre.
        try:
            with open(video_loc + 'analysis/' + file_ext[1:] +
                      '_plume_scales.pickle', 'rb') as pickle_in:
                plume_scales = pickle.load(pickle_in)
        except FileNotFoundError:
            plume_scales = raw_img.make_dimensionless(img, box_dims, plume_area)
            with open(video_loc + 'analysis/' + file_ext[1:] +
                      '_plume_scales.pickle', 'wb') as pickle_out:
                pickle.dump(plume_scales, pickle_out) 

        plume_scale = plume_scales[2]

        # Plume Time Averaging
        ##########################################

        if PLUME == 1:
            img.define_analysis_strips(plume_area, plume_scale, plume=True)
            if count == 0:
                image_time_ave = plume_time_ave(img, count, thres=plume_absorbance_thres)

            elif count == len(FNAMES) - 1: # pickle the last time average
                image_time_ave = plume_time_ave(img, count, img_ave=image_time_ave,
                                                save=True, thres=plume_absorbance_thres)
                with open(f'''{img.img_loc}analysis/plume_time_ave/
                           {str(img.time)}_secs.pickle''', 'wb') as pickle_out:
                    pickle.dump(image_time_ave, pickle_out)
            else:
                image_time_ave = plume_time_ave(img, count,
                                                img_ave=image_time_ave,
                                                thres=plume_absorbance_thres)


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


        if (count % 25 == 0) and count >= 75:
            raw_img.grad_2_plot(img, vertical_scale, rolling_mean=20)

        # If there is no preprocessing to do.  break to analysis section
        if (DENSITY_PROFILES == 0) and (INTERFACE_HEIGHT == 0) and (PLUME == 0):
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


    # Analysis Section
    ##########################################

    DATA = density['front'].xs(key=('box', 'mean'), axis=1, level=[1, 2])
    # this will flip the DATAframe and therefore hopefully the plot
    DATA_FLIP = DATA.sort_index(axis=0, ascending=True)
    # drop rows which are higher than the door
    DATA_FLIP_DROP = DATA_FLIP[DATA_FLIP.index.values < door_scale.loc['door', 'front']]
    DATA_DROP = DATA[DATA.index.values < door_scale.loc['door', 'front']]

    if GEN_ENT_COEFFS == 1:
        THRESHOLDS = np.linspace(0.10, 0.5, 80)
        v_origins = np.linspace(0, 0.15, 150)
        RMS = pd.DataFrame(index=v_origins, columns=THRESHOLDS)
        ENT_COEFF = pd.DataFrame(index=v_origins, columns=THRESHOLDS)
        for thres in THRESHOLDS:
            time, y = get_threshold_contour(DATA_DROP, thres, (S, plume_F))
            for z_v in v_origins:
                RMSx , ENT_COEFFx = ent_coeff_and_v_origin(DATA_DROP, thres, z_v, time, y)
                RMS.loc[z_v, thres] = RMSx
                ENT_COEFF.loc[z_v, thres] = ENT_COEFFx
        
        # get the final values RMS_min
        MIN_VALS = {'rms': RMS.min().min(),
            'v origin': RMS.min(axis=1).idxmin(),
            'thres': RMS.min(axis=0).idxmin(),
            'ent coeff': ENT_COEFF.loc[RMS.min(axis=1).idxmin(), RMS.min(axis=0).idxmin()]}
        

        with open(f'{video_loc}analysis/rms.pickle', 'wb') as pickle_out:
            pickle.dump(RMS, pickle_out)
        with open(f'{video_loc}analysis/ent_coeff.pickle', 'wb') as pickle_out:
            pickle.dump(ENT_COEFF, pickle_out)
        with open(f'{video_loc}analysis/min_vals.pickle', 'wb') as pickle_out:
            pickle.dump(MIN_VALS, pickle_out)

    else:
        try:
            with open(f'{video_loc}analysis/rms.pickle', 'rb') as pickle_in:
                RMS = pickle.load(pickle_in)
            with open(f'{video_loc}analysis/ent_coeff.pickle', 'rb') as pickle_in:
                ENT_COEFF = pickle.load(pickle_in)
            with open(f'{video_loc}analysis/min_vals.pickle', 'rb') as pickle_in:
                MIN_VALS = pickle.load(pickle_in)
        except FileNotFoundError:
            print('Need to run code to generate values first. turn GEN_ENT_COEFFS = 1')
            exit()

    if PLOTTING == 1:

        # plot_filling_box_density(DATA_FLIP_DROP, door_scale.loc['door', 'front'],
                            #   [THRESHOLDS[0], THRESHOLDS[-1]], video_loc)

        # rerun the function to plot the important graphs
        time, y = get_threshold_contour(DATA_DROP, MIN_VALS['thres'], (S, plume_F))
        _ = ent_coeff_and_v_origin(DATA_DROP, MIN_VALS['thres'], MIN_VALS['v origin'], time, y,
                                          plot_regression=True, video_loc=video_loc)

        plot_2d_data(RMS, ENT_COEFF, video_loc, thres=MIN_VALS['thres'])

        plot_min_each_thres(RMS, ENT_COEFF, MIN_VALS, video_loc)
       
        # plot_surface(RMS, video_loc, ent_coeff=True)
        