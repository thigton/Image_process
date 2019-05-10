import os
import RAW_img
import PLOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D 
from lmfit.models import GaussianModel

def plume_time_ave(img, count, **kwargs):
    '''function will add new image to existing 
    set of averaged images and find the new time_average of the images'''
    def save_plume_img(df, img):
        if not os.path.exists(img.img_loc + 'analysis/plume_time_ave/'):
            os.makedirs(img.img_loc + 'analysis/plume_time_ave/')
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        cbaxes = fig.add_axes([0.9, 0.1275, 0.03, 0.74]) 
        image = ax1.imshow(df, cmap = 'inferno', vmin = kwargs['thres'][0] , vmax = kwargs['thres'][1])
        plt.colorbar(image, cax = cbaxes, orientation = 'vertical')
        ax1.axis('off')
        fname = f'{img.img_loc}analysis/plume_time_ave/{str(img.time)}_secs.png'
        fig.savefig(fname)
        plt.close()

        # plt.imsave(fname, df, vmin = kwargs['thres'][0], vmax = kwargs['thres'][1], cmap = 'viridis', dpi = 100)

    try:
        ave = kwargs['img_ave']
        ave *= count
        ave += img.plume
        ave /= (count + 1)

        save_plume_img(ave,img)
        return ave
    except KeyError:
        save_plume_img(img.plume, img)
        return img.plume


def plume_area_hist(img, **kwargs):
    '''saves a histogram of the plume area to see what the
    spread is and how you might want to rescale it.'''
    if not os.path.exists(img.img_loc +  'analysis/hist/'):
        os.makedirs(img.img_loc +  'analysis/hist/')
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,12))
    ax1.hist(img.plume.to_numpy().reshape(-1), bins =50 , range=(0, 1))
    hist_name = img.img_loc +'analysis/hist/' + img.filename + 'plume_area.png'
    ax1.set_title('Plume area Histogram - red channel')
    ax1.set_ylim([0,1000])
    image = ax2.imshow(img.plume, cmap = 'inferno', vmin = kwargs['thres'][0] , vmax = kwargs['thres'][1])
    plt.colorbar(image,ax = ax2, orientation = 'vertical')
    fig.savefig(hist_name)
    plt.close()

def analyse_plume(img_ave, rel_imgs_dir, **kwargs):
    '''Input - time_ave of the plume density images
    Produces a plot of the plume centre line and gaussian distributions'''
    # find plume centre line (max absorbance)
    plume_cl = img_ave.rolling(25, center = True, min_periods = 1, axis = 0).mean().idxmax(axis = 1)

    # position of centre line at the nozzle
    plume_cl_0 = plume_cl.iloc[0]

    fig, (ax1,ax2) = plt.subplots(2, 1, figsize = (9, 9))
    x_range = [500,1500]
    rows = np.arange(150,img_ave.shape[0]-100, 125)
    colors = ['red','blue','green','black','orange','yellow','purple']
    for row, color in zip(rows, colors):
        data = img_ave.iloc[row,:]
        # print(img_ave.columns.values)
        # exit()
        ax1.scatter(img_ave.columns[::25], data[::25],marker = 'x', color = color, 
        label = f'h/H: {img_ave.iloc[row].name:.2f}')
        mod = GaussianModel()
        pars = mod.guess(data, x=img_ave.columns.values) # guesses starting value for gaussian

        out = mod.fit(data, pars, x=img_ave.columns.values) # finds best fit of gaussian
        # print(out.fit_report(min_correl=0.25))
        ax1.plot(img_ave.columns.values, out.best_fit, color = color, ls = '--')
    ax1.legend()
    ax1.set_xlim(x_range)
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('Absorbance')
    ax1.set_title('Gaussian Fit to plume concentration')

    image = ax2.imshow(img_ave, cmap = 'inferno', vmin = kwargs['thres'][0] , vmax = kwargs['thres'][1])
    ax2.plot(plume_cl.rolling(25, center = True, min_periods = 1).mean(),range(len(plume_cl)), color = 'green', label = 'max concentration')
    ax2.plot([plume_cl_0]*2 ,[0, len(plume_cl)], color = 'green', ls = '--', label = 'plume origin centreline' )
    for row in rows:
        ax2.plot([0, img_ave.shape[1]], [row, row], color = 'red', lw = 2)
    plt.colorbar(image,ax = ax2, orientation = 'vertical')
    ax2.axis('on')
    ax2.legend()
    ax2.set_title('Time Averaged Plume')
    fname = f'{rel_imgs_dir}analysis/plume.png'
    fig.savefig(fname)
    plt.close()
    




if __name__ == '__main__':

    # OPTIONS
    GOT_PLUME_TIME_AVE = 1
    

    #CODE
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    theory_df = PLOT.import_theory() # import the theory steady state dataframe

    # data_loc = ['190405','190405_2']
    data_loc = ['190328','190328_3' ,'190329','190405','190405_2', '190405_3'] 

    for data in data_loc:
        
        rel_imgs_dir = './Data/' + data + '/' # File path relative to the script
        print(rel_imgs_dir)
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
        plume_absorbance_thres = (0.0, 0.15) # this is the range of absorbance to get a good image of the plume
        if GOT_PLUME_TIME_AVE == 1:
            try:
                with open(rel_imgs_dir + 'analysis/plume_time_ave/plume_time_ave.pickle', 'rb') as pickle_in:
                    image_time_ave = pickle.load(pickle_in)
                analyse_plume(image_time_ave,rel_imgs_dir, thres = plume_absorbance_thres)
                continue
            except FileNotFoundError:
                    print('Need to create plume_time_ave first.')

        for count, f in enumerate(filenames): # Analyse in reverse so that the crop images are of the steady state


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
             #crop image
            img.crop_img(crop_pos)
            #normalise images
            img.normalise(BG) 

            # Plume analysis ============================
            # define the door level , box top and bottom returns a dict
            try:
                with open(rel_imgs_dir + 'box_dims.pickle', 'rb') as pickle_in:
                    box_dims = pickle.load(pickle_in)
            except FileNotFoundError:
                print('No box dimensions have been previously defined for experiment: ' + data)
                print('Please run Execute.py to define box scales')

            # define and pickle the plume crop (initially based on the the theory steady state level)
            try:
                with open(rel_imgs_dir + 'plume_area.pickle', 'rb') as pickle_in:
                    plume_area = pickle.load(pickle_in)
            except FileNotFoundError as e:
                    img1 = RAW_img.Raw_img(rel_imgs_dir, filenames[-1], file_ext) 
                    img1.crop_img(crop_pos)
                    # img1.normalise(BG) 
                    print('Choose plume area...')
                    plume_area = img1.choose_crop()
                    with open(rel_imgs_dir + 'plume_area.pickle', 'wb') as pickle_out:
                        pickle.dump(plume_area, pickle_out)                                   

            # get the scales of analysis area in dimensionless form for both the front and back of the box.
            # Door level, vertical and horizontal scale, camera centre.
            try:
                with open(rel_imgs_dir + 'analysis/plume_scales.pickle', 'rb') as pickle_in:
                    plume_scales = pickle.load(pickle_in)
            except FileNotFoundError:
                plume_scales = RAW_img.make_dimensionless(img,box_dims,plume_area)
                with open(rel_imgs_dir + 'analysis/plume_scales.pickle', 'wb') as pickle_out:
                    pickle.dump(plume_scales, pickle_out) 
    
            vertical_scale = plume_scales[2]

            
            img.define_analysis_strips(plume_area, vertical_scale, plume = True)

            
            
            plume_area_hist(img, thres = plume_absorbance_thres)
            if count == 0:
                image_time_ave = plume_time_ave(img, count, thres = plume_absorbance_thres )
            elif count == len(filenames) - 1:
                image_time_ave = plume_time_ave(img, count, img_ave = image_time_ave, thres = plume_absorbance_thres)
                with open(rel_imgs_dir + 'analysis/plume_time_ave/plume_time_ave.pickle','wb') as pickle_out:
                    pickle.dump(image_time_ave, pickle_out)
            else:
                image_time_ave = plume_time_ave(img, count, img_ave = image_time_ave, thres = plume_absorbance_thres)

            
               
                
            # housekeeping
            print( str(count+1) + ' of ' + str(len(filenames)) + ' images processed in folder: ' + data + 
            ' - folder ' + str(data_loc.index(data) +1) + ' of ' + str(len(data_loc)) )
        
 