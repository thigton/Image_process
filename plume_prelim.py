'''Script is run to look at the plume of a standard experiment
   Will create:
   Crop area for the plume
   Time averaged images of the plume'''
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import raw_img
import plot

def plume_time_ave(img, count, save=False, **kwargs):
    '''function will add new image to existing
    set of averaged images and find the new time_average of the images'''
    def save_plume_img(dataf, img):
        if not os.path.exists(img.img_loc + 'analysis/plume_time_ave/'):
            os.makedirs(img.img_loc + 'analysis/plume_time_ave/')
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        cbaxes = fig.add_axes([0.9, 0.1275, 0.03, 0.74])
        image = ax1.imshow(dataf, cmap='inferno', vmin=kwargs['thres'][0], vmax=kwargs['thres'][1])
        plt.colorbar(image, cax=cbaxes, orientation='vertical')
        fname = f'{img.img_loc}analysis/plume_time_ave/{str(img.time)}_secs.png'
        fig.savefig(fname)
        plt.close()
        # plt.imsave(fname, dataf, vmin = kwargs['thres'][0],
        #            vmax = kwargs['thres'][1], cmap = 'viridis', dpi = 100)
    try:
        # omit rows in img.plume if they look like they are below the interface height

        ave = kwargs['img_ave']
        def include_in_plume_ave(df_row):
            '''Function will return a bool of whether a
            row should be used in the plume_time_ave
            This is done by looking at the edge pixels,
            if above a threshold, don't include the row'''
            if df_row.mean() > 0.05: #or (df_row[-10:].mean() > 0.2):
                return False
            else:
                return True
        row_bools = img.plume.apply(include_in_plume_ave, axis=1).to_numpy()
        try:
            first_false = [i for i, x in enumerate(row_bools) if not x][0]
        except IndexError:
            first_false = img.plume.shape[0]
        row_bools[(first_false - 80):] = False
        edited_img = img.plume
        for bol, row in zip(row_bools, img.plume.index):
            # if the row_bool is true, do nothing (incude the row in the edited img to average)
            if bol:
                continue
            # if not, input the current time_averaged value
            else:
                edited_img.loc[row, :] = ave.loc[row, :]
        ave = kwargs['img_ave']
        ave *= count
        ave += edited_img
        ave /= (count + 1)
        if save:
            save_plume_img(ave, img)
        return ave
    except KeyError:
        if save:
            save_plume_img(img.plume, img)
        return img.plume


def plume_area_hist(img, **kwargs):
    '''saves a histogram of the plume area to see what the
    spread is and how you might want to rescale it.'''
    if not os.path.exists(img.img_loc +  'analysis/hist/'):
        os.makedirs(img.img_loc +  'analysis/hist/')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.hist(img.plume.to_numpy().reshape(-1), bins=50, range=(0, 1))
    hist_name = img.img_loc +'analysis/hist/' + img.filename + 'plume_area.png'
    ax1.set_title('Plume area Histogram - red channel')
    ax1.set_ylim([0, 1000])
    image = ax2.imshow(img.plume, cmap='inferno', vmin=kwargs['thres'][0], vmax=kwargs['thres'][1])
    plt.colorbar(image, ax=ax2, orientation='vertical')
    fig.savefig(hist_name)
    plt.close()



if __name__ == '__main__':

    # OPTIONS
    GOT_PLUME_TIME_AVE = 0
    #CODE
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    THEORY_DF = plot.import_theory() # import the theory steady state dataframe



    DATA_LOC = ['190521_4']
    # DATA_LOC = ['190405']# ,'190329','190405','190405_2', '190405_3']
    for data in DATA_LOC:
        rel_imgs_dir = './Data/' + data + '/' # File path relative to the script
        file_ext = '.jpg'

        # load in camera matrix
        with open(f'{rel_imgs_dir[:7]}cam_mtx.pickle', 'rb') as pickle_in:
            camera_params = pickle.load(pickle_in)
        # Get list of file names
        file_ids = raw_img.get_image_fid(rel_imgs_dir, file_ext)
        filenames = file_ids[file_ext]
        # Get background images
        BG_ids = raw_img.get_image_fid(rel_imgs_dir + 'BG/', file_ext)
        BG_filenames = BG_ids[file_ext]
        #crop background imgs and get crop_pos
        (BG, crop_pos, box_dims) = raw_img.prep_background_imgs(
            [raw_img.raw_img(rel_imgs_dir + 'BG/',
                             f, file_ext) for f in BG_filenames], camera_params)
        #----------------------------------------------------------------
        plume_absorbance_thres = (0.0, 0.15)
        for COUNT, f in enumerate(filenames):
            # Image preprocessing ========================
             # import image
            IMG = raw_img.raw_img(rel_imgs_dir, f, file_ext)
            IMG.get_experiment_conditions()
            IMG.convert_centre_pixel_coordinate(crop_pos)
            if COUNT == 0:
                metadata = IMG.get_metadata()
                IMG.get_time()
                t0 = IMG.time # intial time
            else:
                IMG.get_time(t0)

            if file_ext == '.ARW':
                IMG.undistort(camera_params)
                IMG.black_offset(metadata['BlackLevel'], method=0)
                # realign


             #crop image
            IMG.crop_img(crop_pos)
            #normalise images
            IMG.normalise(BG)

            # Plume analysis ============================

            # define and pickle the plume crop
            # (initially based on the the theory steady state level)
            try:
                with open(rel_imgs_dir + file_ext[1:] + '_plume_area.pickle', 'rb') as pickle_in:
                    plume_area = pickle.load(pickle_in)
            except FileNotFoundError:
                IMG1 = raw_img.raw_img(rel_imgs_dir, filenames[-1], file_ext)
                IMG1.crop_img(crop_pos)
                # IMG1.normalise(BG)
                print('Choose plume area...')
                plume_area = IMG1.choose_crop(plume = True)
                with open(rel_imgs_dir + file_ext[1:] + '_plume_area.pickle', 'wb') as pickle_out:
                    pickle.dump(plume_area, pickle_out)

            # get the scales of analysis area in dimensionless
            # form for both the front and back of the box.
            # Door level, vertical and horizontal scale, camera centre.
            try:
                with open(rel_imgs_dir + 'analysis/' + file_ext[1:] +
                          '_plume_scales.pickle', 'rb') as pickle_in:
                    plume_scales = pickle.load(pickle_in)
            except FileNotFoundError:
                plume_scales = raw_img.make_dimensionless(IMG, box_dims, plume_area, plume=True)
                with open(rel_imgs_dir + 'analysis/' + file_ext[1:] +
                          '_plume_scales.pickle', 'wb') as pickle_out:
                    pickle.dump(plume_scales, pickle_out)
            plume_scale = plume_scales[4]
            IMG.define_analysis_strips(plume_area, plume_scale, plume=True)
            plume_area_hist(IMG, thres=plume_absorbance_thres)
            if COUNT == 0:
                image_time_ave = plume_time_ave(IMG, COUNT, thres=plume_absorbance_thres)

            # time has been picked out manually as a good
            # time average before the centreline starts to move
            elif (COUNT == len(filenames) - 1) or (IMG.time == 317):
                image_time_ave = plume_time_ave(IMG, COUNT, save = True,
                                                img_ave=image_time_ave,
                                                thres=plume_absorbance_thres)
                with open(f'{IMG.img_loc}analysis/plume_time_ave/{str(IMG.time)}_secs.pickle',
                          'wb') as pickle_out:
                    pickle.dump(image_time_ave, pickle_out)

            else:
                image_time_ave = plume_time_ave(IMG, COUNT, save = True,
                                                img_ave=image_time_ave,
                                                thres=plume_absorbance_thres)
            # housekeeping
            print(str(COUNT+1) + ' of ' + str(len(filenames)) +
                  ' images processed in folder: ' + data +
                  ' - folder ' + str(DATA_LOC.index(data) +1) + ' of ' + str(len(DATA_LOC)))
