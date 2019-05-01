import os
import RAW_img
import PLOT
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def plume_time_ave(img, count, **kwargs):
    '''function will add new image to existing 
    set of averaged images and find the new time_average of the images'''
    def save_plume_img(df, img):
        fname = f'{img.img_loc}analysis/plume_time_ave/{str(img.time)}_secs.png'
        plt.imsave(fname, df, vmin = 0, vmax = 1, cmap = 'inferno', dpi = 100)

    try:
        ave = kwargs['img_ave']
        if not os.path.exists(img.img_loc + 'analysis/plume_time_ave/'):
            os.makedirs(img.img_loc + 'analysis/plume_time_ave/')
        ave *= count
        ave += img.plume
        ave /= (count + 1)
        save_plume_img(ave,img)
        return ave
    except KeyError:
        save_plume_img(img.plume, img)
        return img.plume








if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    theory_df = PLOT.import_theory() # import the theory steady state dataframe

    data_loc = ['190328']#, '190328_3' , '190329','190405','190405_2', '190405_3'] 

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

            if count == 0:
                image_time_ave = plume_time_ave(img, count)
            else:
                image_time_ave = plume_time_ave(img, count, img_ave = image_time_ave)
                
                
            # time average all the images