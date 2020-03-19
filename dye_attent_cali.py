import os
import pickle
import time
import pandas as pd
import numpy as np
import pickle
import raw_img
import plot
import matplotlib.pyplot as plt
from EFB_unbalanced_theory.caseA import caseA
#-------------------------------------------#
class dye_attenuation_images(raw_img.raw_img):
    """Subclass of main RAW_imgs type

    Parent Class:
        raw_img {obj} -- main image processing class

    """


    def get_stats_on_analysis_area(self, analysis_area, conc_df):
        """function will return the stats for the analysis area for the dye attentuation calibrations.
        Stats - min / max, mean, std
        Get stats for horizontal extent, vertical extent and overall.  So we can check the fluid was well mixed.

        Arguments:
            analysis_area {dict} -- the area of analysis specified
            conc_df {[type]} -- data frame with the data about the ppm of dye
        """
        x1 = analysis_area['x1'] ; y1 = analysis_area['y1']
        x2 = analysis_area['x1'] + analysis_area['width']
        y2 = analysis_area['y1'] + analysis_area['height']
        for channel in ['red','green','blue']:

            datapoints = getattr(self, channel)[y1:y2, x1:x2]

            absorbance = (0 - np.log(datapoints)) / 30 # absorbance per unit length
            image_no = self.filename[-5:].lstrip("0")
            image_no = int(image_no)

            conc_df.loc[image_no-1,(channel, ['max', 'h max', 'v max'])] = [absorbance.max(), absorbance.max(axis=1),absorbance.max(axis=0)]
            conc_df.loc[image_no-1,(channel, ['min', 'h min', 'v min'])] = [absorbance.min(), absorbance.min(axis=1),absorbance.min(axis=0)]
            conc_df.loc[image_no-1,(channel, ['mean', 'h mean', 'v mean'])] = [absorbance.mean(), absorbance.mean(axis=1),absorbance.mean(axis=0)]
            conc_df.loc[image_no-1,(channel, ['std', 'h std', 'v std'])] = [absorbance.std(), absorbance.std(axis=1),absorbance.std(axis=0)]
        return conc_df


def plot_checks_for_well_mixed(conc):
    fig1 , ax1 = plt.subplots(1,2, figsize=(12,6))
    fig2 , ax2 = plt.subplots(2,2, figsize=(8,8))
    for i in conc.index:
        if isinstance(conc.loc[i,('red','h mean')], float):
            continue
        ax1[0].plot(conc.loc[i, ('red','h mean')], label=conc.loc[i,'image'])

        ax1[0].fill_between(range(len(conc.loc[i, ('red','h mean')])),
                            conc.loc[i, ('red','h mean')] - 2*conc.loc[i, ('red','h std')],
                            conc.loc[i, ('red','h mean')] + 2*conc.loc[i, ('red','h std')], alpha=0.2)
        ax1[1].plot(conc.loc[i, ('red','v mean')], label=conc.loc[i,'image'])
        ax1[1].fill_between(range(len(conc.loc[i, ('red','v mean')])),
                            conc.loc[i, ('red','v mean')] - 2*conc.loc[i, ('red','v std')],
                            conc.loc[i, ('red','v mean')] + 2*conc.loc[i, ('red','v std')], alpha=0.2)

        ax2[0,0].plot(conc.loc[i, ('red','h min')], label=conc.loc[i,'image'])
        ax2[0,1].plot(conc.loc[i, ('red','v min')], label=conc.loc[i,'image'])
        ax2[1,0].plot(conc.loc[i, ('red','h max')], label=conc.loc[i,'image'])
        ax2[1,1].plot(conc.loc[i, ('red','v max')], label=conc.loc[i,'image'])
    plt.suptitle('red channel')
    ax1[0].set_title('horizontal mean')
    ax1[1].set_title('vertical mean')
    ax2[0,0].set_title('horizontal min')
    ax2[0,1].set_title('vertical min')
    ax2[1,0].set_title('horizontal max')
    ax2[1,1].set_title('vertical max')
    fig1.savefig(f'{rel_imgs_dir}analysis/data_validation.png', dpi=300)
    fig2.savefig(f'{rel_imgs_dir}analysis/data_validation_2.png', dpi=300)
    plt.close()
    plt.close()

def plot_calibration_graph(conc):
    # determine best polynomial fit
    degree_of_fit = [1,2,3]
    styles = [':','-','-.']
    min_var = 1e6
    x = conc['dye conc (ppm)'].to_numpy(dtype='float')
    for channel in ['red','green','blue']:

        y_true = conc[(channel,'mean')].to_numpy(dtype='float')
        ax = plt.axes()
        for degree, sty in zip(degree_of_fit, styles):
            # color = next(ax._get_lines.prop_cycler)['color']
            z = np.polyfit(x, y_true, degree, full=True)
            ax.plot(x, np.polyval(z[0], x), label=f'poly d = {degree}',
              color=channel,ls=sty)
            var = z[1]/(len(conc['dye conc (ppm)'])-degree-1)
            if var < min_var:
                min_var = var


        ax.errorbar(conc['dye conc (ppm)'], conc[(channel, 'mean')], yerr=4*conc[(channel, 'std')],
                        lw=0, elinewidth=1, marker='x', capsize=3, color=channel)

    ax.set_xlabel(r"c (ppm)")
    ax.set_ylabel(r'\$A/b [cm^-1]\$')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.savefig(f'{rel_imgs_dir}analysis/calibration_curve.png', dpi=500)
    plt.close()

def get_best_fit_coeff(conc, data):
    """Saves coefficient data to the coefficient dictionary
    Arguments:
        conc {df} -- dataframe of concentrations
        data {str} -- location of data

    Saves:
        coeffs {dict} -- with structure {sol no : {date collected : {rgb : coeffs}}}}

        red channel - 2nd degree polynomial
        green channel - 1st degree polynomial
        blue channel - 1st degree polynomial
    """

    try:
        with open("./Data/dye_cali_coeff.pickle", "rb") as pickle_in:
            coeffs = pickle.load(pickle_in)
    except FileNotFoundError:
        coeffs = {}

    degrees = {'red' : 2, 'green': 1, 'blue': 1}
    x = conc['dye conc (ppm)'].to_numpy(dtype='float')
    sol_no = data.split('_')[2]
    date = data.split('_')[0]

    if not sol_no in coeffs:
        coeffs[sol_no] = {}
    if not date in coeffs[sol_no]:
        coeffs[sol_no][date] = {}

    for channel , i in degrees.items():
        y_true = conc[(channel,'mean')].to_numpy(dtype='float')
        z = np.polyfit(y_true, x, i, full=True)
        coeffs[sol_no][date][channel] = z[0]

    with open("./Data/dye_cali_coeff.pickle", "wb") as pickle_out:
        pickle.dump(coeffs, pickle_out)


if __name__ == '__main__':
    #pylint: disable=no-member

    FILE_EXT = '.ARW'
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    DATA_LOC = ['200131_sol_9_dye_cali']

    for data in DATA_LOC:
        rel_imgs_dir = './Data/' + data + '/' # File path relative to the script
        try:

            with open(f'{rel_imgs_dir}analysis/data.pickle', 'rb') as pickle_in:
                conc = pickle.load(pickle_in)

        except FileNotFoundError:
            print('Making data.pickle')
            rel_imgs_dir = './Data/' + data + '/' # File path relative to the script

            conc_data = pd.read_csv(rel_imgs_dir + 'conc.csv')
            conc_data = conc_data[['image','dye conc (ppm)']]
            channels = ['red','green','blue']
            stats = ['max','h max', 'v max',
                     'min','h min', 'v min',
                     'mean','h mean', 'v mean',
                     'std','h std', 'v std']
            multi_col = pd.MultiIndex.from_product([channels, stats],)
            conc = pd.DataFrame(columns=multi_col)
            conc[conc_data.columns] = conc_data


            # load in camera matrix
            with open(f'{rel_imgs_dir[:7]}cam_mtx.pickle', 'rb') as pickle_in:
                camera_params = pickle.load(pickle_in)

            # Get list of file names
            file_ids = raw_img.get_image_fid(rel_imgs_dir, FILE_EXT)
            filenames = file_ids[FILE_EXT[1:]]
            # Get background images
            BG_ids = raw_img.get_image_fid(rel_imgs_dir + 'BG/', FILE_EXT)
            BG_filenames = BG_ids[FILE_EXT[1:]]

            #crop background imgs and get crop_pos
            (BG, crop_pos, box_dims) = raw_img.prep_background_imgs(
                [raw_img.raw_img(rel_imgs_dir + 'BG/',
                                 f, FILE_EXT) for f in BG_filenames], camera_params)
            #----------------------------------------------------------------



            # Analyse in reverse so that the crop images are of the steady state
            for count, f in enumerate(filenames):
                try:
                    img = dye_attenuation_images(rel_imgs_dir, f, FILE_EXT)
                except:
                    print(f'Failed to read the RAW image: {f}')
                if count == 0:
                    metadata = img.get_metadata()
                img.convert_centre_pixel_coordinate(crop_pos)
                img.undistort(camera_params)
                img.black_offset(metadata['BlackLevel'], method=0)
                img.crop_img(crop_pos)
                img.normalise(BG)


                if count == 0:
                    try:
                        with open(rel_imgs_dir + FILE_EXT[1:] +
                                  '_analysis_area.pickle', 'rb') as pickle_in:
                            analysis_area = pickle.load(pickle_in)
                    except FileNotFoundError:
                        img1 = dye_attenuation_images(rel_imgs_dir, filenames[-1], FILE_EXT)
                        if FILE_EXT == '.ARW':
                            img1.undistort(camera_params)
                            img1.black_offset(metadata['BlackLevel'], method=0)
                        img1.crop_img(crop_pos)
                        img1.normalise(BG)
                        print('''Choose analysis area... \n
                              Ensure the top and bottom are within the depth of the box''')
                        analysis_area = img1.choose_crop()

                        with open(rel_imgs_dir + FILE_EXT[1:] +
                                  '_analysis_area.pickle', 'wb') as pickle_out:
                            pickle.dump(analysis_area, pickle_out)
                        del img1


                conc = img.get_stats_on_analysis_area(analysis_area, conc)

                print(str(count+1) + ' of ' + str(len(filenames)) +
                      ' images processed in folder: ' + data +
                      ' - folder ' + str(DATA_LOC.index(data) +1) + ' of ' + str(len(DATA_LOC)))


            with open(f'{rel_imgs_dir}analysis/data.pickle', 'wb') as pickle_out:
                pickle.dump(conc, pickle_out)
        plot_calibration_graph(conc)
        plot_checks_for_well_mixed(conc)
        get_best_fit_coeff(conc, DATA_LOC[0])