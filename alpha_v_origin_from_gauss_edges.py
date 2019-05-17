import os
import RAW_img
import PLOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D 
from lmfit.models import GaussianModel


def get_plume_centreline(img_ave):
    '''Returns the maximum absorbance in each pixel row
    Add some filters to only get the plume'''
    # find plume centre line (max absorbance)
    plume_cl = img_ave.rolling(25, center = True, min_periods = 1, axis = 0).mean().idxmax(axis = 1)
    # position of centre line at the nozzle
    plume_cl_0 = plume_cl.iloc[0]
    return (plume_cl, plume_cl_0)


def get_plume_gaussian_curves(img_ave):
    '''returns a list of gaussian fit curves for every pixel row'''
    curves = []
    for row in img_ave.index[200::200]:
        data = img_ave.iloc[row,:]
        mod = GaussianModel()
        pars = mod.guess(data, x=img_ave.columns.values.astype(np.float64)) # guesses starting value for gaussian
        out = mod.fit(data, pars, x=img_ave.columns.values.astype(np.float64)) # finds best fit of gaussian
        curves.append(out.best_fit)
        plt.plot(img_ave.columns.values, out.best_fit, ls = '--', label = str(row))
        plt.xlim([500,1500])
    plt.legend()
    plt.show()
        
def get_plume_edges(img_ave):
    '''Function will return the plume edges (index) and plume width (in terms of pixels)
     from the gaussian curve'''  
    pass

def plume_edge_linear_regression():
     '''Returns coefficients for a linear regression 
     for the plume width and distance from source  '''
     pass
        
def entrainment_coefficient():
    '''returns a value for the entrainment coefficient'''
    pass

def virtual_origin():
    '''returns a value for the virtual origin'''
    pass

def plot_plume_gaussian(img_ave, scaled = False):
    '''Function takes a list of gaussian fits from the plume and plots'''
    pass
    # ax1.scatter(img_ave.columns[::25], data[::25],marker = 'x', color = color, 
    #     label = f'h/H: {img_ave.iloc[row].name:.2f}')
    # # print(out.fit_report(min_correl=0.25))
    # ax1.plot(img_ave.columns.values, out.best_fit, color = color, ls = '--')
    # ax1.legend()
    # ax1.set_xlim(x_range)
    # ax1.set_xlabel('pixels')
    # ax1.set_ylabel('Absorbance')
    # ax1.set_title('Gaussian Fit to plume concentration')


def plot_plume_lines(img_ave):
    pass
    # image = ax2.imshow(img_ave, cmap = 'inferno', vmin = kwargs['thres'][0] , vmax = kwargs['thres'][1])
    # ax2.plot(plume_cl.rolling(25, center = True, min_periods = 1).mean(),range(len(plume_cl)), color = 'green', label = 'max concentration')
    # ax2.plot([plume_cl_0]*2 ,[0, len(plume_cl)], color = 'green', ls = '--', label = 'plume origin centreline' )
    # for row in rows:
    #     ax2.plot([0, img_ave.shape[1]], [row, row], color = 'red', lw = 2)
    # plt.colorbar(image,ax = ax2, orientation = 'vertical')
    # ax2.axis('on')
    # ax2.legend()
    # ax2.set_title('Time Averaged Plume')
    # fname = f'{rel_imgs_dir}analysis/plume.png'
    # fig.savefig(fname)
    # plt.close()


if __name__ == '__main__':


    plume_absorbance_thres = (0.0, 0.15) # this is the range of absorbance to get a good image of the plume
    
    
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    data_loc = ['190405_3']
    # data_loc = ['190328','190328_3' ,'190329','190405','190405_2', '190405_3']   
    
    for data in data_loc:
        rel_imgs_dir = './Data/' + data + '/analysis' # File path relative to the script

        pickles = [x for x in  os.listdir(f'{rel_imgs_dir}/plume_time_ave') if x.endswith('.pickle')]

        for p in pickles:
            try:
                with open(f'{rel_imgs_dir}/plume_time_ave/{p}', 'rb') as pickle_in:
                    image_time_ave = pickle.load(pickle_in)
                
                # Get rid of index for now as I think it will make it easier
                image_time_ave.reset_index(inplace = True)
                image_time_ave.drop(columns = ['h/H'], inplace = True)
                
                # Plume centreline
                (plume_cl, plume_cl_0) = get_plume_centreline(image_time_ave)
                
                #Plume Gaussian fits
                get_plume_gaussian_curves(image_time_ave)
            except FileNotFoundError:
                    print('Need to create plume_time_ave first.')