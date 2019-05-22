import os
import RAW_img
import PLOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D 
from lmfit.models import GaussianModel
from scipy.stats import linregress
import matplotlib.patches as patches
from scipy.signal import medfilt

def remove_img_background_noise(img_ave):
    '''will modify the the image to approximately find the edges of the plume.
    Everything outside will be made zero'''
    # smooth out in axes
    smoothed = img_ave.rolling(25, min_periods = 1, center = True, axis = 0).mean()
    smoothed = smoothed.rolling(25, min_periods = 1, center = True, axis = 1).mean()
	
    plt.ion()
    ax = plt.axes()
    level = [0.002]
    image = ax.imshow(smoothed, vmin = 0 , vmax = 0.15)
    plt.colorbar(image,ax = ax)
    thres = ax.contour(smoothed, level,colors='r', origin='lower')
    ax.clabel(thres, thres.levels, inline = True, fmt = '%1.4f')
    plt.draw()
    
    response = input('Are you happy threshold to remove background noise?  Do you want to continue? [Y/N]')
    while 'y' not in response.lower():
        level = [float(input('Enter new threshold:'))]
        thres = ax.contour(smoothed, level,colors='r', origin='lower')
        ax.clabel(thres, thres.levels, inline = True)
        plt.draw()
        response = input('Are you happy threshold to remove background noise?  Do you want to continue? [Y/N]')
    plt.ioff()
    plt.close()
    # get bool of all smoothed
    X = img_ave.to_numpy()
    X[ (smoothed.to_numpy() < level) | (X < 0)] = 0

    return pd.DataFrame(X,index = img_ave.index, columns = img_ave.columns)

def get_img_width(img_ave):
    return img_ave.columns.values.astype(np.float64)

def get_plume_gaussian_model(data,img_width):
    '''returns a single gaussian fit for the pd series provided
    data = horizontal row of plume time average df'''

    mod = GaussianModel()
    pars = mod.guess(data, x=img_width) # guesses starting value for gaussian
    out = mod.fit(data, pars, x=img_width) # finds best fit of gaussian  
    return out
        
def get_plume_edges(img_ave):
    '''Function will return the plume edges (index) and plume width (in terms of pixels)
     from the gaussian curve'''
    left = []
    right = []
    c = []
    img_width = get_img_width(img_ave)
    
    for i,row in enumerate(reversed(img_ave.index)):
        data = img_ave.iloc[row,:]
        mod = get_plume_gaussian_model(data, img_width)
        sigma = mod.best_values['sigma']
        center = mod.best_values['center']

        # some filters to deal with bubbles at the top of the image
        # if i == 0:
        #     c.append(round(center))
        #     edges.append( ( round(center - 2**(0.5)*sigma), round(center + 2**(0.5)*sigma) ) )
        # elif ( (center - c[-1]) / center > 0.1 ) or ( (center - 2**(0.5)*sigma - edges[-1][0]) / (center - 2**(0.5)*sigma) > 0.2):
        #     c.append(c[-1])
        #     edges.append( edges[-1] )
        # else:
        c.append( round(center) )
        left.append( round(center - 2**(0.5)*sigma) )
        right.append( round(center + 2**(0.5)*sigma) )

    c = medfilt(c, kernel_size=9)
    left = medfilt(left, kernel_size=9)
    right = medfilt(right, kernel_size=9)
    edges = [(l,r) for l,r in zip(left,right)]

    radius = [(center-edge[0],edge[1]-center) for edge, center in zip(edges, c)]
    return (c[::-1], radius[::-1])


def plume_edge_linear_regression(img_ave, radius,centre):
    '''Returns coefficients for a linear regression 
    for the plume width and distance from source  '''
    top = 100
    bot = 700
    # just going to look at the middle of the plume at the moment until data is cleaner
    left_r = [r[0] for r in radius[top:bot]]
    right_r = [r[1] for r in radius[top:bot]]
    y = img_ave.index[top:bot]
    # plume_width = [r[0]+r[1] for r in radius[top:bot]]
    left = linregress(left_r,y)
    right = linregress(right_r,y)
    # width = linregress(plume_width,y)
    print(5/(6*left.slope))
    print(5/(6*right.slope))
    fig, (ax1, ax2) = plt.subplots(1,2, sharey = True)
    ax1.plot(left_r, y , 'kx', label = 'left_r_raw')
    ax1.plot(left_r,[left.slope*r + left.intercept for r in left_r],'r',label = 'left_r')
    ax1.plot(right_r, y , 'g>', label = 'right_r_raw')
    ax1.plot(right_r,[right.slope*r + right.intercept for r in right_r],'b',label = 'right_r')
    # ax1.grid('on')
    ax1.set_xlabel('Plume radius (px)')
    ax1.set_ylabel('Plume height (px)')
    
    def average(lst): 
        return sum(lst) / len(lst)
    centre_ave = average(centre)
    plt.imshow(img_ave, vmin= 0, vmax = 0.15, aspect = 'auto')
    rect = patches.Rectangle( (centre_ave - 500 ,top), 1000, bot-top, linewidth = 1, edgecolor='r', facecolor = 'none')
    ax2.add_patch(rect)    
     
    plt.show()

def plot_plume_gaussian(img_ave, scaled = False):
    '''Function takes a list of gaussian fits from the plume and plots'''
    img_width = get_img_width(img_ave)
    for row in img_ave.index[400::400]:
        data = img_ave.iloc[row,:]
        mod = get_plume_gaussian_model(data, img_width)
        fig = plt.figure()
        mod.plot(xlabel = 'x (px)', ylabel = 'A', fig = fig, data_kws = {'markersize':2, 'marker':'o','color':'k'})
        plt.show()


def plot_plume_lines(img_ave):
    pass
    # image = ax2.imshow(img_ave, cmap = 'inferno', vmin = kwargs['thres'][0] , vmax = kwargs['thres'][1])
    # ax2.plot(plume_cl.rolling(25, center = True, min_periods = 1).mean(),range(len(plume_cl)), color = 'green', label = 'max concentration')
    # ax2.plot([plume_cl_0]*2 ,[0, len(plume_cl)], color = 'green', ls = '--', label = 'plume origin centreline' )
    # for row in rows:
    #     ax2.plot([0, img_ave.shape[1]], [row, row], color = 'red', lw = 2)
    # plt.colorbar(image,ax = ax2, orientation = 'vertical')




if __name__ == '__main__':


    plume_absorbance_thres = (0.0, 0.15) # this is the range of absorbance to get a good image of the plume
    
    
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    data_loc = ['190329']
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
                
                image_time_ave = remove_img_background_noise(image_time_ave)

                # plot_plume_gaussian(image_time_ave)

                
                centre, radius =  get_plume_edges(image_time_ave)

                plume_edge_linear_regression(image_time_ave, radius, centre)

                # plot_plume_lines(img_ave)

                

                plt.imshow(image_time_ave, vmin= 0, vmax = 0.15)
                plt.plot(centre,image_time_ave.index,'r', label = 'centre')
                plt.plot([c - r[0] for c,r in zip(centre, radius)],image_time_ave.index,'orange', label = 'left')
                plt.plot([c + r[1] for c,r in zip(centre, radius)],image_time_ave.index,'g', ls = '--', label = 'right')
                plt.legend()
                plt.show()
                exit()
            except FileNotFoundError:
                    print('Need to create plume_time_ave first.')