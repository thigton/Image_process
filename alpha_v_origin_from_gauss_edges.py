'''Script will analyse data from plume prelims
    main aim is to get the entrainment coefficient and virtual origin
    by directly measuring the plume edge using a gaussian fit'''
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
from scipy.stats import linregress
from scipy.signal import medfilt
import pandas as pd

def remove_img_background_noise(img_ave):
    '''will modify the the image to approximately find the edges of the plume.
    Everything outside will be made zero'''
    # smooth out in axes
    smoothed = img_ave.rolling(25, min_periods=1, center=True, axis=0).mean()
    smoothed = smoothed.rolling(25, min_periods=1, center=True, axis=1).mean()
    vert_mean = img_ave.mean(axis=0)
    plt.ion()
    level = [0.002]
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3,3), (1,0), colspan=3, rowspan=2, sharex=ax1)
    ax1.plot(smoothed.columns,vert_mean, color='black')
    ax1.set_xlim([smoothed.columns[0], smoothed.columns[-1]])
    line  = ax1.plot(smoothed.columns, [level]*len(smoothed.columns), color='red')
    ax1.set_ylabel('pixel column mean')
    level = [0.002]
    ax2.imshow(smoothed, vmin=0, vmax=0.15, aspect='auto')
    thres = ax2.contour(smoothed, level, colors='r', origin='lower')
    ax2.clabel(thres, thres.levels, inline=True, fmt='%1.4f')
    plt.draw()

    response = input('''Are you happy threshold to remove background noise?
                     Do you want to continue? [Y/N]''')
    while 'y' not in response.lower():
        level = [float(input('Enter new threshold:'))]
        for coll in thres.collections: 
            plt.gca().collections.remove(coll) 
        thres = ax2.contour(smoothed, level, colors='r', origin='lower')
        l = line.pop(0)
        l.remove()
        del l
        line  = ax1.plot(smoothed.columns, [level]*len(smoothed.columns), color='red')
        ax2.clabel(thres, thres.levels, inline=True)
        plt.draw()
        response = input('''Are you happy threshold to remove background noise?
                         Do you want to continue? [Y/N]''')
    plt.ioff()
    plt.close()
    # get bool of all smoothed
    img_as_numpy = img_ave.to_numpy()
    img_as_numpy[(smoothed.to_numpy() < level) | (img_as_numpy < 0)] = 0

    return pd.DataFrame(img_as_numpy, index=img_ave.index, columns=img_ave.columns)

def get_img_width(img_ave):
    '''returns to width of the time averaged image'''
    return img_ave.columns.values.astype(np.float64)

def get_plume_gaussian_model(dat, img_width):
    '''returns a single gaussian fit for the pd series provided
    dat = horizontal row of plume time average df'''

    mod = GaussianModel()
    pars = mod.guess(dat, x=img_width) # guesses starting value for gaussian
    out = mod.fit(dat, pars, x=img_width) # finds best fit of gaussian
    return out

def get_plume_edges(img_ave):
    '''Function will return the plume edges (index) and plume width (in terms of pixels)
     from the gaussian curve'''
    left = []
    right = []
    cent = []
    img_width = get_img_width(img_ave)

    for row in reversed(img_ave.index):
        dat = img_ave.loc[row, :]
        mod = get_plume_gaussian_model(dat, img_width)
        sigma = mod.best_values['sigma']
        center = mod.best_values['center']

        # some filters to deal with bubbles at the top of the image
        # if i == 0:
        #     c.append(round(center))
        #     edges.append( ( round(center - 2**(0.5)*sigma), round(center + 2**(0.5)*sigma) ) )
        # elif ( (center - c[-1]) / center > 0.1 ) or ( (center - 2**(0.5)*sigma - edges[-1][0])
        # / (center - 2**(0.5)*sigma) > 0.2):
        #     c.append(c[-1])
        #     edges.append( edges[-1] )
        # else:
        cent.append(round(center))
        left.append(round(center - 2**(0.5)*sigma))
        right.append(round(center + 2**(0.5)*sigma))

    cent = medfilt(cent, kernel_size=9)
    left = medfilt(left, kernel_size=9)
    right = medfilt(right, kernel_size=9)
    edges = [(l, r) for l, r in zip(left, right)]

    rad = [(center-edge[0], edge[1]-center) for edge, center in zip(edges, cent)]
    return (cent[::-1], rad[::-1])


def plume_edge_linear_regression(img_ave, rad, cent, rel_imgs_dir):
    '''Returns coefficients for a linear regression
    for the plume width and distance from source  '''
    top = 0
    bot = 300
    bot_raw = 500
    # just going to look at the middle of the plume at the moment until data is cleaner
    left_r = [r[0] for r in rad[top:bot]]
    left_r = list(map(lambda r: img_ave.index[int(r)], left_r))
    right_r = [r[1] for r in rad[top:bot]]
    right_r = list(map(lambda r: img_ave.index[int(r)], right_r))
    y_range = img_ave.index[top:bot]

    # plume_width = [r[0]+r[1] for r in rad[top:bot]]
    left = linregress(left_r, y_range)
    right = linregress(right_r, y_range)

    params = pd.DataFrame(np.array([[5/(6*left.slope),5/(6*right.slope)],
                           [left.intercept -1, right.intercept-1]]),
                          index=['alpha','v origin'],
                          columns=['left','right'])
    print(params)        
    fname = f'{rel_imgs_dir}/plume_time_ave/params_from_gauss.pickle'
    with open(fname, 'wb') as pickle_out:
        pickle.dump(params, pickle_out)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot([r[0] for r in rad[top:bot_raw]], img_ave.index[top:bot_raw],
             'kx', markersize=4, label='left_r_raw')
    ax1.plot(left_r, [left.slope*r + left.intercept for r in left_r], 'r', label='left_r')
    ax1.plot([r[1] for r in rad[top:bot_raw]], img_ave.index[top:bot_raw],
             'g>', markersize=4, label='right_r_raw')
    ax1.plot(right_r, [right.slope*r + right.intercept for r in right_r],
             'b', label='right_r')
    ax1.grid(True)
    ax1.set_xlabel('Plume radius (px)')
    ax1.set_ylabel('Plume height (px)')
    ax1.legend()

    ax2.imshow(img_ave, vmin=0, vmax=0.15, aspect='auto')

    ax2.plot(cent[top:bot_raw], np.arange(bot_raw-top), 'r', label='centre')
    ax2.plot([c - r[0] for c, r in zip(cent[top:bot_raw], rad[top:bot_raw])],
             np.arange(bot_raw-top), 'orange', label='left')
    ax2.plot([c + r[1] for c, r in zip(cent[top:bot_raw], rad[top:bot_raw])],
             np.arange(bot_raw-top), 'g', ls='--', label='right')
    ax2.legend()
    fig.suptitle(f'''Experiment: {rel_imgs_dir[7:-9]}
                 alpha_G - left : {params.loc['alpha','left']:0.4f} / right : {params.loc['alpha','right']:0.4f}
                 v origin - left : {params.loc['v origin','left']:0.4f} / right : {params.loc['v origin','right']:0.4f}''')
    plt.savefig(f'{rel_imgs_dir}/plume_time_ave/edge_lin_regress.png', dpi=300)
    plt.close()



def plot_plume_gaussian(img_ave, rel_imgs_dir):
    '''Function takes a list of gaussian fits from the plume and plots'''
    img_width = get_img_width(img_ave)
    for row in img_ave.index[10::50]:
        dat = img_ave.loc[row, :]
        mod = get_plume_gaussian_model(dat, img_width)
        _, (ax1, ax2) = plt.subplots(1, 2)
        mod.plot(xlabel='x (px)', ylabel='A', fig=ax1,
                 data_kws={'markersize':2, 'marker':'o', 'color':'k'})
        ax2.imshow(img_ave)
        plt.savefig(f'{rel_imgs_dir}/plume_time_ave/row_{row}_gauss.png', dpi=300)
        plt.close()






if __name__ == '__main__':

    # this is the range of absorbance to get a good image of the plume
    PLUME_ABSORBANCE_THRES = (0.0, 0.15)
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    DATA_LOC = ['190405']
    # DATA_LOC = ['190328','190328_3' ,'190329','190405','190405_2', '190405_3']

    for data in DATA_LOC:
        REL_IMGS_DIR = './Data/' + data + '/analysis' # File path relative to the script

        pickles = [x for x in  os.listdir(f'{REL_IMGS_DIR}/plume_time_ave')
                   if x.endswith('secs.pickle')]

        for p in pickles:
            try:
                with open(f'{REL_IMGS_DIR}/plume_time_ave/{p}', 'rb') as pickle_in:
                    image_time_ave = pickle.load(pickle_in)
                # Get rid of index for now as I think it will make it easier
                # image_time_ave.reset_index(inplace=True)
                # image_time_ave.drop(columns=['h/H'], inplace=True)

                image_time_ave = remove_img_background_noise(image_time_ave)

                plot_plume_gaussian(image_time_ave, REL_IMGS_DIR)

                centre, radius = get_plume_edges(image_time_ave)

                plume_edge_linear_regression(image_time_ave, radius, centre, REL_IMGS_DIR)


            except FileNotFoundError:
                print('Need to create plume_time_ave first.')
