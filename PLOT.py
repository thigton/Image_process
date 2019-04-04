import numpy as np
import pandas as pd
import os 
import RAW_img
import matplotlib.pyplot as plt

def steadystate(df, time, data_loc):
    '''returns the time when the steady state is believed to have been reached
    This is determined by looking at the change in the density profiles between time steps'''

    # only look at the box strip
    box_mean = df.xs(key = ('box', 'mean'), level = [1,2] , axis = 1)
    # rolling mean across time steps
    n = 5
    box_roll = box_mean.rolling(n, axis = 1).mean()
    # find the difference between time steps
    box_roll_diff = box_roll.diff(axis = 1)


    # threshold for steady state
    # plot both to see
    plt.style.use('seaborn-white')
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (12, 9))
    time1 = time[30::10] # only plot every 5 just for the graph
    for t in time1:
        ax2.plot(box_roll_diff[str(t)] , box_roll_diff.index.values, label = str(t))
        ax1.plot(box_roll[str(t)], box_roll.index.values, label = str(t))
    ax1.legend()
    ax2.legend()
    ax2.set_title('difference between rolling averages')
    ax1.set_title(f'rolling average from previous {n} time steps')
    fig.suptitle('box strip - steady state?' )
    plt.savefig(rel_data_dir + '/rho_profile_finding_steady_state.png')
    plt.close()
    
    x = []
    return x





def timeaverage(df , time_ss):
    '''returns a single array for the time averaged denisty profile (mean and std) after the identified steady state
    note std can be calculated by averaging the variance and then sqrt to get std.'''
    pass

def plot_timeaverage(df, exp):
    '''takes in a dataframe of the timeaverage values and saves a plot
    exp - lst of str which represent the data you want to compare.'''
    pass


if __name__ == '__main__':
    data_loc = ['190329','190328_3', '190328']

    for exp in data_loc:
        rel_data_dir = 'Data/' + exp + '/analysis/'
        os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

        try:
            # read in the dataframe
            df = pd.read_csv(rel_data_dir + '/rho.csv', sep = ',' , 
            index_col= [0] , header = [0,1,2] )
        except FileNotFoundError:
            print(f'rho.csv has not been generated for this experiment ({exp}) run in excecute.py')

        else:
            # time is a list of the time each image was taken.
            time = sorted( { int(x) for x in df.columns.get_level_values(0) } )
            time_ss = steadystate(df, time, exp)
            box_dims = RAW_img.read_dict( rel_data_dir[:-9], csv_name = 'box_dims')
            RAW_img.plot_density_transient(df, box_dims, time, steadystate = 500, save_loc = rel_data_dir)
            
            exit()