import numpy as np
import pandas as pd
import os 
import RAW_img
import matplotlib.pyplot as plt
import csv


def steadystate(df, time, data_loc):
    '''returns the time when the steady state is believed to have been reached
    This is determined by looking at the change in the density profiles between time steps'''

    # only look at the box strip
    box_mean = df.xs(key = ('box', 'mean'), level = [1,2] , axis = 1)

    # rolling mean across time steps
    n = 5
    box_roll = box_mean.rolling(n, axis = 1).mean()

    # find the difference between rolling mean and the horizontal average of that time step.
    box_roll_diff = box_mean - box_roll

    # find the first column in the box_roll_diff where it meets the threshold criteria
    thres_percent = 0.85 # 85% of the values
    thres_val = 0.005 # have less than 0.5% difference from the rolling mean
    for col in box_roll_diff.columns[n:]:
        if sum(abs(box_roll_diff[col]) < thres_val)/len(box_roll_diff[col]) > thres_percent:
            time_ss = int(col)
            time_ss_idx = box_roll_diff.columns.get_loc(col)
            break



    # threshold for steady state
    # plot both to see
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (12, 9))
    timex = time[time_ss_idx::5] # only plot every 5 just for the graph
    for t in timex:
        ax2.plot(box_roll_diff[str(t)] , box_roll_diff.index.values, label = str(t))
        ax1.plot(box_roll[str(t)], box_roll.index.values, label = str(t))
    ax1.legend()
    ax2.legend()
    ax2.set_title('diffrence between rolling time average and instance')
    ax1.set_title(f'rolling average from previous {n} time steps')
    fig.suptitle('box strip - steady state?' )
    plt.savefig(rel_data_dir + '/rho_profile_finding_steady_state.png')
    plt.close()
    
    return time_ss

def import_theory():
    '''function inports the theory values for the interface height and tidies up the column headers and index
    to match rest of code'''
    theory_df = pd.read_csv('Data/theory_h.csv', sep = ',', header = [0], index_col = [0] )
    new_col_names = [ str(int(float(i.split('d')[-1].replace('_','.'))*10)) for i in theory_df.columns.values.tolist() ]
    new_row_names = [ str(int(float(i.split('z')[-1].replace('_','.'))*1000)) for i in theory_df.index.values.tolist() ]
    cols = {}
    rows = {}
    for old_name, new_name in zip(theory_df.columns.values.tolist(), new_col_names):
        cols[old_name] = new_name
    for old_name, new_name in zip(theory_df.index.values.tolist(), new_row_names):
        rows[old_name] = new_name

    theory_df.rename(index = rows, columns = cols, inplace = True)
    return theory_df

def experiment_conditions_as_dict(data_loc):
    '''returns the experiment conditions as a dictionary
    data_loc  - location of data'''
    with open('Data/experiment_details.csv','r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        d = {}
        for row in reader:
            if row[10] == data_loc:
                d['bottom_opening_diameter'] = int(row[3])
                d['side_opening_height'] = int(row[4])
                d['sol_no'] = row[5]
        
        return d


def timeaverage(df , time, time_ss, data_loc):
    '''returns a dataframe with the time averaged denisty profile (mean and std) after the identified steady state
    note std can be calculated by averaging the variance and then sqrt to get std.'''
    box_mean = df.xs(key = ('box', 'mean'), level = [1,2] , axis = 1).loc[:, [str(x) for x in time if x >= time_ss]]
    box_std = df.xs(key = ('box', 'std'), level = [1,2] , axis = 1).loc[:, [str(x) for x in time if x >= time_ss]]
    box_var = box_std.apply(lambda x : x**2)

    box_mean_timeave = pd.Series(box_mean.mean(axis = 1 , skipna = True), name = 'mean')
    box_std_timeave = pd.Series(box_var.mean(axis = 1, skipna = True).apply(lambda x : x**0.5), name = 'std')
           
    return pd.concat([box_mean_timeave, box_std_timeave] , axis =1) 


def plot_timeaverage(df_dict, exp_conditions, box_dims):
    '''takes in a dictionary of dataframes of the timeaverage values and saves a plot
    WARNING MAKE SURE THAT THE SAME SOLUTION HAS BEEN USED ON THE EXPERIMENTS YOU WANT TO COMPARE'''

    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    for k,v in df_dict.items():
        ax1.plot(v['mean'], v['mean'].index.values, label = 'Bot: ' + str(exp_conditions[k]['bottom_opening_diameter']) 
        + 'mm / Side: ' + str(exp_conditions[k]['side_opening_height']) + 'mm' )
        ax1.fill_betweenx(v['std'].index.values, v['mean'] - 2*v['std']  , v['mean'] + 2*v['std'], alpha = 0.2)
    ax1.plot([0,1], [box_dims['door'], box_dims['door']], label = 'door_level')
    ax1.set_ylabel('h/H')
    ax1.set_xlabel('$I/I_0$')
    ax1.set_title('Time Averaged Steady State - Experiment comparison')
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    plt.legend()
    plt.savefig('./Data/ss_comp/' + str(df_dict.keys()) + '.png')
    plt.close()


if __name__ == '__main__':
    data_loc = ['190405','190405_2', '190405_3']
    exp_conditions = {} # dictionary for all the experiement parameters to compare
    time_average = {} # empty dictionary for the time averaged data to be sit in and be compared
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    # import the thoery data from the .csv created by in matlab
    theory_df = import_theory()
    
    for exp in data_loc:
        
        rel_data_dir = 'Data/' + exp + '/analysis/'

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
            


            #  get the box_dims
            box_dims = RAW_img.read_dict( rel_data_dir[:-9], csv_name = 'box_dims')
            # append the experiement conditons to the main dictionary
            exp_conditions[exp] = experiment_conditions_as_dict(exp)



            # create timeaverage plot
            time_average[exp] = timeaverage(df, time, time_ss, exp) # time averaged density profiles
            
            # Create transient plot
            RAW_img.plot_density_transient(df, box_dims, time, steadystate = time_ss, save_loc = rel_data_dir)
          
    plot_timeaverage(time_average, exp_conditions, box_dims)