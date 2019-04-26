import numpy as np
import pandas as pd
import os 
import RAW_img
import matplotlib.pyplot as plt
import csv
import pickle


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
    thres_percent = 0.90 # 85% of the values
    thres_val = 0.004 # have less than 0.5% difference from the rolling mean
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
        ax2.plot(box_roll_diff[t] , box_roll_diff.index.values, label = str(t))
        ax1.plot(box_roll[t], box_roll.index.values, label = str(t))
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
    new_row_names = [ int(float(i.split('d')[-1].replace('_','.'))*10) for i in theory_df.index.values.tolist() ]
    new_col_names = [ int(float(i.split('z')[-1].replace('_','.'))*1000) for i in theory_df.columns.values.tolist() ]
    cols = {}
    rows = {}
    for old_name, new_name in zip(theory_df.columns.values.tolist(), new_col_names):
        cols[old_name] = new_name
    for old_name, new_name in zip(theory_df.index.values.tolist(), new_row_names):
        rows[old_name] = new_name

    theory_df.rename(index = rows, columns = cols, inplace = True)

    return theory_df.apply(lambda x : 1 - x) # flip the scale round to match that the experiment is upside down

def experiment_conditions_as_dict(data_loc):
    '''returns the experiment conditions as a dictionary
    data_loc  - location of data'''
    with open('Data/experiment_details.csv','r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        d = {}
        for row in reader:
            if row[10] == data_loc:
                d['bod'] = int(row[3])
                d['soh'] = int(row[4])
                d['sol_no'] = row[5]
        
        return d


def timeaverage(df , time, time_ss, data_loc):
    '''returns a dataframe with the time averaged denisty profile (mean and std) after the identified steady state
    note std can be calculated by averaging the variance and then sqrt to get std.'''
    box_mean = df.xs(key = ('box', 'mean'), level = [1,2] , axis = 1).loc[:, [x for x in time if x >= time_ss]]
    box_std = df.xs(key = ('box', 'std'), level = [1,2] , axis = 1).loc[:, [x for x in time if x >= time_ss]]
    box_var = box_std.apply(lambda x : x**2)

    box_mean_timeave = pd.Series(box_mean.mean(axis = 1 , skipna = True), name = 'mean')
    box_std_timeave = pd.Series(box_var.mean(axis = 1, skipna = True).apply(lambda x : x**0.5), name = 'std')
           
    return pd.concat([box_mean_timeave, box_std_timeave] , axis =1) 


def plot_timeaverage(df_dict, exp_conditions, door, theory_df):
    '''takes in a dictionary of dataframes of the timeaverage values and saves a plot
    WARNING MAKE SURE THAT THE SAME SOLUTION HAS BEEN USED ON THE EXPERIMENTS YOU WANT TO COMPARE'''
    
    colors = ['blue', 'red','green','orange','purple','yellow'] # list of colors to use on the graphs
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    for (k,v) , c in zip(df_dict.items(), colors[:len(df_dict.keys())]):
        # print(theory_df)
        # print(exp_conditions[k]['bod'])
        # print(exp_conditions[k]['soh'])
        # exit()
        ax1.plot(v['mean'], v['mean'].index.values, label = 'Bot: ' + str(exp_conditions[k]['bod']) 
        + 'mm / Side: ' + str(exp_conditions[k]['soh']) + 'mm' , color = c)
        theory_interface = theory_df.loc[exp_conditions[k]['bod'], exp_conditions[k]['soh']]
        ax1.plot([0,1], [theory_interface, theory_interface], color = c , ls = '--')
        ax1.fill_betweenx(v['std'].index.values, v['mean'] - 2*v['std']  , v['mean'] + 2*v['std'], alpha = 0.2)
    ax1.plot([0,1], [door['front'], door['front']], label = 'door_level', color = 'black')
    ax1.set_ylabel('h/H')
    ax1.set_xlabel('$A$')
    ax1.set_title('Time Averaged Steady State - Experiment comparison \n Theory(--) \n Uncalibrated entrainment, discharge coefficient and virtual origin')
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    plt.legend()
    plt.savefig('./Data/ss_comp/' + str(df_dict.keys()) + '.png')
    plt.close()


if __name__ == '__main__':
    data_loc = ['190328']#,'190405','190405_2', '190405_3']
    exp_conditions = {} # dictionary for all the experiement parameters to compare
    time_average = {} # empty dictionary for the time averaged data to be sit in and be compared
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    # import the thoery data from the .csv created by in matlab
    theory_df = import_theory()

    for exp in data_loc:
        
        rel_data_dir = 'Data/' + exp + '/analysis/'

        try:
            # read in the dataframes
            data = {}
            for df in ['density','interface_height','scales']:
                with open(rel_data_dir + df + '.pickle', 'rb') as pickle_in:
                    data[df] = pickle.load(pickle_in) 
            door_scale = data['scales'][1]
    
        except FileNotFoundError:
            print(f'One of the pickle files can''t has not been generated for this experiment ({exp}) run in excecute.py')

        else:
            # time is a list of the time each image was taken.
            time = sorted( { int(x) for x in data['density']['front'].columns.get_level_values(0) } )
            # append the experiement conditons to the main dictionary
            exp_conditions[exp] = experiment_conditions_as_dict(exp)

#            for t in time:
#                data_for_plot = [ data['front_rho'][t], data['back_rho'][t], data['front_interface'][t], data['back_interface'][t] ]
#                RAW_img.plot_density_compare_scales(rel_data_dir, data_for_plot, door_scale,theory_df, exp_conditions[exp])
#                
            
            time_ss = steadystate(data['density']['front'], time, exp)
            

            # create timeaverage plot
            time_average[exp] = timeaverage(data['density']['front'], time, time_ss, exp) # time averaged density profiles
            
            # Create transient plot
            RAW_img.plot_density_transient(data['density']['front'], door_scale, time, save_loc = rel_data_dir, steadystate = time_ss)
            
    plot_timeaverage(time_average, exp_conditions, door_scale, theory_df)