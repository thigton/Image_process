import numpy as np
import pandas as pd
import os 
import RAW_img
import matplotlib.pyplot as plt
import csv
import pickle
import itertools
import seaborn as sns

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
    thres_percent = 0.9 # 90% of the values
    thres_val = 0.1 # have less than 0.5% difference from the rolling mean
    for col in box_roll_diff.columns[n:]:
        
        if sum(abs(box_roll_diff[col]) < thres_val)/len(box_roll_diff[col]) > thres_percent:
            time_ss = int(col)
            time_ss_idx = box_roll_diff.columns.get_loc(col)
            break
    


    # threshold for steady state
    # plot both to see
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (12, 9), sharey = True)
    timex = time[time_ss_idx::5] # only plot every 5 just for the graph
    for t in timex:
        ax2.plot(box_roll_diff[t] , box_roll_diff.index.values, label = str(t))
        ax1.plot(box_roll[t], box_roll.index.values, label = str(t))
    ax1.set_xlabel('Absorbance')
    ax1.set_ylabel('$h/H$')
    ax1.legend()
    ax2.legend()
    ax2.set_title('difference between rolling time average and instance')
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


def timeaverage(data , time, time_ss, data_loc):
    '''returns a dataframe with the time averaged denisty profile (mean and std) after the identified steady state
    note std can be calculated by averaging the variance and then sqrt to get std.'''
  
    df_rho = data['density']['front'] 
    df_h = data['interface_height']['front']
    # print(df_rho.head())
    # exit()
    rho_mean = df_rho.xs(key = ('box', 'mean'), level = [1,2] , axis = 1).loc[:, [x for x in time if x >= time_ss]]
    rho_std = df_rho.xs(key = ('box', 'std'), level = [1,2] , axis = 1).loc[:, [x for x in time if x >= time_ss]]
    rho_var = rho_std.apply(lambda x : x**2)

    rho_mean_timeave = pd.Series(rho_mean.mean(axis = 1 , skipna = True), name = 'mean')
    rho_std_timeave = pd.Series(rho_var.mean(axis = 1, skipna = True).apply(lambda x : x**0.5), name = 'std')

    h_mean = df_h.xs(key = 'grad', level = 1 , axis = 1).loc[:, [x for x in time if x >= time_ss]].mean(axis = 0)
    h_std = df_h.xs(key = 'grad', level = 1 , axis = 1).loc[:, [x for x in time if x >= time_ss]].std(axis = 0)
    h_var = h_std.apply(lambda x : x**2)

    h_mean_timeave = h_mean.mean(axis = 0 , skipna = True)

    h_std_timeave = h_var.mean(axis = 0, skipna = True)**0.5

    return (pd.concat([rho_mean_timeave, rho_std_timeave] , axis =1) , [h_mean_timeave, h_std_timeave] ) 

def plot_transient_interface(data, theory_df, exp_cond, door, data_loc):
    '''Creates figure showing the change in mean interface with time along with the standard deviation'''
    _, (ax1,ax2) = plt.subplots(1,2,figsize=(12,9))
    # ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    # ax2 = fig.add_axes([0.1,0.1,0.8,0.8])
    for meth, c in zip(['grad'], ['blue','red']):
        h_mean = data['interface_height']['front'].xs(key = meth, level = 1 , axis = 1).mean(axis = 0 )
        h_std = data['interface_height']['front'].xs(key = meth, level = 1 , axis = 1).std(axis = 0)
        ax1.plot(h_mean.index,h_mean, color = c, label  = 'Bot: ' + str(exp_cond['bod']) 
        + 'mm / Side: ' + str(exp_cond['soh']) + 'mm' , lw = 2 )
        ax1.fill_between(h_mean.index, h_mean - 2*h_std, h_mean + 2*h_std, color = c, alpha = 0.2)
        rho_mean = data['density']['front'].xs(key = ['box','mean'], level = [1,2], axis = 1)
        # image = ax1.imshow(rho_mean.T, aspect = 'auto', cmap = 'inferno', vmin = 0 , vmax = 1.5)
        # plt.colorbar(image,ax = ax1, orientation = 'vertical')
        # print(rho_mean.T.tail())
        # exit()
        sns.heatmap(rho_mean, cmap = 'coolwarm', ax = ax2, xticklabels = False,yticklabels = False,)
    theory_interface = theory_df.loc[exp_cond['bod'], exp_cond['soh']]
    ax1.plot([0,h_mean.index[-1]], [theory_interface, theory_interface], color = 'red' , ls = ':', label = 'theory - SS')
    ax1.plot([0,h_mean.index[-1]], [door['front'], door['front']], label = 'door_level', color = 'red')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('$h/H$')
    ax1.set_title('Interface height with time')
    ax1.set_xlim([h_mean.index[0], h_mean.index[-1]])
    ax1.set_ylim([0, data['density']['front'].index.max()])
    ax1.legend()
    ax2.axis('off')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Horizontally averaged density with time')
    plt.savefig(data_loc + '/h_rho_trans.png')
    plt.close()

        
        
def compare_timeaverage(rho_dict, h_dict,  exp_conditions, door, theory_df):
    '''takes in a dictionary of dataframes of the timeaverage values and saves a plot
    WARNING MAKE SURE THAT THE SAME SOLUTION HAS BEEN USED ON THE EXPERIMENTS YOU WANT TO COMPARE'''
    colors = ['blue', 'red','green','orange','purple','yellow'] # list of colors to use on the graphs
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    plot_width = 1.0
    for v in rho_dict.values():
        plot_width = v['mean'].max()*1.25 if v['mean'].max() > plot_width else plot_width
        print(plot_width)
        exit()
    for (k,v) , h, c in zip(rho_dict.items(), h_dict.values() ,colors[:len(rho_dict.keys())]):
        #density profile
        ax1.plot(v['mean'], v['mean'].index.values, label = 'Bot: ' + str(exp_conditions[k]['bod']) 
        + 'mm / Side: ' + str(exp_conditions[k]['soh']) + 'mm' , color = c)
        ax1.fill_betweenx(v['std'].index.values, v['mean'] - 2*v['std']  , v['mean'] + 2*v['std'], alpha = 0.2, color = c)
        #interface height
        ax1.plot([0,plot_width],[h[0], h[0]], color = c , ls = '--', label = 'interface - SS')
        ax1.fill_between([0,plot_width], h[0]+2*h[1], h[0]-2*h[1], color = c, alpha = 0.2)
        
        #theory
        theory_interface = theory_df.loc[exp_conditions[k]['bod'], exp_conditions[k]['soh']]
        ax1.plot([0,plot_width], [theory_interface, theory_interface], color = c , ls = ':', label = 'theory - SS')
 
    ax1.plot([0,plot_width], [door['front'], door['front']], label = 'door_level', color = 'black')
    ax1.set_ylabel('$h/H$')
    ax1.set_xlabel('$A$')
    ax1.set_title('Time Averaged Steady State - Experiment comparison \n Theory(--) \n Uncalibrated entrainment, discharge coefficient and virtual origin')
    ax1.set_xlim([0,plot_width])
    ax1.set_ylim([0,1])
    plt.legend()
    plt.savefig('./Data/ss_comp/' + str(rho_dict.keys()) + '.png')
    plt.close()

#####################################################################

if __name__ == '__main__':
    data_loc = ['190405']#,'190405_2', '190405_3']
    exp_conditions = {} # dictionary for all the experiement parameters to compare
    rho_time_ave = {} # empty dictionary for the time averaged data to be sit in and be compared
    h_time_ave = {}
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

    # OPTIONS
    PLOT_COMPARE_SCALES = 0


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


            if PLOT_COMPARE_SCALES == 1:
                for count, t in enumerate(time):
                    interface_to_plot = 'grad'
                    data_for_plot = [data[param][scale][t] for param, scale in itertools.product(['density','interface_height'], ['front','back']) ]

                    RAW_img.plot_density_compare_scales(rel_data_dir, data_for_plot,t, door_scale,theory_df, 
                    exp_conditions[exp], interface_to_plot)     
                    print(f'{count+1} of {len(time)} graphs plotted to compare the scale')
            
        try:    
            time_ss = steadystate(data['density']['front'], time, exp)
        except UnboundLocalError:
            ('No Steady State found,  consider relaxing the required thresholds, time average plot will not be made.')
        else:
            # create timeaverage plot
            rho_time_ave[exp] , h_time_ave[exp]  = timeaverage(data, time, time_ss, exp) # time averaged density profiles
            
            # Create transient plot of individual profiles
            RAW_img.plot_density_transient(data['density']['front'], door_scale, time, save_loc = rel_data_dir, steadystate = time_ss)
            
        #transient plot of interfaces
        plot_transient_interface(data,theory_df, exp_conditions[exp],door_scale, rel_data_dir )

    compare_timeaverage(rho_time_ave, h_time_ave, exp_conditions, door_scale, theory_df)