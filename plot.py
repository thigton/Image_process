'''Script to analyse and plot multiple experiments on the same graph'''


###########################################################################
#  IMPORT THEORY IS STILL AIMED AT THE CSV FILE CREATED BY MATLAB.
#  NEED TO COMBINE WITH PYTHON CODE
###########################################################################
import os
import sys
import csv
import pickle
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import raw_img
import video_maker
import shutil
from EFB_unbalanced_theory.caseA import caseA
import itertools



def steadystate(dataf, times, thres_percent, thres_val, door):

    '''returns the time when the steady state is believed to have been reached
    This is determined by looking at the change in the density profiles between time steps
    BELOW THE LEVEL OF THE DOOR'''

    # only look at the box strip
    box_mean = dataf.xs(key=('box', 'mean'), level=[1, 2], axis=1)
    # rolling mean across time steps
    no_t_steps_ave = 3
    box_roll = box_mean.rolling(no_t_steps_ave, axis=1).mean()

    # find the difference between rolling mean and the horizontal average of that time step.
    box_roll_diff = (box_mean - box_roll) / box_mean
    if 1 - door > 0.6:
        door = 0.6
    # find the first column in the box_roll_diff where it meets the threshold criteria
    box_roll_diff_mean = box_roll_diff[box_mean.index < (1 -door)].mean(axis=0)
    # exit()
    # t_ss = -10
    box_roll_diff_std = box_roll_diff[box_mean.index < (1 -door)].std(axis=0)
    for col in box_roll_diff.columns[no_t_steps_ave:]:
        if sum(abs(box_roll_diff.loc[box_mean.index < (1 -door), col]) < thres_val)/len(box_roll_diff.loc[box_mean.index < (1 -door), col]) > thres_percent:
            t_ss = int(col)
            time_ss_idx = box_roll_diff.columns.get_loc(col)
            break

    # if t_ss==-10:
    #     # if cant find t_ss. use the last 10% of the experiment
    #     t_ss = int(box_mean.columns[round(box_mean.shape[1]*0.9)])
    #     time_ss_idx = box_roll_diff.columns.get_loc(col)
    #plot how the mean and std changes each timestep
    fig = plt.figure()
    plt.plot(box_roll_diff_mean.index, box_roll_diff_mean, label='rolling mean diff')
    plt.fill_between(box_roll_diff_mean.index, box_roll_diff_mean + 2*box_roll_diff_std,
                     box_roll_diff_mean - 2*box_roll_diff_std, label='conf interval', alpha=0.2)
    plt.plot([0, box_roll_diff_mean.index[-1]], [thres_val, thres_val], color='r', ls='--', label=f'thres val={thres_val*100:0.2f}%')
    plt.plot([0, box_roll_diff_mean.index[-1]], [-thres_val, -thres_val], color='r', ls='--')
    plt.plot([t_ss, t_ss], [-1.0, 1.0], color='purple',ls='--', label=f'steady state time={t_ss}s')
    plt.xlabel('Time')
    plt.ylabel('difference in density profile')
    plt.ylim([-0.5,0.5])
    plt.legend()
    # plt.show()
    plt.savefig(rel_data_dir + '/rho_profile_finding_steady_state2.png', dpi=500)
    plt.close()

    # threshold for steady state
    # plot both to see
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
    timex = times[time_ss_idx::5] # only plot every 5 just for the graph
    for t_inst in timex:
        ax2.plot(box_roll_diff[t_inst], box_roll_diff.index.values, label=str(t_inst))
        ax1.plot(box_roll[t_inst], box_roll.index.values, label=str(t_inst))
    ax1.plot([0, 0.5], [door,door], ls='--', color='r')
    ax1.set_xlabel('Absorbance')
    ax1.set_ylabel('$h/H$')
    ax1.legend()
    ax2.legend()
    ax2.set_title('difference between rolling time average and instance')
    ax1.set_title(f'rolling average from previous {no_t_steps_ave} time steps')
    fig.suptitle('box strip - steady state?')
    # plt.show()
    plt.savefig(rel_data_dir + '/rho_profile_finding_steady_state.png')
    plt.close()

    return t_ss

def import_theory(exp_conditions):
    '''function inports the theory values for the interface height
    and tidies up the column headers and index
    to match rest of code'''
    box_height = 300
    box_width = 300
    door_width = 60
    b0 = door_width / box_width
    z0 = exp_conditions['soh'] / box_height
    params =  [1, 1, 0.021, 0.1351, 0.65]
    obj = caseA(params, b0, 1-z0, 0)
    bot_area = (exp_conditions['bod']**2*np.pi/4) / (box_height**2)
    # get value of a3 which is nearest to the experiment
    # at_idx = np.where(abs(obj.at - bot_area) == np.amin(abs(obj.at - bot_area)))[0][0]
    at_idx = min(range(len(obj.at[obj.at>0])), key=lambda i: abs(obj.at[obj.at>0][i]-bot_area))

    return {'h': 1 - obj.h1[at_idx], 'q1': obj.Q1[at_idx],  'q2': obj.Q2[at_idx],  'q3': obj.Q3[at_idx], 'g': obj.g[at_idx]}

def experiment_conditions_as_dict(data_path):
    '''returns the experiment conditions as a dictionary
    data_path  - location of data'''
    with open('./Data/experiment_details.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        dic = {}
        for row in reader:
            if row[11] == data_path:
                dic['bod'] = int(row[3])
                dic['soh'] = int(row[4])
                dic['sow'] = int(row[5])
                dic['sol_no'] = row[6]
                dic['plume_q'] = int(row[12])
    with open('Data/solution.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0] == dic['sol_no']:
                dic['sol_density'] = float(row[6])
                dic['sol_dye_conc'] = float(row[5]) / (float(row[4]) + float(row[2])) * 1000
        return dic


def timeaverage(data, time, time_ss):
    '''returns a dataframe with the time averaged denisty profile
    (mean and std) after the identified steady state
    note std can be calculated by averaging the variance and then sqrt to get std.'''
    try:
        rho_mean = data['density calibrated']['front'].xs(key='box',level=1,
                                            axis=1).loc[:, [x for x in time if x >= time_ss]]
        rho_timeave_std = pd.Series(name='std')

    except KeyError:
        rho_mean = data['density']['front'].xs(key=('box', 'mean'),
                             level=[1, 2],
                             axis=1).loc[:, [x for x in time if x >= time_ss]]
        rho_std = data['density']['front'].xs(key=('box', 'std'),
                            level=[1, 2],
                            axis=1).loc[:, [x for x in time if x >= time_ss]]

        rho_var = rho_std.apply(lambda x: x**2)
        rho_timeave_std = rho_var.mean(axis=1, skipna=True).apply(lambda x: x**0.5)
        rho_timeave_std.reset_index(drop=True, inplace=True)
        rho_timeave_std.rename('std', inplace=True)


    rho_timeave_mean = rho_mean.mean(axis=1, skipna=True)
    rho_timeave_mean.reset_index(drop=True, inplace=True)
    rho_timeave_mean.rename('mean', inplace=True)



    rho_df = pd.concat([rho_timeave_mean, rho_timeave_std,
                        pd.Series(data['density']['front'].index, name='front scale'),
                        pd.Series(data['density']['back'].index, name='back scale')],
                        axis=1)
    try:
        red_grav = data['reduced gravity']['front'].xs(key='box',level=1,axis=0)#.loc[(time > time_ss), :]
        plot_reduced_gravity_vs_time(red_grav, time_ss)
        red_grav = red_grav[red_grav.index > time_ss].mean()
    except KeyError:
        print('No reduced gravity plot at dye not calibrated')

    h_df = pd.DataFrame(index=['front','back'], columns=['mean','std'])
    for scale in ['front', 'back']:
        h_mean = data['interface_height'][scale].xs(key='grad',
                         level=1,
                         axis=1).loc[:, [x for x in time if x >= time_ss]].mean(axis=0)
        h_std = data['interface_height'][scale].xs(key='grad',
                        level=1,
                        axis=1).loc[:, [x for x in time if x >= time_ss]].std(axis=0)
        h_var = h_std.apply(lambda x: x**2)
        h_df.loc[scale,'mean'] = h_mean.mean(axis=0, skipna=True)
        h_df.loc[scale,'std'] = h_var.mean(axis=0, skipna=True)**0.5
    return (rho_df, h_df, red_grav)

def plot_reduced_gravity_vs_time(red_gravity, time_ss):
    """plots the reduced gravity of the upper layer changing with time

    Arguments:
        red_gravity {[type]} -- [description]
    """
    fig = plt.figure()
    plt.plot(red_gravity)
    red_gravity_time_ss = red_gravity[red_gravity.index > time_ss]
    plt.plot(red_gravity_time_ss, marker='*')
    plt.plot(red_gravity_time_ss.index,[red_gravity_time_ss.mean()]*len(red_gravity_time_ss.index), ls='--')
    plt.xlabel('time [s]')
    plt.ylabel('g\' [ms-2]')

    plt.savefig(f'./Data/{exp}/analysis/red_g_vs_t.png', dpi=250)
    plt.close()

def plot_transient_interface(data, theory_df, exp_cond, door, data_loc):
    '''Creates figure showing the change in mean interface
    with time along with the standard deviation'''
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
    # ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    # ax2 = fig.add_axes([0.1,0.1,0.8,0.8])
    for meth, colors in zip(['grad', 'grad2'], ['blue', 'red']):
        h_mean = data['interface_height']['back'].xs(key=meth, level=1, axis=1).mean(axis=0)
        h_std = data['interface_height']['back'].xs(key=meth, level=1, axis=1).std(axis=0)
        ax1.plot(h_mean.index, h_mean, color=colors, label=meth, lw=2)
        ax1.fill_between(h_mean.index, h_mean - 2*h_std, h_mean + 2*h_std,
                         color=colors,
                         alpha=0.2)
    rho_mean = data['density']['back'].xs(key=['box', 'mean'], level=[1, 2], axis=1)
    sns.heatmap(rho_mean, cmap='coolwarm', ax=ax2, xticklabels=False, yticklabels=False)

    ax1.plot([0, h_mean.index[-1]], [theory_df['h'], theory_df['h']],
             color='black', ls=':', label='theory - SS', lw=2)
    ax1.plot([0, h_mean.index[-1]], [door['back'], door['back']], label='door_level', color='black', lw=2)
    ax1.set_xlabel('Time (s)', fontsize=16)
    ax1.set_ylabel(r"$\frac{{h}}{{H}}$", fontsize=16)
    ax1.set_title('Inplot_density_traterface height with time')
    ax1.set_xlim([h_mean.index[0], h_mean.index[-1]])
    ax1.set_ylim([0, data['density']['back'].index.max()])
    ax1.legend()
    ax2.axis('off')
    ax2.set_title('Horizontally averaged density with time')
    eff_at = ((exp_cond['bod']**2 * 3.14) / 4) / 300**2
    eff_ab = (exp_cond['soh'] * 60) / 300**2
    plt.suptitle(r"$\frac{{a_t}}{{H^2}} = {{{A:0.4f}}} \quad \frac{{a_b}}{{H^2}} = {{{B:0.4f}}} \quad  \frac{{z_b}}{{b}} = {{{C:0.2f}}}$".format(A=eff_at, B=eff_ab, C=exp_cond['soh']/60),
                 fontsize=20)
    plt.savefig(data_loc + '/h_rho_trans.png')
    plt.close()

def compare_timeaverage(all_data, door):
    '''takes in a dictionary of dataframes of the timeaverage values and saves a plot
    WARNING MAKE SURE THAT THE SAME SOLUTION HAS BEEN USED ON THE EXPERIMENTS YOU WANT TO COMPARE'''
     # list of colors to use on the graphs
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plot_width = 3.0
    # for rho in rho_dict.values():
    #     plot_width = rho['mean'].max()*1.25 if rho['mean'].max() > plot_width else plot_width

    for k, color in zip(all_data.keys(), colors):
        rho = all_data[k]['steady state']['density']
        exp_conditions = all_data[k]['exp conditions']
        theory_df = all_data[k]['theory']
        int_h = all_data[k]['steady state']['h']

        #density profile
        ax1.plot(rho['mean'], rho['front scale'], label='Bot: ' +
                 str(exp_conditions['bod']) + 'mm / Side: ' +
                 str(exp_conditions['soh']) + 'mm',
                 color=color)
        ax1.fill_betweenx(rho['std'].index.values,
                          rho['mean'] - 2*rho['std'], rho['mean'] + 2*rho['std'],
                          alpha=0.2, color=color)
        #interface height
        ax1.plot([0, plot_width], [int_h.loc['front','mean'], int_h.loc['front','mean']],
                 color=color, ls='--', label='interface - SS')
        ax1.fill_between([0, plot_width],
                         int_h.loc['front','mean']+2*int_h.loc['front','std'],
                         int_h.loc['front','mean']-2*int_h.loc['front','std'],
                         color=color,
                         alpha=0.2)

        #theory
        ax1.plot([0, plot_width], [theory_df['h'], theory_df['h']],
                 color=color,
                 ls=':',
                 label='theory - SS')

    ax1.plot([0, plot_width], [door['front'], door['front']], label='door_level', color='black')
    ax1.set_ylabel('$h/H$')
    ax1.set_xlabel('$A$')
    ax1.set_title('''Time Averaged Steady State - Experiment comparison \n
                  Theory(--) \n
                  Uncalibrated entrainment, discharge coefficient and virtual origin''')
    ax1.set_xlim([0, plot_width])
    ax1.set_ylim([0, 1])
    plt.legend()
    plt.savefig('./Data/ss_comp/' + str(all_data.keys()) + '.png')
    plt.close()

def video_density_transient(save_loc, exp_codes):
    '''Will create a video of the density profiles of the chosen experiments next to each other'''
    global ALL_DATA
    exp_rho = {}
    exp_h = {}
    for exp in exp_codes:
        exp_rho[exp] = ALL_DATA[exp]['density']['front'].xs(key=('box', 'mean'), level=[1, 2], axis=1)
        exp_h[exp] = ALL_DATA[exp]['interface_height']['front'].xs(key='grad', level=1, axis=1).mean(axis=0)

    max_time = min([max(exp.columns) for exp in exp_rho.values()])
    max_data_points = min([len(exp.columns) for exp in exp_rho.values()])
    try:
        os.mkdir(f'{save_loc}tmp/')
        rho_lines = {}
        h_lines = {}
        for t_idx in range(max_data_points):#, (col1, col2) in enumerate(zip(exp1_rho.columns, exp2_rho.columns)):
            if t_idx == 0:
                fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
                for exp in exp_rho.keys():
                    rho_lines[exp], = ax1.plot(exp_rho[exp].iloc[:, t_idx],exp_rho[exp].index[::-1], label=exp)
                    h_lines[exp], = ax2.plot(exp_h[exp].index[t_idx],1 - exp_h[exp][t_idx], label=exp)


                ax1.set_xlabel('Absorbance')
                ax1.set_xlim([0, 2])
                ax1.set_ylabel('h/H')
                ax1.set_ylim([0, 1])
                ax1.set_title('Uncalibrated density profile')
                ax1.legend()
                # line3, = ax2.plot(exp1_h.index[count],1 - exp1_h[count],'k', label='experiment 1')
                # line4, = ax2.plot(exp2_h.index[count],1 - exp2_h[count],'k--', label='experiment 2')
                ax2.set_xlabel('Time (s)')
                ax2.set_xlim([0, max_time])
                ax2.set_ylabel('h/H')
                ax2.set_ylim([0,1])
                ax2.set_title('Interface Height')
                ax2.legend()
            else:
                for exp in exp_rho.keys():
                     rho_lines[exp].set_data(exp_rho[exp].iloc[:, t_idx], exp_rho[exp].index[::-1])
                     h_lines[exp].set_data(exp_h[exp].index[0:t_idx], 1 - exp_h[exp][0:t_idx])
                # line1.set_data(exp1_rho[col1], exp1_rho.index[::-1])
                # line2.set_data(exp2_rho[col2], exp2_rho.index[::-1])

                # line3.set_data(exp1_h.index[0:count], 1 - exp1_h[0:count])
                # line4.set_data(exp2_h.index[0:count], 1 - exp2_h[0:count])

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.savefig(f'{save_loc}tmp/fig_{t_idx:03d}.png', dpi=500)
            print(f'save fig {t_idx}')
    except FileExistsError:
        print('''tmp folder already exists and video will be made from its contents,
              if this is incorrect delete the folder ''')
    video_maker.convert_frames_to_video(f'{save_loc}tmp/', f'{save_loc}{exp_codes}.mp4', 5.0)
    shutil.rmtree(f'{save_loc}tmp/')
    plt.close()

def calibrate_density_df(DATA):
    """Function will:
    1. Check if there are calibration coefficients for the solution number
    2. If there is, density dataframe to convert aborbance to a density
    3. calculate a mean and standard deviation of the portion above the transistion potion of the dyed layer.

    Arguments:
        DATA {dict} -- This is all the data for the experiment.
    """

    try:
        with open("./Data/dye_cali_coeff.pickle", "rb") as pickle_in:
            all_coeffs = pickle.load(pickle_in)
    except FileNotFoundError:
        print("No coefficients file!")
        sys.exit()
    # 1
    if DATA['exp conditions']['sol_no'] in all_coeffs:
        all_relevant_coeffs = all_coeffs[DATA['exp conditions']['sol_no']]
        coeffs = np.zeros(3)
        # finds the mean of all the coefficients created.
        for i, coeff in enumerate(all_relevant_coeffs.values()):
            coeffs = np.add(coeffs, coeff['red'])
            coeffs = np.divide(coeffs, i+1)

        # 2
        def absorbance_to_density(x, coeff, c_sol, rho_sol):
            x /= 30 # A/b
            c = np.polyval(coeff, x)
            return c / c_sol*(rho_sol-0.9996) + 0.9996
        print("Densities being calibrated!")
        c_sol = DATA['exp conditions']['sol_dye_conc']
        rho_sol = DATA['exp conditions']['sol_density']
        density_calibrated = {}
        reduced_gravity = {}
        for scale, density_df in DATA['density'].items():
            density_calibrated[scale] = density_df.xs('mean',axis=1, level=2).applymap(lambda x: absorbance_to_density(x, coeffs, c_sol, rho_sol))
            # 3
            # lets find the 2nd differential again from scratch
            # from the densities as not all the information is in grad2 interface height
            def get_reduced_gravity(x):
                """use with df.apply down the columns of the density_calibrated
                just made
                Arguments:
                    x {pd.series} -- array to find the top of the interface zone.
                """
                x = x[x.index > 0.1]
                rolling_mean = x.rolling(50, center=True).mean()
                rolling_mean.fillna(method='ffill', inplace=True)
                rolling_mean.fillna(method='bfill', inplace=True)
                first_diff = pd.Series(np.gradient(rolling_mean), index=x.index)
                first_diff = first_diff.rolling(50, center=True).mean()
                first_diff.fillna(method='ffill', inplace=True)
                first_diff.fillna(method='bfill', inplace=True)
                top_of_interface_zone = np.argmin(np.gradient(first_diff))
                mean_rho = x.iloc[top_of_interface_zone:].mean()
                return (mean_rho - 0.9996) / 0.9996 * 9.81

            reduced_gravity[scale] = density_calibrated[scale].apply(get_reduced_gravity, axis=0)

        with open(f'./Data/{exp}/analysis/density_calibrated.pickle','wb') as pickle_out:
                pickle.dump(density_calibrated, pickle_out)
        with open(f'./Data/{exp}/analysis/reduced_gravity.pickle','wb') as pickle_out:
                pickle.dump(reduced_gravity, pickle_out)
        return (density_calibrated, reduced_gravity)
    else:
        print("Experiment solution has not been calibrated")
        return (None, None)


#####################################################################

if __name__ == '__main__':
    # DATA_LOC = ['190712',
    #             '190717','190717_2','190717_3', '190717_4',
    #             '190718','190718_2','190718_3', '190718_4','190718_5','190718_6',
    #             '190719','190719_2','190719_3',
    #             '190722', '190722_2','190722_3','190722_4','190722_5','190722_6',
    #             '190724','190724_2','190724_3','190724_4','190724_5','190724_6','190724_7',
    #             '190725', '190725_2', '190725_3', '190725_4',
    #             '190726', '190726_2', '190726_3', '190726_4',
    #             '190727', '190727_2', '190727_3', '190727_4','190727_5', '190727_6',
    #             '190729', '190729_2', '190729_3', '190729_4', '190729_5', '190729_6', '190729_7',
    #             '190730', '190730_2', '190730_3', '190730_4', '190730_5', '190730_6',
    #             '190926', '190926_2', '190926_3', '190926_4', '190926_5',
    #             '190927', '190927_2', '190927_3', '190927_4',
    #             '191003', '191003_2', '191003_3',
    #             '191007', '191007_3','191007_4', '191007_5',
                # '191009', '191009_2', '191009_3','191009_4',
                #   '200127', '200128', '200128_2',
                #   '200129', '200129_2', '200129_3','200129_4','200129_5',
                #   '200130', '200130_2', '200130_3','200130_4','200130_5','200130_6',
                #   '200131', '200131_2', '200131_3','200131_4',
                #   '200206', '200206_2', '200206_3', '200206_4']
    DATA_LOC = [
                '200311_2',
                ]
    os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location
    thres_percent = 0.9 # 90% of the values
    thres_val = 0.0015
     # have less than 0.5% difference from the rolling mean
    # OPTIONS
    PLOT_COMPARE_SCALES = 0
    SAVE_EXP_DATA_FOR_THEORY = 1
    COMPARE_TIMEAVERAGE = 0
    VIDEO = 0
    file_ext = '.ARW'

    ALL_DATA = {}
    for exp in DATA_LOC:
        print(f'exp:{exp}')
        rel_data_dir = 'Data/' + exp + '/analysis/'
        try:
            # read in the dataframes
            DATA = {}
            for df in ['density', 'interface_height', 'scales']:
                with open(f'{rel_data_dir}{file_ext[1:]}_{df}.pickle', 'rb') as pickle_in:
                    DATA[df] = pickle.load(pickle_in)
            door_scale = DATA['scales'][1]
        except FileNotFoundError:
            print(f'One of the pickle files can''t has not been')
            print(f'generated for this experiment ({exp}) run in excecute.py')

        else:
            # time is a list of the time each image was taken.
            TIME = sorted({int(x) for x in DATA['density']['front'].columns.get_level_values(0)})

            # append the experiement conditons to the main dictionary
            DATA['exp conditions'] = experiment_conditions_as_dict(exp)
            DATA['theory'] = import_theory(DATA['exp conditions'])



            if PLOT_COMPARE_SCALES == 1:
                for count, t in enumerate(TIME):
                    interface_to_plot = 'grad'
                    data_for_plot = [DATA[param][scale][t]
                                     for param, scale in
                                     itertools.product(['density', 'interface_height'],
                                                       ['front', 'back'])]
                    raw_img.plot_density_compare_scales(rel_data_dir, data_for_plot,
                                                        t, door_scale, DATA['theory'],
                                                        DATA['exp conditions'], interface_to_plot)
                    print(f'{count+1} of {len(TIME)} graphs plotted to compare the scale')

        DATA['density calibrated'], DATA['reduced gravity'] = calibrate_density_df(DATA)

        DATA['steady state'] = {}
        try:
            DATA['steady state']['time'] = steadystate(DATA['density']['front'],
                                                       TIME, thres_percent,
                                                       thres_val, DATA['exp conditions']['soh']/300)
        except UnboundLocalError:
            print('''No Steady State found,  consider relaxing the required thresholds,
             time average plot will not be made.''')
        else:
            # create timeaverage plot
            DATA['steady state']['density'], DATA['steady state']['h'], DATA['steady state']['reduced gravity'] = timeaverage(DATA, TIME, DATA['steady state']['time'])
            # Create transient plot of individual profiles
            raw_img.plot_density_transient(DATA['density']['front'], door_scale,
                                           TIME, save_loc=rel_data_dir, steadystate=DATA['steady state']['time'])

        #transient plot of interfaces
        plot_transient_interface(DATA, DATA['theory'], DATA['exp conditions'], door_scale, rel_data_dir)

        ALL_DATA[exp] = DATA

    # save ss values to a pickle to be used on theory plots
    if SAVE_EXP_DATA_FOR_THEORY == 1:
        try:
            with open('./EFB_unbalanced_theory/exp_data.pickle','rb') as pickle_in:
                ss_dict = pickle.load(pickle_in)
        except FileNotFoundError:
            ss_dict = {}
        except EOFError:
            ss_dict = {}

        for k in ALL_DATA.keys():
            ss_dict[k] = ALL_DATA[k]['steady state']
            ss_dict[k]['conditions'] = ALL_DATA[k]['exp conditions']
        with open('./EFB_unbalanced_theory/exp_data.pickle','wb') as pickle_out:
            pickle.dump(ss_dict, pickle_out)
            print('creating exp_data.pickle')

    if COMPARE_TIMEAVERAGE == 1:
        compare_timeaverage(ALL_DATA, door_scale)
    if VIDEO == 1:
        video_density_transient('./Data/videos/', DATA_LOC)

