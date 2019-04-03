import numpy as np
import pandas as pd
import os 
import RAW_img

data_loc = '190329'


os.chdir(os.path.dirname(os.path.realpath(__file__))) # change cwd to file location

try:
    # read in the dataframe
    df = pd.read_csv('Data/' + data_loc + '/analysis/rho.csv', sep = ',' , 
    index_col= [0] , header = [0,1,2] )
except FileNotFoundError:
    print('rho.csv has not been generated for this experiment run in excecute.py')

# get the data of ten equally spaced times
time = sorted( { int(x) for x in df.columns.get_level_values(0) } )

box_dims = RAW_img.read_dict('Data/' + data_loc + '/', csv_name = 'box_dims')
RAW_img.plot_density_transient(df, box_dims, time, save_loc = 'Data/' + data_loc + '/analysis/')