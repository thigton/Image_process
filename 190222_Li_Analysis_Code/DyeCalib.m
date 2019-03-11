clear; clc; close all

%% User inputs
caliDate = '20180910';
info = 'Potassium Permanganate, 50g in 2000g of water.';
% info = 'Red food dye, BLENDS LTD, GRN: 103829';
volWater = 20 * 20 * 40; % [ml]
blackLevel = [512, 512, 512];
maxC = 1.4/1000;%10/volWater;

% data file
matfile = ['/home/lm808/LHD/Data2/BRAAC/dyecali/pp_',caliDate,'/results.mat'];

% output file
outfile = 'pp_20180910';

%% Read data
volDye = fMATRead(matfile, 'volDye');
I = fMATRead(matfile, 'I');

%% Save data
dye = cDye(caliDate, info, volWater, blackLevel, volDye, I);
dye.setMaxC(maxC)
% dye.genFits('lin2cubicIntp', [-3 -1 -2])
dye.genFits('poly3y0')
dye.plot('ir');
dye.plot('ln');

% save(['/home/lm808/LHD/Data2/BRAAC/dyecali/mblue_20180806/', outfile], 'dye')