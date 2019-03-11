clear; clc; close all

load ./metadata/refimgs.mat
camcali = './metadata/camcalib.mat';
dyecali = './metadata/red_20190802.mat';

data_dir = './Q1000/';
dye_max = 2.163966/1000;
rho_s = 1101.67;
rho_w = 1000;

files = dir([data_dir,'*.ARW']);
nf = length(files);

% Initial processing
for i = 1:nf
    p(i) = icBRAACx([files(i).folder,'/',files(i).name], 'camcali', camcali, 'dyecali', dyecali, ...
                     'dye_max', dye_max, 'saline_density', rho_s, 'water_density', rho_w);
    p(i) = p(i).reAlign(bg);
end
black = black.undistort;
bg = bg.undistort;
for i = 1:nf
    p(i) = p(i).undistort;
    p(i) = p(i).blackLevel(black);
    p(i) = p(i).normalise(bg);
end

crop_range = [1780 210 2120 1610];
p = p.crop(crop_range).mean;
save(lower(strrep(strrep(data_dir,'.',''),'/','')), 'p','-v7.3')