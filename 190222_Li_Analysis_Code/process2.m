clear; clc; close all

load ./metadata/refimgs.mat bg
bg = bg.undistort;
% Find wall
[wall, th] = bg.crop([1780 210 2120 1610]).findWall('output','world');
wx = wall(:,1);
wy = wall(:,2);

%% Settings and prep
yr = 200;
w = interp1(wy, wx, yr);

files = {'q200','q200b','q400','q600','q800','q800b','q1000'};
%% Analysis
h1 = figure;
hold on
for i = 1:length(files)
    
    load(files{i})

    % settings
    channel = 'green';
    p = p.prepField('density',channel);

    % Figures
    figure
    [s, x, y] = p.intpField;
    [~,h] = contourf(x,y,s,100,'linewidth',0.1);
    caxis([1000 1020])
    colormap jet
    set(h,'LineStyle','none');
    axis equal
    hold on
    plot(wx, wy, 'r')
    rule(yr, 'h', 'r--')
    hold off
    colorbar
    title(files{i})

    figure(h1)
    h2(i) = plot(x-w, p.intpField(x,yr),'color',fClr(i));
end

rule(0,'v','--')
hold off
legend(h2, files)