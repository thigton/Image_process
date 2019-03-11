clear; clc; close all

%% User inputs
caliDate = '20181002';
baseDir = ['D:\Li\camcali\cam_', caliDate, '\'];
sqrSz = 25; % [mm]

%% Read and convert raw files
arw = dir([baseDir, 'geo/arw/*.ARW']);
for i = 1:length(arw)
    system(['dcraw -v -h -w -g 2.222 12.92 -t 0 -j -T ', baseDir, 'geo/arw/', arw(i).name]);
    tiff_name = strrep(arw(i).name, 'ARW', 'tiff');
    movefile([baseDir, 'geo/arw/', tiff_name], [baseDir, 'geo/', tiff_name]);
end

%% Interatively calibrate camera
cameraCalibrator([baseDir, 'geo/'], sqrSz)
drawnow
input('<Press [enter] to continue...>')

%% Compute camera extrinsics
% load background size reference
bg_sizeRef = imread([baseDir, 'geo/', tiff_name]);
imshow(bg_sizeRef)
title('size reference')

% map image points to world points
[bg_sizeRef_ud, newOrigin] = undistortImage(bg_sizeRef,cameraParams,'OutputView','same');
[imgPnts, boardSz] = detectCheckerboardPoints(bg_sizeRef_ud);
wldPnts = generateCheckerboardPoints(boardSz, sqrSz);
[R, t] = extrinsics(imgPnts + newOrigin, wldPnts, cameraParams);

%% Method 1: use pointsToWorld to map 2 axis to real world units
H = size(bg_sizeRef_ud,1);
W = size(bg_sizeRef_ud,2);

x = (1:W)';
y = (1:H)';

xr1 = pointsToWorld(cameraParams, R, t, [x, ones(size(x))] + newOrigin);
xr1_ = pointsToWorld(cameraParams, R, t, [x, H*ones(size(x))] + newOrigin);
xr1 = (xr1 + xr1_)/2; clear xr1_
xr1 = xr1(:,2);

yr1 = pointsToWorld(cameraParams, R, t, [ones(size(y)), y] + newOrigin);
yr1_ = pointsToWorld(cameraParams, R, t, [W*ones(size(y)), y] + newOrigin);
yr1 = (yr1 + yr1_)/2; clear yr1_
yr1 = yr1(:,1);

realPnts = pointsToWorld(cameraParams, R, t, imgPnts + newOrigin);
rp1 = fliplr(realPnts);

% errors
X1 = reshape(rp1(:,1), 5, 8)';
X1d = X1(:, 2:end) - X1(:, 1:end-1);
Y1 = reshape(rp1(:,2), 5, 8)';
Y1d = Y1(2:end, :) - Y1(1:end-1, :);

% figure
% image(xr1,yr1,bg_sizeRef_ud)
% hold on
% plot(rp1(:,1),rp1(:,2),'ro')
% set(gca,'YDir','normal')
% axis equal

fprintf('pointsToWorld errors:\nX - min %fmm(%f%%), mean %fmm(%f%%), max %fmm(%f%%).\nY - min %fmm(%f%%), mean %fmm(%f%%), max %fmm(%f%%)\n\n', ...
         min(abs(sqrSz-X1d(:))), min(abs(1-X1d(:)/sqrSz) * 100),...
         mean(abs(sqrSz-X1d(:))), mean(abs(1-X1d(:)/sqrSz) * 100),...
         max(abs(sqrSz-X1d(:))), max(abs(1-X1d(:)/sqrSz) * 100),...
         min(abs(sqrSz-Y1d(:))), min(abs(1-Y1d(:)/sqrSz) * 100),...
         mean(abs(sqrSz-Y1d(:))), mean(abs(1-Y1d(:)/sqrSz) * 100),...
         max(abs(sqrSz-Y1d(:))), max(abs(1-Y1d(:)/sqrSz) * 100))

%% Method 2: using extent of picture, generate equally-spaced axis
corners = [1,1; W,H; W,1; 1,H];
realCorners = pointsToWorld(cameraParams, R, t, corners + newOrigin);

xmin = realCorners(1,2);
xmax = realCorners(2,2);
ymin = realCorners(2,1);
ymax = realCorners(1,1);
xmin = (xmin + realCorners(4,2))/2;
xmax = (xmax + realCorners(3,2))/2;
ymin = (ymin + realCorners(4,1))/2;
ymax = (ymax + realCorners(3,1))/2;
xr2 = linspace(xmin, xmax, W)';
yr2 = linspace(ymin, ymax, H)';
rp2 = [xr2(round(imgPnts(:,1))), yr2(round(H+1-imgPnts(:,2)))];

% errors
X2 = reshape(rp2(:,1), 5, 8)';
X2d = X2(:, 2:end) - X2(:, 1:end-1);
Y2 = reshape(rp2(:,2), 5, 8)';
Y2d = Y2(2:end, :) - Y2(1:end-1, :);

% average conversions
mmpX = (xmax-xmin)/W;
mmpY = (ymax-ymin)/H;
fprintf('Averaged constant conversion:\nX: %fmm/pixel, Y: %fmm/pixel\n\n', mmpX, mmpY)

figure
image(xr2,yr2,flipud(bg_sizeRef_ud))
hold on
plot(rp2(:,1),rp2(:,2),'ro')
set(gca,'YDir','normal')
axis equal

fprintf('Constant mm/pixel errors:\nX - min %fmm(%f%%), mean %fmm(%f%%), max %fmm(%f%%).\nY - min %fmm(%f%%), mean %fmm(%f%%), max %fmm(%f%%)\n', ...
         min(abs(sqrSz-X2d(:))), min(abs(1-X2d(:)/sqrSz) * 100),...
         mean(abs(sqrSz-X2d(:))), mean(abs(1-X2d(:)/sqrSz) * 100),...
         max(abs(sqrSz-X2d(:))), max(abs(1-X2d(:)/sqrSz) * 100),...
         min(abs(sqrSz-Y2d(:))), min(abs(1-Y2d(:)/sqrSz) * 100),...
         mean(abs(sqrSz-Y2d(:))), mean(abs(1-Y2d(:)/sqrSz) * 100),...
         max(abs(sqrSz-Y2d(:))), max(abs(1-Y2d(:)/sqrSz) * 100))
     
%% Save calibration file
save([baseDir, 'camcalib.mat'], 'caliDate', 'sqrSz', 'cameraParams', 'R', 't', 'mmpX', 'mmpY') % 'bg_sizeRef', 
% save([baseDir, 'camcalib.mat'], 'sqrSz', 'cameraParams', 'newOrigin', 'R', 't', 'bg_sizeRef_ud', 'imgPnts', 'wldPnts')

%% Read in black-level data
arw = dir([baseDir, 'blk/arw/*.ARW']);
for i = 1:length(arw)
    black(i) = icBRAACx([baseDir, 'blk/arw/', arw(i).name], 'camcali', [baseDir, 'camcalib.mat']);
end
black = black.mean;

%% Read in background data
arw = dir([baseDir, 'bkg/arw/*.ARW']);
for i = 1:length(arw)
    bg(i) = icBRAACx([baseDir, 'bkg/arw/', arw(i).name], 'camcali', [baseDir, 'camcalib.mat']);
    alignPnt(i, :) = bg(i).findAlignPoint;
end

alignPnt(:,1) = alignPnt(:,1) - alignPnt(1,1); % x-values
alignPnt(:,2) = alignPnt(:,2) - alignPnt(1,2); % y-values

for i = 2:length(bg)
    bg(i) = bg(i).reAlign(alignPnt(i, :));
end
bg = bg.mean;
bg = bg.blackLevel(black);
bg.show

%% Save background files
save([baseDir, 'refimgs.mat'], 'black', 'bg', 'bg_sizeRef')
