classdef icBRAACx
    properties
        rawfile = struct('input', '', 'path', '', 'name', '', 'ext', '')
        status = struct('undistorted', false,...
                        'averaged', false,...
                        'black_leveled', false,...
                        'normalised', false,...
                        'cropped', false,...
                        'aligned', false)
        info                    % embeded metadata
        spi                     % Sensor Pixel Intensity
        field                   % Interpolable field function
        bit = 14                % sensor bit depth
        wb = [1, 1, 1]          % [R, G, B]
        crop_range = []         % diagonal: [X1, Y1, X2, Y2] 
        camcali                 % file path to calibration file, or supply as struct
        dyecali                 % file path to calibration file, or supply as object
        fluid = struct('saline_density', [],... % [kg/cm(-3)] 
                       'dye_max', [],...        % [ml/ml]
                       'water_density', 1000)   % [kg/cm(-3)]
    end
    
    methods
        %% Constructor
        function obj = icBRAACx(filename, varargin)
            % process inputs
            n = length(varargin);
            for i = 1:2:n-1
                switch lower(varargin{i})
                    case 'show'
                        show = varargin{i+1};
                    case 'camcali'
                        camcali = varargin{i+1};
                        if isstruct(camcali)
                            obj.camcali = camcali;
                        elseif isfile(camcali)
                            obj.camcali = load(camcali);
                        end
                    case 'dyecali'
                        dyecali = varargin{i+1};
                        if isa(dyecali, 'cDye')
                            obj.dyecali = dyecali;
                        elseif isfile(dyecali)
                            obj.dyecali = load(dyecali);
                            obj.dyecali = obj.dyecali.dye;
                        end
                    case 'dye_max'
                        obj.fluid.dye_max = varargin{i+1};
                    case 'saline_density'
                        obj.fluid.saline_density = varargin{i+1};
                    case 'water_density'
                        obj.fluid.water_density = varargin{i+1};
                    case 'fluid'
                        obj.fluid.saline_density = varargin{i+1}(1);
                        obj.fluid.dye_max = varargin{i+1}(2);
                        obj.fluid.water_density = varargin{i+1}(3);
                    otherwise
                        error([mfilename('class'), ': unknown option.'])
                end
            end
            % process filename
            obj.rawfile.input = filename;
            [obj.rawfile.path, obj.rawfile.name, obj.rawfile.ext] = fileparts(filename);
            if strcmpi(obj.rawfile.path, '')
                obj.rawfile.path = '.';
            end
            % read meta-data
            if ~exist(obj.rawfile.input, 'file')
                error([mfilename('class'), ': file does not exist.'])
            end
            [~, obj.info] = system(['dcraw -i -v ', obj.rawfile.input]);
            obj.info = cfParseMetadata(obj.info);
            % stop this script being used for other formats
            if ~strcmpi(obj.info.Camera, 'Sony ILCE-7RM2')
                error([mfilename('class'), ': only SONY raw files taken with camera model ILCE-7RM2 is supported.'])
            end
            % call dcraw to decode raw file
            system(['dcraw -v -D -4 -t 0 -j -T ', obj.rawfile.input]);
            % read converted file to MATLAB
            file_tiff = [obj.rawfile.path, '/', obj.rawfile.name, '.tiff'];
            a = imread(file_tiff);
            % de-bayer & demosaic by halving the image in each dimension
            obj.spi = a(1:2:end-1, 1:2:end-1);
            obj.spi(:,:,2) = idivide( a(1:2:end-1, 2:2:end) + ...
                                      a(2:2:end, 1:2:end-1), uint16(2), 'round' );
            obj.spi(:,:,3) = a(2:2:end, 2:2:end);
            % show imported image
            if exist('show', 'var')
                if show ==  1
                figure;
                imshow(a);
                end
            end
            % clean up
            clear a
            delete(file_tiff)
        end
        
        %% R/G/B colour space extraction methods
        function red = r(obj)
            red = obj.spi(:,:,1);
        end
        
        function green = g(obj)
            green = obj.spi(:,:,2);
        end
        
        function blue = b(obj)
            blue = obj.spi(:,:,3);
        end
        
        %% Get the hight/width of the image
        function height = H(obj)
            height = size(obj.spi, 1);
        end

        function width = W(obj)
            width = size(obj.spi, 2);
        end
        
        %% Display image
        function fh = show(obj, option)
            if nargin == 1
                option = 'original';
            end
            switch lower(option)
                case 'original'
                    fh = figure;
                    imshow(obj.spi)
                case {'jpeg', 'jpg'} 
                    % display in-camera JPEG
                    % extract JPEG from RAW file
                    system(['dcraw -e ', obj.rawfile.input]);
                    file_jpeg = [obj.rawfile.path, '/', obj.rawfile.name, '.thumb.jpg'];
                    % read converted file to MATLAB
                    a = imread(file_jpeg);
                    fh = figure;
                    imshow(a);
                    delete(file_jpeg)
                case 'peak'
                    disp('Highlights peaked areas, function not implemented.')
                case 'estimatedlengths'
                    [x, y] = obj.estimateAxis;
                    fh = figure;
                    if obj.status.normalised
                        imagesc(x,y,flipud(obj.undistort.spi))
                    else
                        image(x,y,flipud(obj.undistort.spi))
                    end
                    set(gca,'YDir','normal')
                    axis equal tight
                otherwise
                    error([mfilename('class'), ': show: unknown option.'])
            end
        end
        
        %% Display image histogram
        function fh = hist(obj)
            edges = 0:(2^obj.bit); % Left edge is included in each bin, but the right edge is not. The final bin includes both edges. 
            clr = 'rgb';
            fh = figure;
            for i = 1:3
                subplot(3, 1, i)
                histogram(obj.spi(:,:,i), edges, 'edgecolor', fClr(clr(i)), 'facecolor', fClr(clr(i)));
                xlim([0, 2^(obj.bit+0.1)]);
                hold on
                rule(2^obj.bit, 'v', '--', 'color', fClr('m'));
                grid on
                grid minor
                box on
            end
        end
        
        %% Operator overloads
        function obj3 = minus(obj1, obj2)
            obj3 = obj1;
            obj3.spi = obj1.spi - obj2.spi;
        end
        
        function obj3 = mrdivide(obj1, obj2)
            obj3 = obj1;
            obj3.spi = double(obj1.spi) ./ double(obj2.spi);
        end
        
        %% Average pictures (overload)
        function obj2 = mean(obj)
            n = length(obj);
            if n < 2
                obj2 = obj; return
            elseif n >= (2^31 / 2^obj(1).bit)
                error([mfilename('class'), ': mean: too many elements in the object array.'])
            end
            if isfloat(obj(1).spi)
                accumulator = obj(1).spi;
                for i = 2:n
                    accumulator = accumulator + obj(i).spi;
                end
                obj2 = obj(1);
                obj2.spi = accumulator / n;
            else
                accumulator = uint32(obj(1).spi);
                for i = 2:n
                    accumulator = accumulator + uint32(obj(i).spi);
                end
                obj2 = obj(1);
                obj2.spi = idivide( accumulator, uint32(n), 'round' );
                obj2.spi = uint16(obj2.spi);
            end
            
            % housekeeping
            obj2.status.averaged = true;     
            file_list = {};
            for i = 1:n
                if iscell(obj(i).info.Filename)
                    file_list = [file_list; obj(i).info.Filename];
                else
                    file_list = [file_list; {obj(i).info.Filename}];
                end
            end
            obj2.info.Filename = file_list;
        end
        
        %% Background divide
        function obj2 = normalise(obj, obj_bg)
            if obj.status.normalised
                warning([mfilename('class'), ': normalise: will not do this twice.'])
                return
            end
            if ~ (obj.status.black_leveled && obj_bg.status.black_leveled)
                error([mfilename('class'), ': normalise: adjust black-levels first.'])
            end
            if ~isscalar(obj_bg)
                obj_bg = obj_bg.mean;
            end
            if obj.status.undistorted && ~obj_bg.status.undistorted
                warning([mfilename('class'), ': normalise: undistorted intensity reference.'])
                obj_bg = obj_bg.undistort;
            elseif ~obj.status.undistorted && obj_bg.status.undistorted
                warning([mfilename('class'), ': normalise: undistorted original picture.'])
                obj = obj.undistort;
            end
            obj2 = obj / obj_bg;
            % housekeeping
            obj2.status.normalised = true;
            obj2.info.Background_intensity = obj_bg.info;
        end
        
        %% Black-level adjustment
        function obj2 = blackLevel(obj, obj_blk)
            if obj.status.black_leveled
                warning([mfilename('class'), ': blackLevel: will not do this twice.'])
                return
            end
            if ~isscalar(obj_blk)
                obj_blk = obj_blk.mean;
            end
            if obj.status.undistorted && ~obj_blk.status.undistorted
                warning([mfilename('class'), ': blackLevel: undistorted black-level reference.'])
                obj_blk = obj_blk.undistort;
            elseif ~obj.status.undistorted && obj_blk.status.undistorted
                warning([mfilename('class'), ': blackLevel: undistorted original picture.'])
                obj = obj.undistort;
            end
            obj2 = obj - obj_blk;
            % housekeeping
            obj2.status.black_leveled = true;
            obj2.info.Black_level = obj_blk.info;
        end
        
        %% Crop image
        function obj = crop(obj, crop_range)  
            for i = 1:length(obj)
                % crop_range = diagonal: [ROW1, COL1, ROW2, COL2] 
                h1 = crop_range(1);
                h2 = crop_range(3);
                w1 = crop_range(2);
                w2 = crop_range(4);
                % perform crop
                obj(i).spi = obj(i).spi(w1:w2, h1:h2, :);
                % housekeeping
                if obj(i).status.cropped
                    obj(i).crop_range([1,3]) = crop_range([1,3]) + obj(i).crop_range(1) - 1;
                    obj(i).crop_range([2,4]) = crop_range([2,4]) + obj(i).crop_range(2) - 1;
                    % obj.crop_range always refer to the original image
                else
                    obj(i).crop_range = crop_range;                
                end
                obj(i).status.cropped = true;
            end
        end
        
        %% Adjust image colours
        function obj2 = adj(obj, varargin)
            obj2 = obj;
            % process inputs
            n = length(varargin);
            for i = 1:2:n-1
                switch varargin{i}
                    case 'wb'
                        wb = varargin{i+1};
                        if isnumeric(wb) && isvector(wb) && numel(wb) == 3
                            obj2.wb = varargin{i+1};
                        end
                        for j = 1:3
                            obj2.spi(:,:,j) = obj2.spi(:,:,j) * obj2.wb(j);
                        end
                    case 'gamma'
                        gamma = varargin{i+1};
                        obj2.spi = imadjust(obj2.spi,[],[],gamma);
                    otherwise
                        error([mfilename('class'), ': adj: unknown option.'])
                end
            end
        end
        
        %% Find re-alignment corner
        function alignPnt = findAlignPoint(obj, varargin)
            % Defaults
            
            % Assuming the alignment target block is about 200x200 pixels
            % on the lower-RHS of the image.
            
            % It should be a dark rectangular block surrounded by bright
            % pixels; the latter extending all the way to to the right and
            % lower edges of the picture.
            
            % The gaps between the alignment target, and the right / lower 
            % edges of the picture are about 100 pixels wide
            
            % The startPnt identifies a relatively central point inside the
            % alignment target. Boundary tracing is the performed on the
            % right / lower edges of the alignment target.
            
            startPnt = [obj.H - 200, obj.W - 200];
            
            % If no channel is specified, then all channels will be used to
            % produce a grayscale image           
            ch = 0;
            
            % Maximum amount of points to trace
            maxTracePnts = 80;
            
            % Whether to show the figure
            dispFig = 0;
            
            % Input proccessing
            n = length(varargin);
            for i = 1:2:n-1
                switch lower(varargin{i})
                    case 'startpnt'
                        startPnt = varargin{i+1};
                    case {'channel', 'ch'}
                        ch = cfParseChannel(varargin{i+1});
                    case 'maxtracepnts'
                        maxTracePnts = varargin{i+1};
                    case 'dispfig'
                        dispFig = varargin{i+1};
                    otherwise
                        error([mfilename('class'), ': findAlignPoint: unknown option.'])
                end
            end
                      
            % Produce grayscale image
            if ch == 0
                mask = rgb2gray(obj.spi);
            else
                mask = obj.spi(:,:,ch);
            end
            mask = imbinarize(mask, 'adaptive');
            % imshow(mask);
            
            % Starting location for the vertical edge
            rowV = startPnt(1);
            colV = find(mask(rowV, startPnt(2):end)==1, 1, 'first') + (startPnt(2)-1);
            
            % Starting location for the horizontal edge
            colH = startPnt(2);
            rowH = find(mask(startPnt(1):end, colH)==1, 1, 'first') + (startPnt(1)-1);
            
            % Trace the edges
            try
                boundV = bwtraceboundary(mask, [rowV, colV], 'W', 8, maxTracePnts, 'counterclockwise');
                boundH = bwtraceboundary(mask, [rowH, colH], 'N', 8, maxTracePnts, 'clockwise');
            catch
                if obj.status.cropped % the alignment target might have been cropped out
                    warning([mfilename('class'), ': findAlignPoint: image was cropped.'])
                end
                if obj.status.undistorted % this creates additional black boarders
                    warning([mfilename('class'), ': findAlignPoint: image was undistorted.'])
                end     
                error([mfilename('class'), ': findAlignPoint: failed to trace alignment target.'])
            end
            
            % Fit a straight line to the traced edges
            Lv = polyfit(boundV(:,2), boundV(:,1), 1);
            Lh = polyfit(boundH(:,2), boundH(:,1), 1);
            
            % Find the intersection of the two fitted lines
            x = (Lh(2) - Lv(2)) / (Lv(1) - Lh(1));
            y = Lv(1) * x + Lv(2);
            
            x = round(x);
            y = round(y);
            
            alignPnt = [x, y];
            
            fprintf('%s: findAlignPoint: x = %u, y = %u.\n', mfilename('class'), x, y)
            
            if dispFig
                figure
                imshow(mask)
                hold on
                plot(colV, rowV, 'co')
                plot(colH, rowH, 'mo')
                plot(boundV(:,2), boundV(:,1), 'c', 'linewidth', 2)
                plot(boundH(:,2), boundH(:,1), 'm', 'linewidth', 2)
                plot(1:obj.W, polyval(Lv, 1:obj.W),'c--')
                plot(1:obj.W, polyval(Lh, 1:obj.W),'m--')
                plot(x, y, 'g*')
                hold off
            end
        end
        
        %% Re-align an image
        function obj2 = reAlign(obj, alignPnt)
            
            if isa(alignPnt, mfilename('class'))
                alignPnt = alignPnt.findAlignPoint;
                alignPnt = obj.findAlignPoint - alignPnt;
            end
            h = alignPnt(1);
            v = alignPnt(2);
            
            obj2 = obj;
            if sign(h) == -1
                obj2.spi( :, (abs(h) + 1):end, :) = obj2.spi(:, 1:(end-abs(h)), :);
            elseif sign(h) == 1
                obj2.spi(:, 1:(end-abs(h)), :) = obj2.spi( :, (abs(h) + 1):end, :);
            end

            if sign(v) == -1
                obj2.spi( (abs(v) + 1):end, :, :) = obj2.spi( 1:(end-abs(v)), :, :);
            elseif sign(v) == 1
                obj2.spi( 1:(end-abs(v)), :, :) = obj2.spi( (abs(v) + 1):end, :, :);
            end
            
            obj2.status.aligned = true;
        end
        
        %% Find the wall boundary
        function [wall,wall_angle] = findWall(obj, varargin)
            % Defaults
            startPnt = [300, 2062];         
            ch = 0;
            maxTracePnts = 800;
            dispFig = 0;
            output = 'image'; % 'fit', 'image', 'world'
            if obj.status.cropped
                startPnt(1) = startPnt(1) - obj.crop_range(2) + 1;
                startPnt(2) = startPnt(2) - obj.crop_range(1) + 1; % convert to cropped coordinates
            end
            % Input proccessing
            n = length(varargin);
            for i = 1:2:n-1
                switch lower(varargin{i})
                    case 'startpnt'
                        startPnt = varargin{i+1};
                    case {'channel', 'ch'}
                        ch = cfParseChannel(varargin{i+1});
                    case 'maxtracepnts'
                        maxTracePnts = varargin{i+1};
                    case 'dispfig'
                        dispFig = varargin{i+1};
                    case 'output'
                        output = varargin{i+1};
                    otherwise
                        error([mfilename('class'), ': findWall: unknown option.'])
                end
            end     
            % Produce grayscale image
            if ch == 0
                mask = rgb2gray(obj.spi);
            else
                mask = obj.spi(:,:,ch);
            end
            mask = ~imbinarize(mask, 'global');
            % Find a starting point
            row = startPnt(1);
            col = startPnt(2);
            col = col - find(mask(row, startPnt(2):-1:1)==0, 1, 'first') + 2;
            % Trace the receiver edge
            try
                bound = bwtraceboundary(mask, [row, col], 'W', 8, maxTracePnts, 'counterclockwise');
            catch
                error([mfilename('class'), ': findWall: failed to trace alignment target.'])
            end
            % Fit a straight line to the traced edges
            wall = polyfit(bound(:,2), bound(:, 1), 1);
            % Find the angle of the wall
            wall_angle = atand(1/wall(1)); 
            fprintf('%s: findWall: wall angle is %f [deg] to vertical.\n', mfilename('class'), wall_angle)                        
            % If the coordinates are required:
            if any(strcmpi(output, {'image', 'world','estimate'}))
                % compute the intersection of image-borders and wall
                p = [1          0;
                     obj.W      0;
                     0          1;
                     0      obj.H];
                p(1:2, 2) = p(1:2,1) * wall(1) + wall(2);
                p(3:4, 1) = (p(3:4,2) - wall(2)) / wall(1);
                % only retain 2 intersections that are within the bounds of the image
                p = p( (p(:,1)>=1 & p(:,1)<=obj.W) & (p(:,2)>=1 & p(:,2)<=obj.H), :);
                p = unique(p, 'rows'); % corner case
            end            
            % Plot figure
            if dispFig
                figure
                imshow(mask)
                hold on
                plot(col + 1, row, 'ro')
                plot(bound(:,2), bound(:,1), 'r', 'linewidth', 2)
                plot(1:obj.W, polyval(wall, 1:obj.W),'g--')
                hold off
            end
            % Determine output
            switch lower(output)
                case 'fit'
                    return
                case 'image'
                    wall = p;
                case 'world'
                    wall = obj.pixel2mm(p);
                case 'estimate'
                    p(:,1) = p(:,1) * obj.camcali.mmpX;
                    p(:,2) = p(:,2) * obj.camcali.mmpY;
                    wall = p;
                otherwise
                    error([mfilename('class'), ': findWall: invalid output type.'])
            end
        end
        
        %% Undistort image
        function obj2 = undistort(obj)
            if obj.status.cropped
                error([mfilename('class'), ': undistort: cannot do this on a cropped image.'])
            end
            obj2 = obj;
            if obj.status.undistorted
                warning([mfilename('class'), ': undistort: will not do this twice.'])
                return
            end
            [obj2.spi, obj2.camcali.newOrigin] = undistortImage(obj.spi,obj.camcali.cameraParams,'OutputView','same');
            obj2.status.undistorted = true;
        end
        
        %% Convert pixel to world units
        function coords = pixel2mm(obj, imgPnts)            
            obj.require({'undistored'})
            origin = [1, obj.H];
            if obj.status.cropped
%                 disp([mfilename('class'), ': pixel2mm: converted to uncropped imgPnts.'])
                % crop_range = diagonal: [ROW1, COL1, ROW2, COL2]
                imgPnts(:,1) = imgPnts(:,1) + obj.crop_range(2) - 1;
                imgPnts(:,2) = imgPnts(:,2) + obj.crop_range(1) - 1;
                origin = origin + obj.crop_range([2,1]) - 1;
            end

            origin = pointsToWorld(obj.camcali.cameraParams, obj.camcali.R, obj.camcali.t, origin + obj.camcali.newOrigin);
            origin = fliplr(origin);
            
            coords = pointsToWorld(obj.camcali.cameraParams, obj.camcali.R, obj.camcali.t, imgPnts + obj.camcali.newOrigin);
            coords = fliplr(coords);
            
            coords(:,1) = coords(:,1) - origin(1);
            coords(:,2) = coords(:,2) - origin(2);
        end
        
        %% Calculation of dye concentration and density fields
        function obj = prepField(obj, type, ch, interpolant)
            if nargin < 4
                interpolant = true;
            end
            obj.require({'normalised','blackleveled'})
            ch = cfParseChannel(ch);
            switch lower(type)
                case 'dye'
                    s = obj.dyecali.Tf(ch, log(obj.spi(:, :, ch)));
                    dye_max = min([obj.fluid.dye_max, obj.dyecali.maxC]);
                    s( s > dye_max) = dye_max;
                case 'density'
                    s = obj.prepField('dye', ch, 0).field;
                    s = s / obj.fluid.dye_max ...
                        * (obj.fluid.saline_density - obj.fluid.water_density) ...
                        + obj.fluid.water_density;
                case 'dp/p0'
                    s = obj.prepField('density', ch, 0).field;
                    s = ( s - obj.fluid.water_density) / obj.fluid.water_density;
                case 'g`/g0`'
                    s = obj.prepField('density', ch, 0).field;
                    s = ( s - obj.fluid.water_density) / (obj.fluid.saline_density - obj.fluid.water_density);
            end
            % Establish the world coordinates of the pixel grid
            if interpolant
                X = 1:obj.W;
                Y = 1:obj.H;
                [X, Y] = meshgrid(X, Y);
                COORDS = obj.pixel2mm([X(:),Y(:)]);
                obj.field = scatteredInterpolant(COORDS(:,1), COORDS(:,2), s(:));
            else
                obj.field = s;
            end
        end
        
        %% Enquire about field info
        function [s, xq, yq] = intpField(obj, xq, yq)
            % Compute query points and use interpolant
            if nargin < 2 
                % if no query points are specified, do the whole field
                % construct x & y query vectors
                X = [1,obj.W];
                Y = [1,obj.H];
                [X, Y] = meshgrid(X, Y);
                COORDS = obj.pixel2mm([X(:),Y(:)]);
                xq = linspace(0, max(COORDS(:,1)), obj.W)';
                yq = linspace(0, max(COORDS(:,2)), obj.H)';
                % meshgrid vectors
                [X, Y] = meshgrid(xq, yq);
                s = obj.field(X, Y);
            else
                if isscalar(xq)
                    xq = xq + zeros(size(yq));
                end
                if isscalar(yq)
                    yq = yq + zeros(size(xq));
                end
                s = obj.field(xq, yq);
            end
        end
        
        %% Estimate physical x, y axis
        function [x,y] = estimateAxis(obj)
            x = linspace(0, obj.W-1, obj.W)' * obj.camcali.mmpX;
            y = linspace(0, obj.H-1, obj.H)' * obj.camcali.mmpY;
        end
        
        %% Check requirements before performing operation
        function require(obj, reqs)
            pass = true;
            for i = 1:length(reqs)
                switch lower(reqs{i})
                    case 'normalised'
                        pass = pass && obj.status.normalised;
                    case 'blackleveled'
                        pass = pass && obj.status.black_leveled;
                    case 'undistored'
                        pass = pass && obj.status.undistorted;
                    case 'cropped'
                        pass = pass && obj.status.cropped;
                    case '~cropped'
                        pass = pass && (~obj.status.cropped);
                end
            end
            if ~pass
                error([mfilename('class'), ': requirements unmet (', strjoin(reqs,'/'),').'])
            end
        end
    end
end

%% Private
% read metadata
function info = cfParseMetadata(str)
    c = splitlines(str);
    for i = 1:length(c)
        if ~isempty(c{i})
            c2 = strsplit(c{i}, ':\s*', 'DelimiterType', 'RegularExpression');
            c2 = {c2{1}, strjoin(c2(2:end), ':')};
            c2{1} = strrep(c2{1}, ' ', '_');
            info.(c2{1}) = c2{2};
%             eval(['info.',c2{1}, '= ''', c2{2}, ''';'])
        end
    end
end
% translate channel
function ch = cfParseChannel(channel)
    switch lower(channel)
        case 0
            ch = 0;
        case {1, 'r', 'red'}
            ch = 1;
        case {2, 'g', 'green'}
            ch = 2;
        case {3, 'b', 'blue'}
            ch = 3;
        otherwise
            error([mfilename('class'), ': cfParseChannel: invalid channel.'])
    end
end

