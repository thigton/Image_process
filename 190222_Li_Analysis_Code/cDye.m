classdef cDye < handle
    properties
        caliDate
        info
        volWater
        blackLevel
        volDye
        I
        maxC
        fits = cell(1, 3);
    end
    
    methods
        %% Constructor
        function obj = cDye(caliDate, info, volWater, blackLevel, volDye, I)
            obj.caliDate = caliDate;
            obj.info = info;
            obj.volWater = volWater;
            obj.blackLevel = blackLevel;
            obj.volDye = volDye;
            obj.I = I;
        end
        
        %% Black level adjust
        function I = adjBlack(obj)
            I = obj.I - repmat(obj.blackLevel, size(obj.I,1), 1);
        end
        
        %% Normalise
        function I = normalise(obj)
            I = obj.adjBlack;
            I = I ./ repmat(I(1,:), size(I,1), 1);
        end
        
        %% Natural log values
        function out = ln(obj)
            out = log(obj.normalise);
        end
        
        %% Concentration
        function out = c(obj, unit)
            out = obj.volDye / obj.volWater;
            if ~exist('unit', 'var')
                unit = '';
            end
            switch lower(unit)
                case 'ml/l'
                    out = out * 1000;
            end  
        end
        
        %% I/I0 gradient
        function [grad, xNew] = diff(obj)
            x = obj.c;
            y = obj.normalise;
            xNew = x(2:end);
            grad = (y(2:end,:)-y(1:end-1,:)) ./ diff(x);
        end
        
        %% Recommended starting concentration
        function rc = cp(obj)
            rc = zeros(1,3);
            [grad, xN] = obj.diff;
            gradn = grad ./ repmat(grad(1,:), size(grad,1), 1);
            for i = 1:3
                j = find(gradn(:,i)<=(1/4), 1, 'first');
                if isempty(j)
                    rc(i) = NaN;
                else
                    rc(i) = xN(j);
                end
%                 rc(i) = interp1(gradn(j-1:j+1), xN(j-1:j+1), 1/4);
            end
        end
        
        %% Generate fits for the transfer function
        function genFits(obj, method, linC_ind_adj)
            if nargin < 3
                linC_ind_adj = [0, 0, 0];
            end
            switch lower(method)
                case  'poly3y0'
                    model = fittype( 'a*x^3 + b*x^2 + c*x', 'independent', 'x', 'dependent', 'y' );
                    x = obj.ln;
                    y = obj.c;
                    % fit data
                    for i = 1:3
                        w = zeros(size(y));
                        w(y <= obj.maxC(i)) = 1;
                        [xData, yData, weights] = prepareCurveData( x(:,i), y, w );
                        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
                        opts.Display = 'Off';
                        opts.StartPoint = [0.0987122786555743 0.261871183870716 0.335356839962797];
                        opts.Weights = weights;
                        obj.fits{i} = fit( xData, yData, model, opts );
                    end
                case 'lin2cubicintp'
                    rc = obj.cp; % auto-compute linear region
                    for i = 1:3
                        x = obj.ln;
                        x = x(:,i);
                        y = obj.c;
                        % ensure all data points are numerically unique
                        [~, j] = unique(x, 'stable');
                        x = x(j);
                        y = y(j);
                        % find 'linear' region
                        w = ones(size(y));
                        j = find(y > rc(i), 1, 'first');
                        w(j:end) = 0;
                        % fit 'linear' region
                        [xData, yData, weights] = prepareCurveData( x, y, w);
                        ft = fittype( 'a*x', 'independent', 'x', 'dependent', 'y' );
                        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
                        opts.Display = 'Off';
                        opts.Weights = weights;
                        fit1 = fit( xData, yData, ft, opts );
                        % fit the remainder with a cubic interpolator
                        [xData, yData] = prepareCurveData(x, y);
                        ft = 'splineinterp';
                        opts = fitoptions( 'Method', 'CubicSplineInterpolant' );
                        opts.Normalize = 'on';
                        fit3 = fit( xData, yData, ft, opts );
                        % bridge transition zone with a 2nd-order polynomial
                        % alternative: first point the concentration dips below the linear model (indicates linear model validity)
                        %     j = find(c_fit(end:-1:1) < fit1(ln_fit(end:-1:1)), 1, 'first'); 
                        %     j = length(c_fit) - j + 1;
                        b1 = j-1 + linC_ind_adj(i);
                        b2 = b1 + 2;
                        ln1 = x(b1);
                        ln2 = x(b2);
                        fit2 = polyfit(x(b1:b2),...
                                       [fit1(ln1); y(b1+1:b2-1); fit3(ln2)],...
                                       2);
                                              
                        obj.fits{i} =  struct('method', method, 'linC_ind_adj', linC_ind_adj',...
                                              'fit1', fit1, 'fit2', fit2, 'fit3', fit3, 'ln1', ln1, 'ln2', ln2);
                    end
                otherwise
                    error('cDye: genFits: invalid method.')
            end
                
        end
        
        %% Set maximum allowable concentration
        function setMaxC(obj, maxC)
            obj.maxC = zeros(1,3);
            if isscalar(maxC)
                obj.maxC = obj.maxC + maxC;
            else
                for i = 1:3
                    obj.maxC(i) = maxC(i);
                end
            end
        end
        
        %% Transfer fucntions
        function c = Tf(obj, ch, ln_eval)
            ch = cfParseChannel(ch);       
            [i, j] = size(ln_eval);  
            tf = obj.fits{ch};
            if ~isstruct(tf) % not a composite fit
                c = tf(ln_eval);
                c = reshape(c, i, j);
            else
                % unpack fits
                fit1 = obj.fits{ch}.fit1;
                fit2 = obj.fits{ch}.fit2;
                fit3 = obj.fits{ch}.fit3;
                ln1 = obj.fits{ch}.ln1;
                ln2 = obj.fits{ch}.ln2;
                % calculate concentration
                c = zeros(size(ln_eval));
                c(ln_eval > ln1) = fit1(ln_eval(ln_eval > ln1));
                c(ln_eval < ln2) = fit3(ln_eval(ln_eval < ln2));
                c(ln_eval <= ln1 & ln_eval >= ln2) = polyval(fit2, ln_eval(ln_eval <= ln1 & ln_eval >= ln2));
            end
            c(c > obj.maxC(ch)) = obj.maxC(ch);
            c(c < 0) = 0;
        end
        
        %% Plotter
        function fh = plot(obj, option)
            x = obj.c('ml/L');
            switch lower(option)
                case {'ir', 'i/i_0', 'intensity_ratio'}
                    y = obj.normalise;
                    ylbl = '$ I/I_0 $ [-]';
                    xf = zeros(size(y)) * NaN;
                case {'ln', 'ln(i/i_0)', 'intensity_ratio_ln'}
                    y = obj.ln;
                    ylbl = '$ \ln(I/I_0)$ [-]';
                    xf = zeros(size(y));
                    for i = 1:3
                        if ~isempty(obj.fits{i})
                            tf = obj.fits{i};
                            xf(:,i) = obj.Tf(i, y(:,i)) * 1000;
                            xf(xf(:,i)<0, i) = 0;
                        end
                    end
            end
            rc = obj.cp * 1000;
            clr = [2 4 1];
            fh = figure;
            hold on
            for i = 1:3
                plot(x, y(:,i), '.', 'color', fClr(clr(i)), 'markersize', 10)
            end
            xlm = get(gca, 'XLim');
            ylm = get(gca, 'YLim');
            for i = 1:3 
                if ~isempty(obj.fits{i})    
                    w = xf(:,i)/1000 < obj.maxC(i);
                    plot(xf(w,i), y(w,i), '-', 'color', fClr(clr(i) + 10), 'linewidth', 1.5)
                    plot(xf(~w,i), y(~w,i), ':', 'color', fClr(clr(i) + 10), 'linewidth', 1)
                    rule(obj.maxC(i)*1000, 'v', '-.', 'color', fClr(clr(i)));
    %                 plot(xf(:,i), y(:,i), '-', 'color', fClr(clr(i) + 10), 'linewidth', 1.5)
                end
                    rule(rc(i), 'v', ':', 'color', fClr(clr(i)));
            end
            hold off; box on; grid on; 
            xlim(xlm)
            ylim(ylm)
            xlabel('Dye concentration [ml/L]')
            ylabel(ylbl)
            set(gcf,'OuterPosition',[1776,576,605,544])
        end
        
    end
end

%% Privates
% translate channel
function ch = cfParseChannel(channel)
    switch lower(channel)
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