

%---------------------------------------------------
%   Calculating cosine window
%   Multi-Channel Correlation Filters : ICCV'13
%   author    : Hamed Kiani
%   date      : 30 June 2014
%---------------------------------------------------

%   step : 4
%   dimW : window size

function [w] = get_cosine_window(dimW, step)

w1 = cos(linspace(-pi/step, pi/step, dimW(1)));
w2 = cos(linspace(-pi/step, pi/step, dimW(2)));
w = w1' * w2;

end
