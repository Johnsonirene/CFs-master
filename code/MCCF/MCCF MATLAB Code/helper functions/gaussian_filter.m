
%---------------------------------------------------
%   Calculating Correlation Output using Gaussian function
%   Multi-Channel Correlation Filters : ICCV'13
%   author    : Hamed Kiani
%   date      : 30 June 2014
%---------------------------------------------------

% Inputs:
%       n   : the size of correlation output e.g. [128 128]
%       s   : sigma, Gaussian bandwidth, e.g. 2
%       pos : the location of the object of interest, e.g. [40 96]



% output
%       rsp :  a 2D Gaussian function with a high peak located at the pos



function rsp = gaussian_filter(n,s, pos)

rsp = zeros(n);
for i=1:n(1)
    for j=1:n(2)
        rsp(i,j) = exp( -((i-pos(1)).^2+(j-pos(2)).^2)/(2*s^2) );
    end;
end;

end

