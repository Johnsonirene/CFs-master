clear all; close all; clc;
addpath('helper functions/');
imgsPath = 'multiPie Face Dataset/';
imgs     = dir(fullfile(imgsPath, '*.jpg'));

im_sz      = [128 128];
target_pos = [40 96];

%   HoG parameters. Pleae refer to "calc_hog" function for more details.
nbins      = 5;
cell_size  = [5 5];
block_size = [5 5];

%   MCCF Gaussian sigma and lambda. Please refer to the reference paper for
%   more details.
sigma  = 2;
lambda = 0.1;

cos_window = get_cosine_window(im_sz,2);

im       = imread([imgsPath imgs(3).name]);
% imshow(im)
if size(im,3) == 3
    im = double(rgb2gray(im));
end;
% im_ = rgb2gray(im);
% imshow(im)
nor_im = powerNormalise(double(im));
imshow(nor_im)

corr_rsp = gaussian_filter(size(im),sigma, target_pos);
corr_rsp_f = reshape(fft2(corr_rsp), [],1);
% imshow(corr_rsp)

hogs = calc_hog(nor_im, nbins, cell_size, block_size);
hogs = bsxfun(@times, hogs, cos_window);
hogs_f = fft2(hogs);
diag_hogs_f = spdiags(reshape(hogs_f, prod(im_sz), []), ...
        [0:nbins-1]* prod(im_sz), prod(im_sz), prod(im_sz)*(nbins));

xxF = diag_hogs_f'*diag_hogs_f  ;
xyF = diag_hogs_f'*corr_rsp_f ;

I     = speye(size(xxF,1)); %取xxF的行
filtF = (xxF + I*lambda)\xyF; %直接求逆比左除耗时，某些情况下两者可以达到相同的结果
filtF = (reshape(filtF, im_sz(1), im_sz(2), []));
filt  = real(ifft2(filtF));
filt  = circshift(filt, floor(im_sz/2));

for j = 1: nbins
    subplot(2,3,j) ;imagesc(real(filt(:,:,j)));colormap gray;
    axis off; axis image; title(['MCCF Channel # : ' num2str(j)]);
end;

save ('filt','filt');
