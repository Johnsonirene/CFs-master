% This function implements the VACF tracker.
function [results] = tracker(params)

num_frames     = params.no_fram;
newton_iterations = params.newton_iterations;
global_feat_params = params.t_global;
featureRatio = params.t_global.cell_size;
search_area = prod(params.wsize * params.search_area_scale);
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);
learning_rate = params.learning_rate;

[currentScaleFactor, base_target_sz, ~, sz, use_sz] = init_size(params,target_sz,search_area);
[y, cos_window] = init_gauss_win(params, base_target_sz, featureRatio, use_sz);
yf          = fft2(y);
[features, im, colorImage] = init_features(params);
[ysf, scale_window, scaleFactors, scale_model_sz, min_scale_factor, max_scale_factor] = init_scale(params,target_sz,sz,base_target_sz,im);
% Pre-computes the grid that is used for score optimization
ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
smallsz = floor(base_target_sz/featureRatio);
time = 0;
loop_frame = 1;
Vy=0;
Vx=0;
% avg_list=zeros(num_frames,1);
% avg_list(1)=0;

for frame = 1:num_frames
    im = load_image(params, frame, colorImage);
    tic();  
    %% main loop

    if frame > 1
        pos_pre = pos;
        [xtf, xcf_c, pos, translation_vec, ~, ~, ~] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame);
        Vy = pos(1) - pos_pre(1);
        Vx = pos(2) - pos_pre(2);
               
        % search for the scale of object
        [xs,currentScaleFactor,recovered_scale]  = search_scale(sf_num,sf_den,im,pos,base_target_sz,currentScaleFactor,scaleFactors,scale_window,scale_model_sz,min_scale_factor,max_scale_factor,params);
    end
    % update the target_sz via currentScaleFactor
    target_sz = round(base_target_sz * currentScaleFactor);
    %save position
    rect_position(loop_frame,:) = [pos([2,1]) - (target_sz([2,1]))/2, target_sz([2,1])];
    
    if frame==1 
        % extract training sample image region
        pixels = get_pixels(im, pos, round(sz*currentScaleFactor), sz);
        context_m = context_mask(pixels,round(target_sz/currentScaleFactor));
        x = get_features(pixels, features, params.t_global);
        ORGIN=x(:,:,1);
        sigma = 0.3; % 高斯滤波器的标准差
        xd = imgaussfilt3(x, sigma);
        later=xd(:,:,1);
        for i=size(x,3)
% 计算特征中心的位置（假设特征中心位于 (25, 25)）
centerX = size(x,1)/2;
centerY = size(2,2)/2;
featureChannel=x(:,:,i);
noisyImage=xd(:,:,i);
% 计算每个像素到特征中心的欧几里德距离
% [rows, cols] = size(noisyImage);
% [X, Y] = meshgrid(1:cols, 1:rows);
% distances = sqrt((X - centerX).^2 + (Y - centerY).^2);
% distanceMap = distances;

% 结合特征和分水岭算法
% combinedImage = imfuse(noisyImage, featureChannel, 'blend', 'Scaling', 'none'); % 使用imfuse函数进行图像融合，并禁用灰度缩放
% alpha = 0.5;  % 自定义权重
% combinedImage = imfuse(noisyImage, featureChannel, 'blend', 'Scaling', 'none', 'BlendAlpha', alpha);
% combinedImage = im2single(combinedImage); % 将图像类型转换为single
alpha = 0.5;  % 自定义权重
combinedImage = alpha * noisyImage + (1 - alpha) * featureChannel;
 x(:,:,i)=combinedImage;

% 使用分水岭算法进行区域分割，并引入距离惩罚项
        end
% sigma = 0.3;
% hsize = [3,3];

% 对每个通道进行高斯滤波
% for i = 1:size(x,3) % D为通道数
%     x(:,:,i) = imgaussfilt(x(:,:,i), sigma, 'FilterSize', hsize);
% end
        ct_m = mexResize(context_m,[size(x,1) size(x,2)],'auto');
        xc = x .* ct_m;
        xf=fft2(bsxfun(@times, x, cos_window));
        xcf_c=fft2(bsxfun(@times, xc, cos_window));
        oldR=zeros(size(xf));
        model_xf = xf;
        xcf = xcf_c;
       [g_f] = run_training(model_xf, xcf, oldR,use_sz, params,yf, smallsz);
        oldR=  xcf_c .* g_f; 
        oldR1=oldR;
   elseif frame==2
        % use detection features
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
         xcf = xcf_c;
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
        [g_f] = run_training(model_xf,xcf, oldR, use_sz, params,yf, smallsz);
        oldR2=oldR1;
         oldR1=  xcf_c .* g_f; 
         oldR=0.32*oldR2+0.68*oldR1;
%          oldR=0.68*oldR2+0.32*oldR1;
%         oldR=0.5*oldR2+ 0.5*oldR1;
      elseif frame==3
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
         xcf = xcf_c;
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
        [g_f] = run_training(model_xf,xcf, oldR, use_sz, params,yf, smallsz);
        oldR3=oldR2;
         oldR2=oldR1;
        oldR1= xcf_c .* g_f; 
%            oldR=0.54*oldR3+0.29*oldR2+0.16*oldR1;
        oldR=0.16*oldR3+0.29*oldR2+0.54*oldR1;
%          oldR=0.33*oldR3+0.33*oldR2+0.33*oldR1;
        elseif frame==4
         shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
         xcf = xcf_c;
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
        [g_f] = run_training(model_xf,xcf, oldR, use_sz, params,yf, smallsz);
        oldR3=oldR2;
        oldR2=oldR1;
        oldR1= xcf_c .* g_f; 
%            oldR=0.54*oldR3+0.29*oldR2+0.16*oldR1;
        oldR=0.16*oldR3+0.29*oldR2+0.54*oldR1;
        else
        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
        xf = shift_sample(xtf, shift_samp_pos, kx', ky');
         xcf = xcf_c;
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
        [g_f] = run_training(model_xf,xcf, oldR, use_sz, params,yf, smallsz);
%         oldR4=oldR3;
        oldR3=oldR2;
         oldR2=oldR1;
        oldR1= xcf_c .* g_f; 
%         oldR=0.25*oldR4+0.25*oldR3+0.25*oldR2+0.25*oldR1;
%         oldR=0.07*oldR3+0.14*oldR3+0.27*oldR2+0.52*oldR1;
%         oldR=0.52*oldR3+0.27*oldR3+0.14*oldR2+0.07*oldR1;
        oldR=0.16*oldR3+0.29*oldR2+0.54*oldR1;
%         elseif frame==5
%         shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
%         xf = shift_sample(xtf, shift_samp_pos, kx', ky');
%          xcf = xcf_c;
%         model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
%         [g_f] = run_training(model_xf,xcf, oldR, use_sz, params,yf, smallsz);
%         oldR5=oldR4;
%         oldR4=oldR3;
%         oldR3=oldR2;
%          oldR2=oldR1;
%         oldR1= xcf_c .* g_f; 
% %         oldR=0.2*oldR5+0.2*oldR4+0.2*oldR3+0.2*oldR2+0.2*oldR1;
% %          oldR=0.03*oldR5+0.06*oldR3+0.125*oldR3+0.25*oldR2+0.5*oldR1;
%          oldR=0.5*oldR5+0.25*oldR3+0.125*oldR3+0.06*oldR2+0.03*oldR1;
%         elseif frame==6
%         shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
%         xf = shift_sample(xtf, shift_samp_pos, kx', ky');
%          xcf = xcf_c;
%         model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
%         [g_f] = run_training(model_xf,xcf, oldR, use_sz, params,yf, smallsz);
%         oldR6=oldR5;
%         oldR5=oldR4;
%         oldR4=oldR3;
%         oldR3=oldR2;
%          oldR2=oldR1;
%         oldR1= xcf_c .* g_f; 
% %         oldR=0.16*oldR6+0.16*oldR5+0.16*oldR4+0.16*oldR3+0.16*oldR2+0.16*oldR1;
% %          oldR=0.0138*oldR6+0.028*oldR5+0.0576*oldR3+0.117*oldR3+0.24*oldR2+0.49*oldR1;
%  oldR=0.49*oldR6+0.24*oldR5+0.117*oldR3+0.0576*oldR3+0.028*oldR2+0.0138*oldR1;
%     else
%        shift_samp_pos = 2 * pi * translation_vec ./(currentScaleFactor * sz);
%         xf = shift_sample(xtf, shift_samp_pos, kx', ky');
%          xcf = xcf_c;
%         model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
%         [g_f] = run_training(model_xf,xcf, oldR, use_sz, params,yf, smallsz);
%         oldR6=oldR5;
%         oldR5=oldR4;
%         oldR4=oldR3;
%         oldR3=oldR2;
%          oldR2=oldR1;
%         oldR1= xcf_c .* g_f; 
% %          oldR=0.0138*oldR6+0.028*oldR5+0.0576*oldR3+0.117*oldR3+0.24*oldR2+0.49*oldR1;
%  oldR=0.49*oldR6+0.24*oldR5+0.117*oldR3+0.0576*oldR3+0.028*oldR2+0.0138*oldR1;
    end 
    
    
    % context residual


    
    %% Update Scale
    if frame==1
%         xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz, 0);
        xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    else
        xs= shift_sample_scale(im, pos, base_target_sz,xs,recovered_scale,currentScaleFactor*scaleFactors,scale_window,scale_model_sz);
    end
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end

    time = time + toc();

     %%   visualization
    if params.visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        figure(1);
        imshow(im);
        if frame == 1
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 26, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
        else
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 28, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            text(12, 66, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
         end
        drawnow
    end
    loop_frame = loop_frame + 1;

%   save resutls.
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
end
%   show speed
disp(['fps: ' num2str(results.fps)])
