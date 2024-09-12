function results = tracker(params)
% 'params'结构体中的字段包括帧间隔、特征设置、全局特征参数、数据类型、搜索区域大小、输出标签函数的标准差因子、滤波器、尺度滤波器

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 设置帧间隔
frame_interval = params.frame_interval;

% Get sequence info
% 从不同的序列格式（'vot'和'otb'）中获取初始信息，包括图像文件、初始位置、初始尺寸
[seq, im] = get_sequence_info(params.seq);
% 从'params'数组中删除'seq'字段
params = rmfield(params, 'seq');
if isempty(im)
    % 'rect_position'存储了目标在每一帧中的位置信息；
    % 是一个矩阵，其大小为 (num_frames, 4), x y坐标和宽度、高度
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init position

pos = seq.init_pos(:)'; % 目标初始位置：x y坐标
target_sz = seq.init_sz(:)'; % 目标初始尺寸：宽度、高度
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
% 全局特征参数的配置与算法的全局配置相互独立，但能继承算法的 GPU 设置
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
% 确保算法能够在 CPU 或 GPU 上正常工作
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end

params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3 % If it's 3, it means there are three color channels.
    if all(all(im(:,:,1) == im(:,:,2))) 
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end
% It extracts only the first channel of the image (im(:,:,1)), effectively converting it to grayscale.
if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    warning('BiCF:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
% 根据所选配置设置搜索窗口的大小和形状，初始化用于跟踪的适当特征
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % for testing
end
    
[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info
% 特征提取器在图像上滑动的窗口大小
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
% 确定要使用的特征块数量,每个特征块对应于一组特征图
num_feature_blocks = length(feature_dim);
% 确定不同特征块的最小单元（cell）大小
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
% 存储每个特征块的特征图尺寸
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
filter_sz = feature_sz + mod(feature_sz+1, 2);
% 'permute'根据给定的 order([2 3 1]) 重新排列数组的维度
% 'mat2cell'将其分割为单元数组，以便每个元素对应一个特征块
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = permute(cell(num_feature_blocks, 1), [2 3 1]);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    % 计算高斯标签函数的标准差
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz))) * params.output_sigma_factor;
    % 计算滤波器大小 sz 的行和列范围
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    % 创建一个网格 rs 和 cs，其中包含了行和列范围的所有可能组合
    [rs, cs]     = ndgrid(rg,cg);
    % 'y'是一个二维高斯函数，表示目标的权重分布
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    % 计算高斯标签函数 y 的二维快速傅立叶变换（FFT）
    yf{1, 1, i}  = fft2(y);
end
% 将 yf 中所有元素转换为与参数 params.data_type 类型相匹配的数据类型
%  cellfun 函数遍历不同的特征块，确保所有标签函数的数据类型一致
yf = cellfun(@(yf) cast(yf, 'like', params.data_type), yf, 'uniformoutput', false);

% construct cosine window
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Define spatial regularization windows
use_sz = filter_sz_cell{1};
small_filter_sz = base_target_sz/feature_cell_sz;

reg_window = construct_regwindow(params, use_sz, small_filter_sz);

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

if params.use_scale_filter
    [nScales, scale_step, scaleFactors, scale_filter, params] = init_scale_filter(params);
else
    % Use the translation filter to estimate the scale.
    nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
end

if nScales > 0
    % force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);

while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    frame_tic = tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));  % 使用无穷大（inf）的值来初始化old_pos目的是在循环的第一次迭代时强制进入循环体
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);  % 当前位置，四舍五入以适应像素网格。
            det_sample_pos = sample_pos;
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
            
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = sum(bsxfun(@times, conj(wf{k1}), xtf{k1}), 3);
            scores_fs_sum = scores_fs_feat{k1};
            
            for k = block_inds
                scores_fs_feat{k} = sum(bsxfun(@times, conj(wf{k}), xtf{k}), 3);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            response = ifft2(scores_fs, 'symmetric');
            % 牛顿法用于优化问题中寻找函数的极值点，trans_row, trans_col为最大响应值的行和列坐标。
            [trans_row, trans_col, scale_ind] = resp_newton(response, scores_fs, newton_iterations, ky, kx, output_sz);
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind);
            scale_change_factor = scaleFactors(scale_ind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % Do scale tracking with the scale filter
            if nScales > 0 && params.use_scale_filter
                scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
            end

            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Extract sample
    % Extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    
    % Do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    clear xlw
    
    % Update appearance model
    if (seq.frame == 1)
        model_xf = xlf;
        model_xf_p = cellfun(@(xlf) zeros(size(xlf), 'single'), xlf, 'UniformOutput', false);
        wf_p = cellfun(@(xlf) zeros(size(xlf), 'single'), xlf, 'UniformOutput', false);
    elseif (seq.frame <= frame_interval)
        model_xf = cellfun(@(model_xf, xlf) (1 - params.learning_rate) * model_xf + params.learning_rate * xlf, model_xf, xlf, 'UniformOutput', false);
    else
        model_xf = cellfun(@(model_xf, xlf) (1 - params.learning_rate) * model_xf + params.learning_rate * xlf, model_xf, xlf, 'UniformOutput', false);
        model_xf_p = model_xf_set{1};
        wf_p = wf_set{1};
    end

    % Do training
    wf = train_TB_BiCF(params, model_xf, yf, reg_window, model_xf_p, wf_p);
    
    % Save xf and wf
    if seq.frame <= frame_interval
        model_xf_set{seq.frame} = model_xf;
        wf_set{seq.frame} = wf;
    else
        model_xf_set(1:frame_interval-1) = model_xf_set(2:frame_interval);
        model_xf_set{frame_interval} = model_xf;
        wf_set(1:frame_interval-1) = wf_set(2:frame_interval);
        wf_set{frame_interval} = wf;
    end
    
    % Update the scale filter
    if nScales > 0 && params.use_scale_filter
        scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
    end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    curr_t = toc(frame_tic);
    seq.time = seq.time + curr_t;
    if params.print_screen == 1
        if seq.frame == 1
            fprintf('initialize: %f sec.\n', curr_t);
            fprintf('===================\n');
        else
            fprintf('[%04d/%04d] time: %f\n', seq.frame, seq.num_frames, curr_t);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % visualization
    if params.visualization
        % 中心坐标-目标尺寸减去1后再除2（减去 1 是为了调整坐标使其更精确地指向像素中心），得左上角坐标，与目标尺寸组成向量。
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
%             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
            
%             output_name = 'Video_name';
%             opengl software;
%             writer = VideoWriter(output_name, 'MPEG-4');
%             writer.FrameRate = 5;
%             open(writer);
        else
            % Do visualization of the sampled confidence scores overlayed
            resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(det_sample_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(det_sample_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            
            % To visualize the continuous scores, sample them 10 times more
            % dense than output_sz.
            sampled_scores_display = fftshift(sample_fs(scores_fs(:,:,scale_ind), 10*output_sz));
            
            figure(fig_handle);
%             set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
            alpha(resp_handle, 0.5);  % 叠加响应图，设置透明度
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            
%             axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        end
        
        drawnow
        %         if frame > 1
        %             if frame < inf
        %                 writeVideo(writer, getframe(gcf));
        %             else
        %                 close(writer);
        %             end
        %         end
        %          pause
    end
end
% close(writer);

[seq, results] = get_sequence_results(seq);

if params.disp_fps
    disp(['fps: ' num2str(results.fps)]);
end
