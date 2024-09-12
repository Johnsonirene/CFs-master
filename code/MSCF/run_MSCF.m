%   This function runs the MSCF tracker on the video specified in 
%   "configSeqs".
%   This function is based on ARCF paper.
%   Details of some parameters are not presented in the paper, you can
%   refer to BACF/DSST/ECO/ARCF paper for more details.

function results = run_MSCF(seq, rp, bSaveImage)
% Feature specific parameters
hog_params.cell_size = 4;
hog_params.compressed_dim = 10;
hog_params.nDim = 31;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.nDim = 10;

% Which features to include
params.t_features = {
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...    
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
};

% Global feature parameters1s
params.t_global.cell_size = 4;                  % Feature cell size

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 5;           % The scaling of the target size to get the search area
params.image_sample_size = 200^2;

% Spatial regularization window_parameters
params.feature_downsample_ratio = 4; %  Feature downsample ratio
params.reg_window_max = 1e5;           % The maximum value of the regularization window
params.reg_window_min = 1e-3;           % the minimum value of the regularization window

%Dynamic hybrid label parameters
params.delta = 0.01;
params.nu    = 1;
params.eta   = 0.044;
params.sz_ratio = 2.5;


% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = 0.066;		% Label function sigma
params.temporal_regularization_factor = [15 15]; % The temporal regularization parameters

% ADMM parameters
params.max_iterations = [2 2];
params.init_penalty_factor = [1 1];
params.max_penalty_factor = [0.1, 0.1];
params.penalty_scale_step = [10, 10];

params.num_scales = 33;
params.hog_scale_cell_size = 4;
params.learning_rate_scale = 0.024;
params.scale_sigma_factor = 1/2;
params.scale_model_factor = 1.0;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16;
params.scale_lambda = 1e-4;

params.learning_rate = 0.0158;

params.admm_iterations = 2;
params.admm_lambda_1 = 20;
params.admm_lambda_2 = 840;
params.admm_phi  = 1;
params.admm_mu   = 0.1;
params.admm_gamma = 27;

% Visualization
params.visualization = 1;               % Visualiza tracking and detection scores

% Initialize
params.seq = seq;

% Run tracker
results = tracker(params);
