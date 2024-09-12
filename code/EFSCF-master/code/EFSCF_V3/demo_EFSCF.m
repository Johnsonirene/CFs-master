clc;
clear;
% Add paths
setup_paths();

% Load video information
base_path  =  './seq';

% video  = choose_video(base_path);

video = 'person3_s';

video_path = [base_path '/' video];

[seq, gt_boxes] = load_video_info(video_path); 

% Run  
results = run_EFSCF_V3(seq); 

%   compute the OP
pd_boxes = results.res;
% pd_boxes = [pd_boxes(:,1:2), pd_boxes(:,1:2) + pd_boxes(:,3:4) - ones(size(pd_boxes,1), 2)  ];
OP = zeros(size(gt_boxes,1),1);  % overlap rate
CE = zeros(size(gt_boxes,1),1);  % ceter location error
for i=1:size(gt_boxes,1)
    b_gt = gt_boxes(i,:);
    b_pd = pd_boxes(i,:);
    OP(i) = computePascalScore(b_gt,b_pd);
    centerGT = [b_gt(1) + (b_gt(3) - 1)/2, b_gt(2) + (b_gt(4) - 1)/2];
    centerPD = [b_pd(1) + (b_pd(3) - 1)/2, b_pd(2) + (b_pd(4) - 1)/2];
    CE(i) = sqrt((centerPD(1) - centerGT(1))^2 + (centerPD(2) - centerGT(2))^2);
end
OP_vid = sum(OP >= 0.5) / numel(OP);
CE_vid = sum(CE <= 20) / numel(CE);
FPS_vid = results.fps;
display([video  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(OP_vid)  '    ce:   '   num2str(CE_vid)]);

