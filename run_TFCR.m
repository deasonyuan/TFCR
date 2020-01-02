function results = run_TFCR(seq, res_path, bSaveImage, parameters)

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', 1)};

run ./matconvnet/matlab/vl_setupnn ;
addpath ./matconvnet/examples ;


opts.modelType = 'tracking' ;
opts.sourceModelPath = 'imagenet-vgg-verydeep-16.mat' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = false ;

display=0;

g=gpuDevice(1);
clear g;                             

config.gt = seq.init_rect;
config.imgList = seq.s_frames;
config.name = seq.name;
config.nFrames = numel(config.imgList);

global resize;

if strcmp(seq.name, 'Singer1_1')==1||strcmp(seq.name,'Liquor_1')==1||strcmp(seq.name,'BlurBody_1')==1||strcmp(seq.name,'Fleetface_1')==1||strcmp(seq.name,'Dudek_1')==1 ...
        ||strcmp(seq.name, 'Faceocc1_1')==1||strcmp(seq.name, 'Faceocc2_1')==1||strcmp(seq.name, 'BlurCar4_1')==1||strcmp(seq.name, 'Board_1')==1||strcmp(seq.name, 'Skating2-1_1')==1 ...
        ||strcmp(seq.name, 'Skating2-2_1')==1||strcmp(seq.name, 'Twinnings_1')==1||strcmp(seq.name, 'Trans_1')==1||strcmp(seq.name, 'BlurFace_1')==1||strcmp(seq.name, 'Human2_1')==1
    
	resize = 120;
elseif strcmp(seq.name, 'Girl2_1')==1||strcmp(seq.name, 'Jump_1')==1
	resize = 80;
end

if strcmp(seq.name, 'Airport_ce_1') == 1||strcmp(seq.name, 'Baby_ce_1') == 1 ...
	||strcmp(seq.name, 'Ironman_1') == 1
	     
	resize = 120;
elseif strcmp(seq.name, 'Iceskater_1') == 1||strcmp(seq.name, 'Motorbike_ce_1') == 1
	resize = 80;
elseif strcmp(seq.name, 'Skating_ce1_1') == 1||strcmp(seq.name, 'Skating2_1') == 1
	resize = 40;
end

results=TFCR_tracking(opts,varargin,config,display);     

end

