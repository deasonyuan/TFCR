function info = Demo()

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', 1)};

run ./matconvnet/matlab/vl_setupnn ;
addpath ./matconvnet/examples ;


opts.modelType = 'tracking' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = false ;

display=1;

g=gpuDevice(1);
clear g;                             

test_seq='Skiing';
[config]=config_list(test_seq);

results = TFCR_tracking(opts,varargin,config,display);        
       



