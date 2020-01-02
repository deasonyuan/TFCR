function results = TFCR_tracking( opts, varargin, config, display)

global objSize;

LocGt=config.gt;
num_channels=64;

% training options (SGD)
opts.train = struct([]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
output_sigma_factor = 0.1;

gpuDevice(1);
[net1,avgImg]=initVGG16Net();

nFrame=config.nFrames;
[Gt]=config.gt;
result = zeros(length(nFrame), 4); result(1,:) = Gt(1,:);

scale=1;
global resize;
objSize=Gt(1,3:4);
if objSize(1)*objSize(2)>resize*resize
    scale=resize/max(objSize);    
    disp('resized');
end

im1=imread(config.imgList{1});
im=imresize(im1,scale);
cell_size=4;
if size(im,3)==1
    im = cat(3, im, im, im);
    im1 = cat(3, im1, im1, im1);
end

targetLoc=round(Gt(1,:)*scale);
target_sz=[targetLoc(4) targetLoc(3)];
im_sz=size(im);
window_sz = get_search_window(target_sz, im_sz);
l1_patch_num = ceil(window_sz/ cell_size);
l1_patch_num=l1_patch_num-mod(l1_patch_num,2)+1;
cos_window = hann(l1_patch_num(1)) * hann(l1_patch_num(2))';

sz_window=size(cos_window);
pos = [targetLoc(2), targetLoc(1)] + floor(target_sz/2);
patch = get_subwindow(im, pos, window_sz);
meanImg=zeros(size(patch));
meanImg(:,:,1)=avgImg(1);
meanImg(:,:,2)=avgImg(2);
meanImg(:,:,3)=avgImg(3);
patch1 = single(patch) - meanImg;
net1.eval({'input',gpuArray(patch1)});

index=[23];
feat=cell(length(index),1);
for i=1:length(index)
    feat1 = gather(net1.vars(index(i)).value);
    feat1 = imResample(feat1, sz_window(1:2));
    feat{i} = bsxfun(@times, feat1, cos_window);                        
end
feat=feat{1};

[hf,wf,cf]=size(feat);
matrix=reshape(feat,hf*wf,cf);
coeff = pca(matrix);
coeff=coeff(:,1:num_channels);

target_sz1=ceil(target_sz/cell_size);
output_sigma = target_sz1*output_sigma_factor;
label=gaussian_shaped_labels(output_sigma, l1_patch_num);

label1=imresize(label,[size(im1,1) size(im1,1)])*255;
patch1=imresize(patch,[size(im1,1) size(im1,1)]);
imd=[im1];

%%%%%%%%%-----------Display First frame----------
if display    
    figure(2);
    set(gcf,'Position',[200 300 480 320],'MenuBar','none','ToolBar','none');
    hd = imshow(imd,'initialmagnification','fit'); hold on;
    rectangle('Position', Gt(1,:), 'EdgeColor', [0 0 1], 'Linewidth', 1);    
    set(gca,'position',[0 0 1 1]);
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;   
end

%%%%%%%%%-----------first frame initialization-----------
trainOpts.numEpochs=20;

feat_=reshape(feat,hf*wf,cf);
feat_=feat_*coeff;
featPCA=reshape(feat_,hf,wf,num_channels);
net_online=initNet(target_sz1);

trainOpts.batchSize = 1 ;
trainOpts.numSubBatches = 1 ;
trainOpts.continue = true ;
trainOpts.gpus = 1 ;
trainOpts.prefetch = true ;

trainOpts.expDir = opts.expDir ;
trainOpts.learningRate=5e-8;
trainOpts.weightDecay= 1;

train=1;
imdb=[];
input={featPCA label};
featPCA1st=featPCA;
opts.train.gpus=1;
bopts.useGpu = numel(opts.train.gpus) > 0 ;
info = cnn_train_dag(net_online, imdb, input,getBatchWrapper(bopts),trainOpts, 'train', train, opts.train) ;                
net_online.move('gpu');

%%%%%%%%%-----------online prediction----------
motion_sigma_factor=0.6;
cell_size=4;
global num_update;
num_update=2;
cur=1;
feat_update=cell(num_update,1);
label_update=cell(num_update,1);
target_szU=target_sz;
time = 0;
ScaleFactor = 1;
window_sz_ori = window_sz;
suppress_map = ones(size(cos_window,1),size(cos_window,2));

for i=2:nFrame       
    im1=imread(config.imgList{i});          
    im=imresize(im1,scale);    
    if size(im1,3)==1
        im = cat(3, im, im, im);
        im1 = cat(3, im1, im1, im1);
    end 
    fprintf('frame=%.0f\n',i);
    tic 
    window_sz = window_sz * ScaleFactor;
    window_sz = round(window_sz);
    patch = get_subwindow(im, pos, window_sz);  
    patch = imresize(patch, window_sz_ori);
    patch1 = single(patch) - meanImg;    
    net1.eval({'input',gpuArray(patch1)});
    
    bb = result(i-1,:);
    resp_sz = window_sz;    
    
    feat=cell(length(index),1);
    feat_suppress = cell(length(index),1);
    
    for j=1:length(index)
        feat1 = gather(net1.vars(index(j)).value);
        feat1 = imResample(feat1, sz_window(1:2));
        feat{j} = bsxfun(@times, feat1, cos_window); 
        feat_suppress{j} = feat{j};
    end
    
    feat = feat{1};
    feat_suppress = feat_suppress{1};
        
    feat_=reshape(feat,hf*wf,cf);
    feat_=feat_*coeff;
    featPCA=reshape(feat_,hf,wf,num_channels);  
    
    feat_suppress=reshape(feat_suppress,hf*wf,cf);
    feat_sup =feat_suppress*coeff;
    feat_supPCA=reshape(feat_sup,hf,wf,num_channels); 
    feat_catPCA = cat(4,featPCA, feat_supPCA);
    net_online.eval({'input1',gpuArray(feat_catPCA)}); 
    regression_map=gather(net_online.vars(2).value);        
              
    response = regression_map(:,:,1,2);
    response_aux = regression_map(:,:,1,1);
    response_szwin = imresize(response,window_sz);
    [vert_delta, horiz_delta] = find(response_szwin == max(response_szwin(:)), 1);
    vert_delta  = vert_delta  - ceil(window_sz(2)/2);
    horiz_delta = horiz_delta - ceil(window_sz(1)/2);
    pos = pos + [vert_delta, horiz_delta];  

    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
    vert_delta  = vert_delta  - ceil(hf/2);
    horiz_delta = horiz_delta - ceil(wf/2);  
               
    target_szU_new=scale_estimation(im,pos,target_szU,window_sz_ori, net1,net_online,coeff,meanImg);            
    ScaleFactor = min(target_szU_new/target_szU);  
    target_szU = target_szU_new;
    targetLoc=[pos([2,1]) - target_szU([2,1])/2, target_szU([2,1])];    
    result(i,:)=round(targetLoc/scale);                  
            
    label1=imresize(regression_map,[size(im1,1) size(im1,1)])*255;
    patch1=imresize(patch,[size(im1,1) size(im1,1)]);     
    imd=[im1];    
    
%%%%%%%%%-----------Display current frame---------
    if display   
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',imd);    
        hold on;                                
        rectangle('Position', result(i,:), 'EdgeColor', [0 0 1], 'Linewidth', 1);                       
        set(gca,'position',[0 0 1 1]);
        text(10,10,num2str(i),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        bb = result(i-1,:);
        resp_sz = window_sz;
        xs = floor(bb(1)+bb(3)/2) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
        ys = floor(bb(2)+bb(4)/2) + (1:resp_sz(2)) -  floor(resp_sz(2)/2);
        resp_handle = imagesc(xs, ys, imresize(response,resp_sz));colormap jet;
        colorbar;
        alpha(resp_handle, 0.5);
        hold off;
        drawnow;  
    end
    
%%%%%%%%%-----------Model update-----------
    labelU=circshift(label,[vert_delta,horiz_delta]);
    feat_update{cur}=featPCA;              
    label_update{cur}=labelU; 
     
    if cur==num_update    
        trainOpts.batchSize = 1 ;
        trainOpts.numSubBatches = 1 ;
        trainOpts.continue = true ;
        trainOpts.gpus = 1 ;
        trainOpts.prefetch = true ;
        trainOpts.learningRate = 2e-8;        
        trainOpts.weightDecay= 1;
        trainOpts.numEpochs = 2;

        train=1;
        imdb=[];
        input={feat_update label_update};
        opts.train.gpus=1;
        bopts.useGpu = numel(opts.train.gpus) > 0 ;
                            
        info = cnn_train_dag_update(net_online, imdb, input,getBatchWrapper(bopts), trainOpts, 'train', train, opts.train) ;        
        net_online.move('gpu');   
        cur=1;        
    else 
        cur=cur+1;            
    end
    time = time + toc;
end
results.res = result;
results.fps = nFrame/time;
results.type = 'rect';
end

function fn = getBatchWrapper(opts)
fn = @(imdb,batch) getBatch(imdb,batch,false,opts,'prefetch',nargout==0) ;
end
