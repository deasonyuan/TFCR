function net_online = initNet(target_sz1)
%Init network

channel=64;

rw=ceil(target_sz1(2)/2);
rh=ceil(target_sz1(1)/2);
fw=2*rw+1;
fh=2*rh+1;

net_online=dagnn.DagNN();

net_online.addLayer('conv11', dagnn.Conv('size', [fw,fh,channel,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1]), 'input1', 'conv_11', {'conv11_f', 'conv11_b'});

f = net_online.getParamIndex('conv11_f') ;
net_online.params(f).value=single(randn(fh,fw,channel,1) /...
    sqrt(rh*rw*channel))/1e8;
net_online.params(f).learningRate=1;
net_online.params(f).weightDecay=1e3;

f = net_online.getParamIndex('conv11_b') ;
net_online.params(f).value=single(zeros(1,1));
net_online.params(f).learningRate=2;
net_online.params(f).weightDecay=1e3;

net_online.addLayer('L2Loss',...
    RegressionL2Loss(),{'conv_11','label'},'objective');

end
