function Y = L2FocalLoss(X, C, dzdy)

assert(numel(X) == numel(C));
n = size(X,1) * size(X,2);
if nargin <= 2       
  Y = sum(((X(:)-C(:))).^2) ;  
else
    assert(numel(dzdy) == 1);    
    pre = X(:,:,:,1);
    label = C(:,:,:,1);
    u_pre = mean(mean(pre));
    std_pre = std(std(pre));
    u_label =  mean(mean(label));
    std_label = std(std(label));
    stan_pre = (pre-u_pre)/std_pre;
    stan_label = (label-u_label)/std_label;
    
    Y = reshape((2*(0.5*X(:)-C(:))), size(X)); 
end

end