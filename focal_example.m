pre = X(:,:,:,1);
label = C(:,:,:,1);
u_pre = mean(mean(pre));
std_pre = std(std(pre));
u_label =  mean(mean(label));
std_label = std(std(label));
stan_pre = (pre-u_pre)/std_pre;
stan_label = (label-u_label)/std_label;
figure(7);imagesc(stan_pre);colorbar;
figure(8);imagesc(stan_label);colorbar;