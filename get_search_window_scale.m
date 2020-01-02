function window_sz = get_search_window_test( target_sz, im_sz)
ratio=target_sz(1)/target_sz(2);
if ratio>1    
    window_sz=round(target_sz.*[2,2*ratio]);
else
    window_sz=round(target_sz.*[2/ratio,2]);
end
window_sz=window_sz-mod(window_sz,2)+1;
end

