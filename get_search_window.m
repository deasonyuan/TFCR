function window_sz = get_search_window( target_sz, im_sz)
 
ratio=target_sz(1)/target_sz(2);
if ratio>1    
    window_sz=round(target_sz.*[5,5*ratio]);
else
    window_sz=round(target_sz.*[5/ratio,5]);
end

window_sz=window_sz-mod(window_sz,2)+1;

end

