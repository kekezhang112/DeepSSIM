
function score = DeepSSIM_woSC(ref,dis,net_params,use_gpu,vggmean,vggstd)

ref_features = exf_s5c1_vgg16(ref,net_params,use_gpu,vggmean,vggstd);
dis_features = exf_s5c1_vgg16(dis,net_params,use_gpu,vggmean,vggstd);

% Calculate the gram matrix of the feature maps.
ref_gram = gram_matrix(ref_features);
dis_gram = gram_matrix(dis_features);
sim_matrix = zeros(128,128);
c1 = 1e-6;

for i = 1:4:512  
    for j = 1:4:512 
          
        A_4 = ref_gram(i:i+3, j:j+3);  
        B_4 = dis_gram(i:i+3, j:j+3);  
        ref_mean = mean(A_4,'all');
        dis_mean = mean(B_4,'all');
        ref_var = mean((A_4-ref_mean).^2,[1,2]);
        dis_var = mean((B_4-dis_mean).^2,[1,2]);
        ref_dist_cov = mean(A_4.*B_4,[1,2])-ref_mean.*dis_mean;
        
        sim_4x4 = (2*ref_dist_cov+c1)./(ref_var+dis_var+c1);
        sim_matrix((i-1)/4 + 1, (j-1)/4 + 1) = sim_4x4;
    end
end

score = mean2(sim_matrix);

