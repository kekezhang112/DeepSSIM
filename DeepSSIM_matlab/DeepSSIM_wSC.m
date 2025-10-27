
function score = DeepSSIM_wSC(ref,sal_ref,dis,sal_dis,net_params,use_gpu,vggmean,vggstd)

[h1,w1,c1] = size(sal_ref);
if c1==3
    sal_ref_1 = sal_ref(:,:,1);
else
    sal_ref_1 = sal_ref;
end

sal_ref_1 = double(sal_ref_1)/255;
new_sal_ref = zeros(h1,w1);
for i = 1:h1
    for j = 1:w1
        if sal_ref_1(i,j)==0 % It can also be smaller than a certain threshold, ...
                             % such as 0.05, 0.01, etc.
            new_sal_ref(i,j)=0;
        else
            new_sal_ref(i,j)=1;
        end
    end
end

[h2,w2,c2] = size(sal_dis);
if c2 == 3
    sal_dis_1 = sal_dis(:,:,1);
else
    sal_dis_1 = sal_dis;
end
sal_dis_1 = double(sal_dis_1)/255;
new_sal_dis = zeros(h2,w2);
for i = 1:h2
    for j = 1:w2
        if sal_dis_1(i,j)==0 % It can also be smaller than a certain threshold, ...
                             % such as 0.05, 0.01, etc.
            new_sal_dis(i,j)=0;
        else
            new_sal_dis(i,j)=1;
        end
    end
end
ref_features = exf_s5c1_addsal_vgg16(ref,new_sal_ref,net_params,use_gpu,vggmean,vggstd);
dis_features = exf_s5c1_addsal_vgg16(dis,new_sal_dis,net_params,use_gpu,vggmean,vggstd);


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

end

