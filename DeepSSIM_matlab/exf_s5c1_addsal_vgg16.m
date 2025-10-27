
function features = exf_s5c1_addsal_vgg16(I, I_sal,params, gpu, vggmean, vggstd)
    if gpu
        I = gpuArray(I);
    end
    I = dlarray(double(I)/255,'SSC');
    vgg_mean = vggmean;
    vgg_std = vggstd;
    
    dlX = (I - vgg_mean)./vgg_std;
    % stage 1
    weights = dlarray(params.conv1_1.Weights);
    bias = dlarray(squeeze(params.conv1_1.Bias)');
    dlY = relu(dlconv(dlX,weights,bias,'Stride',1,'Padding','same'));
    for i = 1:64
        f64 = dlY(:,:,i);
        f64_r1{i,1} = I_sal.*f64;
    end
    dlY = cat(3,f64_r1{:});

    weights = dlarray(params.conv1_2.Weights);
    bias = dlarray(squeeze(params.conv1_2.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    % stage 2

    dlY = maxpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv2_1.Weights);
    bias = dlarray(squeeze(params.conv2_1.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv2_2.Weights);
    bias = dlarray(squeeze(params.conv2_2.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    % stage 3
    dlY = maxpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv3_1.Weights);
    bias = dlarray(squeeze(params.conv3_1.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv3_2.Weights);
    bias = dlarray(squeeze(params.conv3_2.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv3_3.Weights);
    bias = dlarray(squeeze(params.conv3_3.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    % stage 4
    dlY = maxpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv4_1.Weights);
    bias = dlarray(squeeze(params.conv4_1.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv4_2.Weights);
    bias = dlarray(squeeze(params.conv4_2.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv4_3.Weights);
    bias = dlarray(squeeze(params.conv4_3.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    % stage 5
    dlY = maxpool(dlY,2,'Stride',2);
    
    weights = dlarray(params.conv5_1.Weights);
    bias = dlarray(squeeze(params.conv5_1.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    features = dlY;
     
    weights = dlarray(params.conv5_2.Weights);
    bias = dlarray(squeeze(params.conv5_2.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
    
    weights = dlarray(params.conv5_3.Weights);
    bias = dlarray(squeeze(params.conv5_3.Bias)');
    dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

end





















