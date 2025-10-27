%%% Calculate Gram matrix

function grammatrix = gram_matrix(extractfeatures)
    f1 = extractfeatures; 
    [~,~,ch] = size(f1);
    clear ftmp3
    for j = 1:ch
        ftmp1 = f1(:,:,j);
        ftmp2 = ftmp1(:);
        ftmp3(:,j) = extractdata(ftmp2); 
    end
       
     grammatrix = ftmp3'*ftmp3;
     
end



