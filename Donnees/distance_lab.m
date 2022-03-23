function d_lab = distance_lab(i,j,imag,S)
    d_lab = [];
    pixel = imag(i,j,:);
    for k=max(i-S,1):min(i+S,size(imag,1))
        for l=max(j-S,1):min(j+S,size(imag,2))
            if k~=i || j~=l
                dist = sqrt(sum((pixel - imag(k,l,:)).^2));
                d_lab = [d_lab ; dist];
            end
        end
    end
end