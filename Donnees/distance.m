function [new_class,x,y] = distance(i,j,imag,S,m,kmeans)
    new_class = kmeans(i,j);
    x,y = i,j;
    pixel = imag(i,j,:);
    dist = Inf;
    for k=max(i-S,1):min(i+S,size(imag,1))
        for l=max(j-S,1):min(j+S,size(imag,2))
            if k~=i || j~=l
                dist_xy = sqrt((i-k)^2 + (j-l)^2);
                dist_lab = sqrt(sum((pixel - imag(k,l,:)).^2));
                distance = dist_lab + (m/S)*dist_xy;
                if (distance<=dist)
                    dist = distance;
                    x = k;
                    y = l;
                    new_class = kmeans(k,l);
                end
            end
        end
    end
end