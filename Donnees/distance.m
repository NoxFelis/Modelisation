function new_class = distance(i,j,pixel,centers1,S,m,kmeans)
    new_class = kmeans(i,j);
    dist = Inf;
%     for k=max(i-S,1):min(i+S,size(imag,1))
%         for l=max(j-S,1):min(j+S,size(imag,2))
%             if k~=i || j~=l
%                 dist_xy = sqrt((i-k)^2 + (j-l)^2);
%                 dist_lab = sqrt(sum((pixel - imag(k,l,:)).^2));
%                 distance = dist_lab + (m/S)*dist_xy;
%                 if (distance<=dist)
%                     dist = distance;
%                     x = k;
%                     y = l;
%                     new_class = kmeans(k,l);
%                 end
%             end
%         end
%     end
    for t=1:size(centers1,1)
        if (abs(centers1(t,4)-i)<S && abs(centers1(t,5)-j)<S)
            dist_xy = sqrt((i-centers1(t,4))^2+(j-centers1(t,5))^2);
            dist_lab = sqrt(sum((pixel-centers1(t,[1:3])).^2));
            distance = dist_lab + (m/S)*dist_xy;
            if (distance<=dist)
                dist = distance;
                new_class = t;
            end
        end
    end
end