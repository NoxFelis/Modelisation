function new_class = distance(i,j,pixel,centers1,S,m,kmeans)
    new_class = kmeans(i,j);
    dist = Inf;
    S2 = (2*S);
    for t=1:size(centers1,1)
        dist_x = (i-centers1(t,4))^2;
        dist_y = (j-centers1(t,5))^2;
        dist_xy = sqrt(dist_x+dist_y);
        if (dist_xy<=S2)
            dist_lab = sqrt(sum(([pixel(1)-centers1(t,1); pixel(2)-centers1(t,2);pixel(3)-centers1(t,3)]).^2));
            distance = dist_lab + (m/S)*dist_xy;
            if (distance<=dist)
                dist = distance;
                new_class = t;
            end
        end
    end
end