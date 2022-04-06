function Error = distance_centers(center,new_center,S,m)
    distance_xy = sqrt((center(4)-new_center(4))^2 + (center(5)-new_center(5))^2);
    distance_lab = sqrt((center(1)-new_center(1))^2 + (center(2)-new_center(2))^2 + (center(3)-new_center(3))^2);
    Error = distance_lab + (m/S)*distance_xy;
end