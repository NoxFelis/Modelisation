close all;
clear all;
load('mask.mat');
script_lecture_masque;
pause
close all;

% sert just à comprendre la mécanique du bwtraceboundary
% 
% test = 1-im_mask(:,:,1);
% figure
% contour = bwtraceboundary(test,[288 360], 'E');
% imshow(test);
% hold on
% plot(contour(:,2),contour(:,1),'g','LineWidth',4)

[r,c,pic] = size(im_mask);
r_depart = floor(r/2);
im_mask = 1-im_mask;
contour = [];
contour_indices = zeros(pic,1);
%colormap("hsv");

for i=1:pic
    c_depart = 1;
    
    while im_mask(r_depart,c_depart,i) == 0
        c_depart = c_depart+1;
    end
    autour = bwtraceboundary(im_mask(:,:,i),[r_depart c_depart], 'N');
    contour_indices(i) = size(autour,1);            % donne la longueur du vecteurs d'indices de contours
    contour = [contour ; autour];
%     figure
%     axis on;
%     imshow(im_mask(:,:,i))
%     hold on;
%     plot(c_depart, r_depart, 'r+', 'MarkerSize', 5, 'Linewidth', 2); %attention c'est inversé à l'affichage
%     hold on;
%     plot(autour(:,2),autour(:,1),'g','LineWidth',4);
%     pause
%     close all;
end

figure
t = contour_indices(1);
autour = contour(1:t,:);
[vx vy] = voronoi(autour(:,1), autour(:,2))
in = inpolygon(vx,vy, autour(:,1),autour(:,2));

plot(vx,vy,'-b',autour(:,1),autour(:,2),'.r');
%voronoi(autour(:,1), autour(:,2))
%DT = delaunay(autour(:,1),autour(:,2))
%triplot(DT,autour(:,1),autour(:,2));
hold on;

