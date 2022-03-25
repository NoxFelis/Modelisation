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
y_depart = floor(c/2);
im_mask = 1-im_mask;
%colormap("hsv");

for i=1:pic
    x_depart = r;
    a=0;
    while im_mask(x_depart,y_depart,pic) == 0
        x_depart = x_depart-1;
    end
    %contour = bwtraceboundary(im_mask(:,:,i),[x_depart y_depart], 'E');
    figure
    axis on;
    imshow(im_mask(:,:,i))
    hold on;
    plot(y_depart,x_depart, 'r+', 'MarkerSize', 5, 'Linewidth', 2);
%     hold on;
%     plot(contour(:,2),contour(:,1),'g','LineWidth',4);
    pause
    close all;
end
