close all;
imag = imread("images/D000.ppm");
figure
imshow(imag)
K = 100;
[r,c,~] = size(imag);
N= size(imag,2) * size(imag,1);
centers1 = zeros(K,3);
S = sqrt(K/N);

for i=1:K
    c1 = mod(i-1,c)+1;
    r1 = mod(i-1,r)+1;
    centers1(i,:) = imag(ceil(r1*S + S/2),ceil(c1*S+S/2),:);
end
imag = reshape(imag,[],3);
imag = cast(imag, "double");
centers1 =repmat(centers1,[1 1 10]);
m = 1;

[cidx,ctrs,sumd,D]=kmeans(imag,K,'dist','custom',m,'emptyaction','drop','rep', 10, 'start',centers1,'disp','final');

figure
imag = reshape(imag,r,c, 3);
imag = cast(imag, "uint8");
imshow(imag)




% figure
% gscatter(imag(:,1),imag(:,2),cidx,'bgm')
% hold on
% plot(ctrs(:,1),ctrs(:,2),'kx');