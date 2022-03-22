close all;
imag = imread("images/D000.ppm");
figure
imshow(imag)
K = 100;
[r,c,~] = size(imag);
N = size(imag,2) * size(imag,1);
centers1 = zeros(K,3);  %valeur des centres
centers2 = zeros(K,2);  %position des centres
S = sqrt(K/N);
kmeans = zeros(r,c);
for i=1:K
    c1 = mod(i-1,c)+1;
    r1 = mod(i-1,r)+1;
    centers2(i,:) = [ceil(r1*S + S/2) ceil(c1*S+S/2)];
    centers1(i,:) = imag(centers2(i,1),centers2(i,2),:);
    kmeans((r1-1)*S:r1*S,(c1-1)*S:c1*S) = i;
end

% imag = reshape(imag,[],3);
% imag = cast(imag, "double");
% centers1 =repmat(centers1,[1 1 10]);

%[cidx,ctrs,sumd,D]=kmeans(imag,K,'dist','sqEuclidean','emptyaction','drop','rep', 10, 'start',centers1,'disp','final');

imag = cast(imag, 'double');
nbar = centers2;
inter = zerors(K,2);

while (max(distance(centers1,inter))>epsilon) 
    nb = zeros(K,1);
    centers2 = nbar;
    centers1 = imag(centers2(:,1),centers2(:,2),:);
    nbar = zeros(K,2);
    for i=1:r
        for j=1:c
            % ca renvoi la classe dont le centre est le plus proche
            classe = centre(imag,centers1,i,j); 
            %ajouter pour le calcul du nouveau barycentre
            nb(classe) = nb(classe) + 1;
            nbar(classe,:) = nbar(classe,:) + [i j];
            kmeans(i,j) = classe;
        end
    end
    nbar = nbar./nb;
    inter = nbar;
end


figure
imag = reshape(kmeans,r,c, 3);
imag = cast(kmeans, "uint8");
imshow(kmeans)




% figure
% gscatter(imag(:,1),imag(:,2),cidx,'bgm')
% hold on
% plot(ctrs(:,1),ctrs(:,2),'kx');