clear all;
close all;
load('mask.mat');
script_lecture_masque;

%récupérer l'image
imag = imread("images/D000.ppm");
figure
imshow(imag)

%définir les variables principales

[r,c,nb_chan] = size(imag);               % format de l'image
K = 108;                            % nombre de superpixels
N = size(imag,2) * size(imag,1);    % nombre de pixels tot
centers1 = zeros(K,5);              % valeur et position des centres 
m = 10;                             % pour le poids de la distance dans l'algo
Seuil = 10;                         % seuil de sortie de l'algorithme SLIC
max_iter = 1000;                    % nombre max d'itération


imag = cast(imag, 'double');

% Initialisation des centres des K superpixels
S = floor(sqrt(N/K));               % taille d'un superpixel initial          
kmeans = zeros(r,c);                % matrices représentant à quel superpixel appartient chaque pixel
nb_lig = floor(r/S);
nb_col = floor(c/S);

lig = 0;
col = 0;
lig_prim = 0;

for i=1:K
    kmeans(lig*S+1:(lig+1)*S,col*S+1:(col+1)*S) = i;
    bout_r = (lig+1)*S;
    if (r-bout_r)<S
        bout_r = r;
        kmeans((lig+1)*S:bout_r,col*S+1:(col+1)*S) = i;
    end
    bout_c = (col+1)*S;
    col_prim = col+1;
    if c-bout_c<S
        bout_c = c;
        kmeans(lig*S+1:bout_r,(col+1)*S:bout_c) = i;
        lig_prim = lig+1;
        col_prim = 0;
    end
    if i==K
        kmeans(lig*S+1:end,col*S:end) =i;
    end
    centers1(i,4:5) = floor([mean(lig*S+1:bout_r) mean(col*S+1:bout_c)]);
    centers1(i,1:3) = imag(centers1(i,4),centers1(i,5),:);
    col = col_prim;
    lig = lig_prim;
end

% affinement à rajouter (voir slides p2)
%% %%%% Partie affinement %%%%%%%%%% %%
for t=1:size(centers1, 1)
    grad = imgradient(imag(centers1(t,4)-1:centers1(t,4)+1, centers1(t,5)-1:centers1(t,5)+1), 'prewitt'); % les coordonnes
    minGrad = min(grad(:));
    [ci, cj] = find(grad == minGrad);
    centers1(t, 4:5) = [centers1(t,4)+ci(1)-2, centers1(t,5)+cj(1)-2];

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%


% imag = reshape(imag,[],3);
% imag = cast(imag, "double");
% centers1 =repmat(centers1,[1 1 10]);

%[cidx,ctrs,sumd,D]=kmeans(imag,K,'dist','sqEuclidean','emptyaction','drop','rep', 10, 'start',centers1,'disp','final');


E = Inf;
ncenters = centers1;
q = 0;
n_kmeans = kmeans;
%distance renvoie la distance proportionalisée des anciens centres aux
%nouveaux
%% Ajouter nb_iterations dans le while ??
while (E>Seuil && q<max_iter) 
    centers1 = ncenters;
    kmeans = n_kmeans;
%     nb = zeros(K,1);
%     centers2 = nbar;
%     centers1 = imag(centers2(:,1),centers2(:,2),:);
%     nbar = zeros(K,2);
%     for i=1:r
%         for j=1:c
%             % ca renvoi la classe dont le centre est le plus proche
%             classe = centre(imag,centers1,i,j); 
%             %ajouter pour le calcul du nouveau barycentre
%             nb(classe) = nb(classe) + 1;
%             nbar(classe,:) = nbar(classe,:) + [i j];
%             kmeans(i,j) = classe;
%         end
%     end
%     nbar = nbar./nb;
%     inter = nbar;

    % Calcul des superpixels
    for i=1:r
        for j=1:c
            % pour chaque pixel, choisir nouvelle classe dans voisinage
                % 2Sx2S qui minimise Ds = d_lab + m/S *d_xy
                %vois = imag(max(i-S,1):min(i+S,size(imag,1)), max(j-S,1):min(j+S,size(imag,2)),:);
                %vois_classe = kmeans(max(i-S,1):min(i+S,size(kmeans,1)), max(j-S,1):min(j+S,size(kmeans,2)));
            new_class = distance(i,j,imag(i,j,:),centers1,S,m,kmeans);
            n_kmeans(r,c) = new_class;
            
            %on fait quoi avec new_class ??
        end
    end
    % Mise à jour des centres
    Error = zeros(K,1);
    for t=1:K
        cluster_t = find(kmeans == t);
        moy_color = zeros(nb_chan,1);
        for chan=1:nb_chan
            I_chan = imag(:,:,chan);
            I_vect = I_chan(:);
            I_vect(cluster_t);
            moy_color(chan) = sum(I_vect(cluster_t))/size(cluster_t,1);
        end
        [row, col] = ind2sub([c,r],cluster_t);
        % Calcul de E (erreur résiduelle) (erreur quadratique moyenne?)
        moy_x = sum(row)/size(cluster_t,1);
        moy_y = sum(col)/size(cluster_t,1);
           
        new_t =  [moy_color', moy_x, moy_y];
        %mettre les bons elements dans distance
        %E(t)  = distance(centers1(t,:), moy_x, moy_y, moy_color, S);
        %calcul de distance entre centers1(t,:) et new_t
        Error(t) = distance_centers(centers1(t,:),new_t,S,m);
        ncenters(t,:) = new_t;
           
    end
    E = norm(Error);
    
    q = q+1;
end

imag = cast(imag, "uint8");

figure
kmeans = reshape(kmeans,r,c);
BW = boundarymask(kmeans);
imshow(imoverlay(imag,BW,'red'))




% figure
% gscatter(imag(:,1),imag(:,2),cidx,'bgm')
% hold on
% plot(ctrs(:,1),ctrs(:,2),'kx');