clear;
close all;
% Nombre d'images utilisees
nb_images = 36; 

% chargement des images
for i = 1:nb_images
    if i<=10
        nom = sprintf('images/viff.00%d.ppm',i-1);
    else
        nom = sprintf('images/viff.0%d.ppm',i-1);
    end;
    % im est une matrice de dimension 4 qui contient 
    % l'ensemble des images couleur de taille : nb_lignes x nb_colonnes x nb_canaux 
    % im est donc de dimension nb_lignes x nb_colonnes x nb_canaux x nb_images
    im(:,:,:,i) = imread(nom); 
end;

% Affichage des images
figure; 
subplot(2,2,1); imshow(im(:,:,:,1)); title('Image 1');
subplot(2,2,2); imshow(im(:,:,:,9)); title('Image 9');
subplot(2,2,3); imshow(im(:,:,:,17)); title('Image 17');
subplot(2,2,4); imshow(im(:,:,:,25)); title('Image 25');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                                             %
% Calculs des superpixels                                 % 
% Conseil : afficher les germes + les régions             %
% à chaque étape / à chaque itération                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ........................................................%
% DÉFINIR LES VARIABLES PRINCIPALES

[r,c,nb_chan,nb_images] = size(im);     % format de l'image
K = 108;                                % nombre de superpixels
N = size(im,2) * size(im,1);            % nombre de pixels tot
centers1 = zeros(K,5,nb_images);        % valeur et position des centres pour chaque image 
m = 10;                                 % pour le poids de la distance dans l'algo
Seuil = 10;                             % seuil de sortie de l'algorithme SLIC
max_iter = 100;                        % nombre max d'itération
kmeans = zeros(r,c,nb_images);          % matrices représentant à quel superpixel appartient chaque pixel pour chaque image

% Initialisation des centres des K superpixels
S = floor(sqrt(N/K));                   % taille d'un superpixel initial          
nb_lig = floor(r/S);                    % nombre initial de superpixels par ligne
nb_col = floor(c/S);                    % nombre initial de superpixels pas colonne

% pour chaque image de la base
for e=1:nb_images
    % on récupère l'image sous le nom imag et on la convertit en double
    imag = cast(im(:,:,:,e),'double');
   
    % variables intermédiaires pour l'initialisation des superpixels
    lig = 0;
    col = 0;
    lig_prim = 0;
    
    %pour chaque superpixel
    for i=1:K
        % on remplie dans kmeans les pixels appartenant à cette classe
        bout_r = (lig+1)*S;
        % on allonge si on est au bout d'une ligne ou d'une colonne et
        % qu'il n'y a pas assez pour faire plus
        if (r-bout_r)<S
            bout_r = r;
        end
%         kmeans(lig*S+1:bout_r,col*S+1:(col+1)*S,e) = i;
        bout_c = (col+1)*S;
        col_prim = col+1;
        if c-bout_c<S
            bout_c = c;
            lig_prim = lig+1;
            col_prim = 0;
        end
        kmeans(lig*S+1:bout_r,col*S+1:bout_c,e) = i;
        % tout ce rui reste (dans le petit coin) est pour la dernière
        % classe
        if i==K
            kmeans(lig*S+1:end,col*S:end,e) =i;
            bout_r = r;
            bout_c = c;
        end
        % on calcule le centre de chaque super pixel en position et en
        % couleur 
        % ATTENTION: IL Y A UN PROBLEME DANS LES CALCULS DE CENTRES
        centers1(i,4:5,e) = floor([(lig*S+1+bout_r)/2 (col*S+1+bout_c)/2]);
        % il faudrait probablement plutot faire la couleur moyenne
        zone = imag(lig*S+1:bout_r,col*S+1:bout_c,:);
        %centers1(i,1:3,e) = mean(reshape(zone,size(zone,1)*size(zone,2),size(zone,3)));
        centers1(i,1:3,e) = imag(centers1(i,4,e),centers1(i,5,e));
        col = col_prim;
        lig = lig_prim;
    end
    
end

% affinement à rajouter (voir slides p2)
% Partie affinement
for e=1:nb_images
    imag = cast(im(:,:,:,e), 'double');
    for t=1:size(centers1, 1)
        grad = imgradient(imag(centers1(t,4,e)-1:centers1(t,4,e)+1, centers1(t,5,e)-1:centers1(t,5,e)+1), 'prewitt'); % les coordonnes
        minGrad = min(grad(:));
        [ci, cj] = find(grad == minGrad);
        centers1(t, 4:5,e) = [centers1(t,4,e)+ci(1)-2, centers1(t,5,e)+cj(1)-2];
    end
end

% affichage intermédiaire pour les 4 memes images
figure
BW1 = boundarymask(reshape(kmeans(:,:,1),r,c));
BW9 = boundarymask(reshape(kmeans(:,:,9),r,c));
BW17 = boundarymask(reshape(kmeans(:,:,17),r,c));
BW25 = boundarymask(reshape(kmeans(:,:,25),r,c));

%attention il faut inverser entre row/column et x/y -> row = y column = x
subplot(2,2,1); imshow(imoverlay(im(:,:,:,1),BW1,'red')); title('Image 1 with initial superpixels');
                hold on; plot(centers1(:,5,1),centers1(:,4,1), '.g');
subplot(2,2,2); imshow(imoverlay(im(:,:,:,9),BW9,'red')); title('Image 9 with initial superpixels');
                hold on; plot(centers1(:,5,9),centers1(:,4,9), '.g');
subplot(2,2,3); imshow(imoverlay(im(:,:,:,17),BW17,'red')); title('Image 17 with initial superpixels');
                hold on; plot(centers1(:,5,17),centers1(:,4,17), '.g');
subplot(2,2,4); imshow(imoverlay(im(:,:,:,25),BW25,'red')); title('Image 25 with initial superpixels');
                hold on; plot(centers1(:,5,25),centers1(:,4,25), '.g');

% Algorithme SLIC

figure1=figure;
for e=1:1
    imag = cast(im(:,:,:,e), 'double');
    % Variables globales
    E = Inf;                            % Erreur
    ncenters = centers1(:,:,e);         % centres intermédiaires
    q = 0;                              % nombre initial de tours
    nkmeans = kmeans(:,:,e);            % kmeans intermédiaire
    initial = kmeans(:,:,e);

    title('image');
    % tant que l'on n'atteint pas le seuil ou le nombre d'itérations max
    while (E>Seuil && q<max_iter)
        centers = ncenters;
        kmeans_e = nkmeans;
        ncenters = zeros(K,5);
        nombre = zeros(K,1);
        
        % Calcul des superpixels
        % pour chaque pixel
        for i=1:r
            for j=1:c
                % on choisit s'il n'y a pas une classe dans un voisinage
                % 2S*2S qui ne serait pas plus proche
                new_class = distance(i,j,imag(i,j,:),centers,S,m,kmeans_e)
                % on met à jour dans nkmeans
                nkmeans(r,c) = new_class;
                % on prépare le calcul des nouveau centres
                ncenters(new_class,:) = ncenters(new_class,:) + [imag(i,j,1) imag(i,j,2) imag(i,j,3) i j];
                nombre(new_class) = nombre(new_class) +1;
            end
        end
        sum(sum(1-(nkmeans==initial)))

        % Mise à jour des centres
        ncenters = ncenters./nombre;

        % Calcul de E (erreur résiduelle)
        Error = zeros(K,1);
        for t=1:K
            Error(t) = distance_centers(centers(t,:),ncenters(t,:),S,m);
        end
        E = max(Error);
        hold off;
        BW = boundarymask(nkmeans);
        imshow(imoverlay(im(:,:,:,e),BW,'red'));
        hold on;
        plot(centers(:,5),centers(:,4), '.g');
        pause(0.2);
        
        
        q = q+1;
    end
    close figure1;
end
a=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                                             %
% Binarisation de l'image à partir des superpixels        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ........................................................%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A FAIRE SI VOUS UTILISEZ LES MASQUES BINAIRES FOURNIS   %
% Chargement des masques binaires                         %
% de taille nb_lignes x nb_colonnes x nb_images           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ... 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET COMPLETER                              %
% quand vous aurez les images segmentées                  %
% Affichage des masques associes                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% subplot(2,2,1); ... ; title('Masque image 1');
% subplot(2,2,2); ... ; title('Masque image 9');
% subplot(2,2,3); ... ; title('Masque image 17');
% subplot(2,2,4); ... ; title('Masque image 25');

% chargement des points 2D suivis 
% pts de taille nb_points x (2 x nb_images)
% sur chaque ligne de pts 
% tous les appariements possibles pour un point 3D donne
% on affiche les coordonnees (xi,yi) de Pi dans les colonnes 2i-1 et 2i
% tout le reste vaut -1
pts = load('viff.xy');
% Chargement des matrices de projection
% Chaque P{i} contient la matrice de projection associee a l'image i 
% RAPPEL : P{i} est de taille 3 x 4
load dino_Ps;

% Reconstruction des points 3D
X = []; % Contient les coordonnees des points en 3D
color = []; % Contient la couleur associee
% Pour chaque couple de points apparies
for i = 1:size(pts,1)
    % Recuperation des ensembles de points apparies
    l = find(pts(i,1:2:end)~=-1);
    % Verification qu'il existe bien des points apparies dans cette image
    if size(l,2) > 1 & max(l)-min(l) > 1 & max(l)-min(l) < 36
        A = [];
        R = 0;
        G = 0;
        B = 0;
        % Pour chaque point recupere, calcul des coordonnees en 3D
        for j = l
            A = [A;P{j}(1,:)-pts(i,(j-1)*2+1)*P{j}(3,:);
            P{j}(2,:)-pts(i,(j-1)*2+2)*P{j}(3,:)];
            R = R + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),1,j));
            G = G + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),2,j));
            B = B + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),3,j));
        end;
        [U,S,V] = svd(A);
        X = [X V(:,end)/V(end,end)];
        color = [color [R/size(l,2);G/size(l,2);B/size(l,2)]];
    end;
end;
fprintf('Calcul des points 3D termine : %d points trouves. \n',size(X,2));

%affichage du nuage de points 3D
figure;
hold on;
for i = 1:size(X,2)
    plot3(X(1,i),X(2,i),X(3,i),'.','col',color(:,i)/255);
end;
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                  %
% Tetraedrisation de Delaunay  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T = ...                      

% A DECOMMENTER POUR AFFICHER LE MAILLAGE
% fprintf('Tetraedrisation terminee : %d tetraedres trouves. \n',size(T,1));
% Affichage de la tetraedrisation de Delaunay
% figure;
% tetramesh(T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET A COMPLETER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcul des barycentres de chacun des tetraedres
% poids = ... 
% nb_barycentres = ... 
% for i = 1:size(T,1)
    % Calcul des barycentres differents en fonction des poids differents
    % En commencant par le barycentre avec poids uniformes
%     C_g(:,i,1)=[ ...

% A DECOMMENTER POUR VERIFICATION 
% A RE-COMMENTER UNE FOIS LA VERIFICATION FAITE
% Visualisation pour vérifier le bon calcul des barycentres
% for i = 1:nb_images
%    for k = 1:nb_barycentres
%        o = P{i}*C_g(:,:,k);
%        o = o./repmat(o(3,:),3,1);
%        imshow(im_mask(:,:,i));
%        hold on;
%        plot(o(2,:),o(1,:),'rx');
%        pause;
%        close;
%    end
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET A COMPLETER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copie de la triangulation pour pouvoir supprimer des tetraedres
% tri=T.Triangulation;
% Retrait des tetraedres dont au moins un des barycentres 
% ne se trouvent pas dans au moins un des masques des images de travail
% Pour chaque barycentre
% for k=1:nb_barycentres
% ...

% A DECOMMENTER POUR AFFICHER LE MAILLAGE RESULTAT
% Affichage des tetraedres restants
% fprintf('Retrait des tetraedres exterieurs a la forme 3D termine : %d tetraedres restants. \n',size(Tbis,1));
% figure;
% trisurf(tri,X(1,:),X(2,:),X(3,:));

% Sauvegarde des donnees
% save donnees;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSEIL : A METTRE DANS UN AUTRE SCRIPT %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load donnees;
% Calcul des faces du maillage à garder
% FACES = ...;
% ...

% fprintf('Calcul du maillage final termine : %d faces. \n',size(FACES,1));

% Affichage du maillage final
% figure;
% hold on
% for i = 1:size(FACES,1)
%    plot3([X(1,FACES(i,1)) X(1,FACES(i,2))],[X(2,FACES(i,1)) X(2,FACES(i,2))],[X(3,FACES(i,1)) X(3,FACES(i,2))],'r');
%    plot3([X(1,FACES(i,1)) X(1,FACES(i,3))],[X(2,FACES(i,1)) X(2,FACES(i,3))],[X(3,FACES(i,1)) X(3,FACES(i,3))],'r');
%    plot3([X(1,FACES(i,3)) X(1,FACES(i,2))],[X(2,FACES(i,3)) X(2,FACES(i,2))],[X(3,FACES(i,3)) X(3,FACES(i,2))],'r');
% end;
