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
    end
    % im est une matrice de dimension 4 qui contient 
    % l'ensemble des images couleur de taille : nb_lignes x nb_colonnes x nb_canaux 
    % im est donc de dimension nb_lignes x nb_colonnes x nb_canaux x nb_images
    im(:,:,:,i) = imread(nom); 
end

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
m = 100;                                % pour le poids de la distance dans l'algo
                                        % m/S doit être grand car il ne faudrait pas attribuer
                                        % à x un superpixel trop loin
Seuil = 0.05;                           % seuil de sortie de l'algorithme SLIC
max_iter = 100;                         % nombre max d'itération
kmeans = zeros(r,c,nb_images);          % matrices représentant à quel superpixel appartient chaque pixel pour chaque image
                                        % ainsi que sa couleur associée

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
        centers1(i,4:5,e) = floor([mean([lig*S:1+bout_r]) mean([col*S+1:bout_c])]);
        % il faudrait probablement plutot faire la couleur moyenne
        zone = imag(lig*S+1:bout_r,col*S+1:bout_c,:);
        centers1(i,1:3,e) = mean(reshape(zone,size(zone,1)*size(zone,2),size(zone,3)));
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
        % il n'y a pas à ajuster la couleur moyenne car la zonne délimitée
        % est la même
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
% figure1=figure;
% for e=1:nb_images
%     imag = cast(im(:,:,:,e), 'double');
%     % Variables globales
%     E = Inf;                            % Erreur
%     ncenters = centers1(:,:,e);         % centres intermédiaires
%     q = 0;                              % nombre initial de tours
%     nkmeans = kmeans(:,:,e);            % kmeans intermédiaire
%     initial = kmeans(:,:,e);
% 
%     title('image');
%     % tant que l'on n'atteint pas le seuil ou le nombre d'itérations max
%     while (E>Seuil && q<max_iter)
%         q = q+1;
%         centers = ncenters;
%         kmeans_e = nkmeans;
%         ncenters = zeros(K,5);
%         nombre = zeros(K,1);
%         
%         % Calcul des superpixels
%         % pour chaque pixel
%         for i=1:r
%             for j=1:c
%                 % on choisit s'il n'y a pas une classe dans un voisinage
%                 % 2S*2S qui ne serait pas plus proche
%                 new_class = distance(i,j,imag(i,j,:),centers,S,m,kmeans_e);
%                 % on met à jour dans nkmeans
%                 nkmeans(i,j) = new_class;
%                 % on prépare le calcul des nouveau centres
%                 ncenters(new_class,:) = ncenters(new_class,:) + [imag(i,j,1) imag(i,j,2) imag(i,j,3) i j];
%                 nombre(new_class) = nombre(new_class) +1;
%             end
%         end
%         % à enlever, juste de la vérification
%         %sum(sum(1-(nkmeans==initial)))
% 
%         % Mise à jour des centres
%         ncenters = ncenters./nombre;
% 
%         % Calcul de E (erreur résiduelle)
%         Error = zeros(K,1);
%         for t=1:K
%             Error(t) = distance_centers(centers(t,:),ncenters(t,:),S,m);
%         end
%         E = sum(Error)/K;
%         
%         
% %         hold off;
% %         BW = boundarymask(nkmeans);
% %         imshow(imoverlay(im(:,:,:,e),BW,'red'));
% %         hold on;
% %         plot(ncenters(:,5),ncenters(:,4), '.g');
% %         pause(0.02);
%         
%         
%     end
%     kmeans(:,:,e) = nkmeans;
% end



%pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                                             %
% Binarisation de l'image à partir des superpixels        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ........................................................%
% au lieu de tout relancer on charge les donnees pour l'exemple
load donnees;

binary = zeros(N,nb_images);
for e=1:nb_images
    kmeans_e = kmeans(:,:,e);
    kmeans_e = kmeans_e(:);
    for k=1:K
        superpixel = centers1(k,:,e);
        % s'il y a plus de rouge que de bleu
        if (superpixel(1)>superpixel(3))
            I = find(kmeans_e == k);
            binary(I,e) = 1;
        end
    end
end
binary = reshape(binary,r,c,nb_images);

figure;
axis on;
subplot(2,2,1); imshow(binary(:,:,1)); title('Mask 1 created');
subplot(2,2,2); imshow(binary(:,:,9)); title('Mask 9 created');
subplot(2,2,3); imshow(binary(:,:,17)); title('Mask 17 created');
subplot(2,2,4); imshow(binary(:,:,25)); title('Mask 25 created');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A FAIRE SI VOUS UTILISEZ LES MASQUES BINAIRES FOURNIS   %
% Chargement des masques binaires                         %
% de taille nb_lignes x nb_colonnes x nb_images           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('mask.mat');
script_lecture_masque;
%pause

% Variables de cette partie
r_depart = floor(r/2);
im_mask = 1-im_mask;
contour = [];
contour_indices = zeros(nb_images,1);

% pour chaque images binaire (chaque masque)
for i=1:nb_images
    c_depart = 1;
    
    % trouver en partant du milieu gauche, le premier pixel de l'objet
    while im_mask(r_depart,c_depart,i) == 0
        c_depart = c_depart+1;
    end
    % on récupère le contour
    autour = bwtraceboundary(im_mask(:,:,i),[r_depart c_depart], 'N');
    % on l'ajoute dans le vecteur contour (avec l'indice associé car chaque
    % contour a une longueur différente
    contour_indices(i) = size(autour,1);            % donne la longueur du vecteurs d'indices de contours
    contour = [contour ; autour];

    % on récupère directement les valeurs pour 1 9 17 et 25 pour
    % l'affichage suivant
    if (i==1)
        contour1 = autour;
    elseif (i==9)
        contour9 = autour;    
    elseif (i==17)
        contour17 = autour;
    elseif (i==25)
        contour25 = autour;
    end
end

% affichage intermédiaire des contours de masques pour vérification
figure;
axis on;
subplot(2,2,1); imshow(im_mask(:,:,1)); title('Mask 1 with boundary');
                hold on; plot(contour1(:,2),contour1(:,1),'g','LineWidth',2);
subplot(2,2,2); imshow(im_mask(:,:,9)); title('Mask 9 with boundary');
                hold on; plot(contour9(:,2),contour9(:,1),'g','LineWidth',2);
subplot(2,2,3); imshow(im_mask(:,:,17)); title('Mask 17 with boundary');
                hold on; plot(contour17(:,2),contour17(:,1),'g','LineWidth',2);
subplot(2,2,4); imshow(im_mask(:,:,25)); title('Mask 25 with boundary');
                hold on; plot(contour25(:,2),contour25(:,1),'g','LineWidth',2);

t = 0;                              % sert à noter où dans le vecteur contour nous sommes
% pour chaque image
% ATTENTION TROUVER COMMENT TOURNER L'IMAGE
figure;
for i=1:nb_images
    % on récupère son contour
    long_full = contour_indices(i);
    autour_full = contour(t+1:t+long_full,:);
    % on échantillonne le contour afin d'avoir un squelette avec moins de
    % bruit
    autour = autour_full([1:10:end],:);
    long = size(autour,1);

    % C sert à donner les indication pour relier les points de contour
    C = [1:long ; 2:long+1]';
    C(end,2) = 1;

    % Delaunay (n'est pas nécessaira à afficher)
%     DT = delaunayTriangulation(autour(:,1),autour(:,2), C);
%     IO = isInterior(DT);
%     triplot(DT(IO,:), DT.Points(:,1),DT.Points(:,2))

    % Voronoï et découpage pour ne garder que l'intérieur
    [vx,vy] = voronoi(autour(:,1), autour(:,2));
    % transforme godzilla en polygone
    pgon = polyshape(autour(:,1), autour(:,2));
    % on découpe pour ne garder que les segments intérieurs
    in1 = isinterior(pgon, vx(1,:),vy(1,:));
    in2 = isinterior(pgon, vx(2,:),vy(2,:));
    in = in1+in2;
    in = in==2;
    
    plot(vx(:,in),vy(:,in),'-b',autour_full(:,1),autour_full(:,2),'.r');
    % mise à jour de T pour aller au début du contour prochain
    t = t+long_full;

    pause(0.1)
    hold off
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET COMPLETER                              %
% quand vous aurez les images segmentées                  %
% Affichage des masques associes                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inutile à décommenter, c'est fait plus haut
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
    if size(l,2) > 1 && max(l)-min(l) > 1 && max(l)-min(l) < 36
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
        end
        [U,S,V] = svd(A);
        X = [X V(:,end)/V(end,end)];
        color = [color [R/size(l,2);G/size(l,2);B/size(l,2)]];
    end
end
fprintf('Calcul des points 3D termine : %d points trouves. \n',size(X,2));

%affichage du nuage de points 3D
figure;
hold on;
for i = 1:size(X,2)
    plot3(X(1,i),X(2,i),X(3,i),'.','col',color(:,i)/255);
end
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A COMPLETER                  %
% Tetraedrisation de Delaunay  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = delaunayTriangulation(X(1,:)',X(2,:)',X(3,:)');


% A DECOMMENTER POUR AFFICHER LE MAILLAGE
fprintf('Tetraedrisation terminee : %d tetraedres trouves. \n',size(T,1));
%Affichage de la tetraedrisation de Delaunay
figure;
tetramesh(T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET A COMPLETER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcul des barycentres de chacun des tetraedres
poids = [0.25 0.25 0.25 0.25 ; 0.4 0.2 0.2 0.2 ; 0.2 0.4 0.2 0.2; 0.2 0.2 0.4 0.2; 0.2 0.2 0.2 0.4];
nb_barycentres = size(poids,1); 
for i = 1:size(T,1)
    % Calcul des barycentres differents en fonction des poids differents
    % En commencant par le barycentre avec poids uniformes
    Ti = T(i,:); 
    Pi = T.Points(Ti,:);
    for k=1:nb_barycentres
         C_g(:,i,k) = [sum(Pi.*(poids(k,:)'*ones(1,size(Pi,2))),1) 1]; 
    end
end 

figure;
% A DECOMMENTER POUR VERIFICATION 
% A RE-COMMENTER UNE FOIS LA VERIFICATION FAITE
% Visualisation pour vérifier le bon calcul des barycentres
for i = 1:nb_images
   for k = 1:nb_barycentres
       o = P{i}*C_g(:,:,k);
       o = o./repmat(o(3,:),3,1);
%        imshow(im_mask(:,:,i));
%        hold on;
%        plot(o(2,:),o(1,:),'rx');
%        pause(0.02);
%        hold off;
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A DECOMMENTER ET A COMPLETER %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copie de la triangulation pour pouvoir supprimer des tetraedres
Tbis=triangulation(T.ConnectivityList,T.Points);
keep = true(size(T,1),1);
% Retrait des tetraedres dont au moins un des barycentres 
% ne se trouvent pas dans au moins un des masques des images de travail
% Pour chaque tétraèdre
for t=1:size(T,1)
     % récupère les barycentres du tétraèdre
     barycentre = C_g(:,t,:);
     % pour chaque image
     inside_image = true;
     for image=1:nb_images
         % pour chaque barycentre
         inside_barycentre = true;
         for b=1:nb_barycentres
            % projeter barycentre sur l'image
            projection = P{image}*barycentre(:,b);
            projection = projection./projection(3);
            if (projection(1)>=1 && projection(2)>=1 && projection(1)<=r && projection(2)<=c)
                if (im_mask(round(projection(1)),round(projection(2)),image)== 0)
                    inside_barycentre = false;
                    break;
                end
            end
         end
        if inside_barycentre == false
            inside_image = false;
            break;
        end
     end
     if (inside_image==false) 
         keep(t) = 0;
     end
end

Tbis = Tbis(keep,:);




% A DECOMMENTER POUR AFFICHER LE MAILLAGE RESULTAT
% Affichage des tetraedres restants
fprintf('Retrait des tetraedres exterieurs a la forme 3D termine : %d tetraedres restants. \n',size(Tbis,1));
figure;
trisurf(Tbis,X(1,:),X(2,:),X(3,:));

% Sauvegarde des donnees
save donnees;

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
