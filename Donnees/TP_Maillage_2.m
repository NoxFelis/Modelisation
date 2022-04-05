load donnees;
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
