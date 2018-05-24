function [ clase1, clase2, clase3, clase4 ] = separarClases( X,Y )
% SEPARARCLASES separa las muestras de cada clase en X suponiendo que
% contiene 4 clases nombradas en n�meros del 1 al 4.
%   Detailed explanation goes here
    % Generaci�n de los �ndices
    ind1 = Y == 1;
    ind2 = Y == 2;
    ind3 = Y == 3;
    ind4 = Y == 4;
    
    % Separaci�n de las clases
    clase1 = X(ind1,:);
    clase2 = X(ind2,:);
    clase3 = X(ind3,:);
    clase4 = X(ind4,:);

end

