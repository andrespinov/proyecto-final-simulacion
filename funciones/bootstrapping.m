function [ Xtrain, Ytrain, Xval, Yval ] = bootstrapping( X, Y, porcentaje )
% BOOSTRAPING realiza la partici�n del conjunto de datos en el de
% entrenamiento y validaci�n repartiendo cada una de las clases
% equitativamente en cada conjunto.
%   Detailed explanation goes here
    
    % Separaci�n de las clases
    [clase1, clase2, clase3, clase4] = separarClases(X,Y);
    % Generaci�n de los conjuntos de entrenamiento y validaci�n tomando las
    % proporciones que les corresponden de cada clase.
    
    for i = 1:4
        % Se obtiene la informaci�n de la clase
        if i == 1
            clasei = clase1;
        elseif i == 2
            clasei = clase2;
        elseif i == 3
            clasei = clase3;
        elseif i == 4
            clasei = clase4;
        end
        % Se definen las variables a utilizar
        N = size(clasei,1);
        ind = randperm(N);
        particion = ceil(N * porcentaje);
        % Se obitene el subconjunto de la clase para entrenamiento y
        % validaci�n
        auxT = clasei(ind(1:particion), :);
        auxV = clasei(ind(particion+1:end), :);
        auyT = ones(particion,1);
        auyV = ones(N - particion, 1);
        if i == 1
            % Si es la primera iteraci�n los conjuntos ser�n los
            % subconjuntos calculados anteriormente
            Xtrain = auxT;
            Xval = auxV;
            Ytrain = auyT;
            Yval = auyV;
        else
            % Las dem�s iteraciones son lo que tiene el conjunto m�s el
            % subconjunto generado anteriormente
            Xtrain = [Xtrain; auxT];
            Xval = [Xval; auxV];
            auyT = auyT.*i;
            auyV = auyV.*i;
            Ytrain = [Ytrain; auyT];
            Yval = [Yval; auyV];
        end
    end
end

