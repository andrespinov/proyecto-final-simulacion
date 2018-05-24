function [ Xtrain, Ytrain, Xval, Yval ] = bootstrapping( X, Y, porcentaje )
% BOOSTRAPING realiza la partición del conjunto de datos en el de
% entrenamiento y validación repartiendo cada una de las clases
% equitativamente en cada conjunto.
%   Detailed explanation goes here
    
    % Separación de las clases
    [clase1, clase2, clase3, clase4] = separarClases(X,Y);
    % Generación de los conjuntos de entrenamiento y validación tomando las
    % proporciones que les corresponden de cada clase.
    
    for i = 1:4
        % Se obtiene la información de la clase
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
        % validación
        auxT = clasei(ind(1:particion), :);
        auxV = clasei(ind(particion+1:end), :);
        auyT = ones(particion,1);
        auyV = ones(N - particion, 1);
        if i == 1
            % Si es la primera iteración los conjuntos serán los
            % subconjuntos calculados anteriormente
            Xtrain = auxT;
            Xval = auxV;
            Ytrain = auyT;
            Yval = auyV;
        else
            % Las demás iteraciones son lo que tiene el conjunto más el
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

