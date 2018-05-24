%Carga de las muestras de entrenamiento
load('training.csv');
X = training(:,1:28);
Y = training(:,29);
load('testing.csv');
X = [X;testing(:,1:28)];
Y = [Y;testing(:,29)];

% Se generan los conjuntos de sobremuestreo de cada clase y se selecciona
% el 12% de muestras nuevas (por clase) para formar parte del nuevo
% conjunto de entrenamiento.
for i=1:4
    % Se obtienen las muestras pertenecientes únicamente a la clase de la
    % iteración
    classY = Y(Y == i,:);
    classX = X(Y == i,:);
    
    if(i ~= 2) % La clase 2 es la mayoritaria (no necesita sobremuestreo).
       [W,] = SMOTE(classX, classY, i);
       Z = ones(size(W,1),1);
       Z = Z.*i;
       C = [W,Z];
       % Como SMOTE incluye las muestras originales, se deben tomar las
       % restantes (sólo las generadas por la función)
       C = C(size(classX,1) + 1:end,:); 
       ind = randperm(size(C,1));
       newClass = C(ind(1:ceil(0.12*size(classX))),:);
       % Se guarda el nuevo subconjunto de muestras de la clase i.
       filename = strcat('C', num2str(i), '-SMOTE.mat');
       save(filename, 'newClass');
    elseif(i == 2) % A la clase 2 se le hace submuestreo
       ind = randperm(size(classX,1));
       newClass = [classX, classY];
       newClass = newClass(ind(1:ceil(0.88*size(newClass))),:);
       filename = strcat('C', num2str(i), '-DSample.mat');
       save(filename, 'newClass');
    end
end

