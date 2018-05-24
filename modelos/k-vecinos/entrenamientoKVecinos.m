function [ Yesti ] = entrenamientoKVecinos( X, Xent, Yent, k )
%ENTRENAMIENTOKVECINOS Summary of this function goes here
%   Detailed explanation goes here
    % Se definen las constantes de los tamaños
    N = size(Xent,1);
    M = size(X,1);
    
    Yesti = zeros(M,1);
    for j=1:M
    	dis = sqrt(sum((repmat(X(j,:),N,1) - Xent).^2,2));
        [~,index] = sort(dis,'ascend');
        knnindex = index(1:k);
        Yesti(j,1) = mode(Yent(knnindex,:));
    end
end
