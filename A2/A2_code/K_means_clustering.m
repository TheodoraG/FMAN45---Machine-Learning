function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X);

intermax = 80;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;

for kiter = 1:intermax

    % Step 1: Assign to clusters
    y = step_assign_cluster(X, Cold);
    
    % Step 2: Assign new clusters
    [C, dist] = step_compute_mean(X, Cold, y, K);
        
    if dist < conv_tol
        disp('converged')
        disp(kiter)
        return
    end
    
    Cold = C;
    if kiter == intermax 
        disp('did not converge') 
    end
end

end

% computes the distance between a single example
% and the K centroids
function d = fxdist(X,C)
    K = size(C, 2);
    
    for i = 1:K
        % euclidean distance
        d(i,:) = sqrt(sum((C(:,i)-X).^2));
    end
end

% computes the distance between two cluster centroids
% choose the largest distance
function d = fcdist(C1,C2)
    % euclidean distance
    d = max(sqrt(sum((C1-C2).^2)));
end

% assign sample to its closest centroid
function y = step_assign_cluster(X, C)
    [~, y] = min(fxdist(X, C)); 
end

% update each centroid by computing the mean
% of all points aasigned to the centroid
function [C_new, dist_measure] = step_compute_mean(X, C, y, K)  
    for k = 1 :K
        C_new(:,k) = mean(X(:,y == k), 2);
        dist_measure(k) = fcdist(C_new(:,k),C(:,k));
    end
end
