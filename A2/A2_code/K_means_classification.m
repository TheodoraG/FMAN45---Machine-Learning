function [miss_train, miss_test] = K_means_classification(y, train_labels, test_data, test_labels, C)
    % assign the label of the closest centroid
    [~, lbl_test] = min(fxdist(test_data, C));

    % compute number of misclassifications
    [miss_train] = classification(y, train_labels, C);
    [miss_test] = classification(lbl_test, test_labels, C);
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

function [results] = classification(y, labels, C)
    K = size(C, 2);  
    % K means classification results 
    results = zeros(K,4);
    
    for i = 1:K
        cluster = labels(y == i);
        % class 0
        results(i,1) = length(cluster) - sum(cluster);
        % class 1
        results(i,2) = sum(cluster);
        % assigned to class
        results(i,3) = mode(cluster);
        
        % misclassified
        if results(i,3) == 0
            results(i,4) = results(i,2);
        else
            results(i,4) = results(i,1);
        end
    end
end