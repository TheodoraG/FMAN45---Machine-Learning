function results = SVM_classifier(y, labels)
    % class labels
    class_0 = labels(y == 0);
    class_1 = labels(y == 1);

    % evaluate the results
    results = zeros(1,4); 
    % correct results for class 0
    results(1,1) = sum(class_0 == 0);
    % wrong results for class 0
    results(1,2) = sum(class_0 == 1);
    % wrong results for class 1
    results(1,3) = sum(class_1 == 0);
    % correct results for class 1
    results(1,4) = sum(class_1 == 1);
end

