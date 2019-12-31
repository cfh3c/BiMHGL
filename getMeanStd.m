    function [vMean vStd] = getMeanStd(results)
    nCr = size(results,2);
    for i = 1:nCr
        vMean(1,i) = mean(results(:,i));
        vStd(1,i) = std(results(:,i));
    end
