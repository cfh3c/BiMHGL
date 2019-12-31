function tmpPerf = HGMultipleFeature(IniY,vdistM,mPara,nTest,gt)
%% this function is to conduct the hypergraph learning with multiple features
% input: IniY: the label information
%           vdistM: the distance matrix
%           mPara: the parameter
%           mPara.mLamda(mPara.iLamda)
%           mPara.mStarExp(mPara.iStarExp
% output: results4test: the classification results, 1 or 0
 
IS_ProH = mPara.IS_ProH;
nObject = size(vdistM{1,1},1);
mStarExp = mPara.mStarExp(mPara.iStarExp); % number of star expansion

mExpFea = mPara.mExpFea{mPara.iExp}; % the feature IDs used here
nExpFea = size(mExpFea,1); % the number of features used here
    
nEdge = nObject*nExpFea;

%% hypergraph construction
H =zeros(nObject,nEdge);

for iFeature = 1:nExpFea
    distM = vdistM{iFeature,1};
    aveDist = mean(mean(distM));
    for iObj = 1:nObject
        vDist = distM(iObj,:);
        [values orders] = sort(vDist,'ascend');
        for iLinked = 1:mStarExp
            if IS_ProH == 0 % if it is not pro H
                H(orders(iLinked),iObj+(iFeature-1)*nObject) = 1;
            else
                H(orders(iLinked),iObj+(iFeature-1)*nObject) = exp(-values(iLinked)^2/(0.1*aveDist)^2);
            end
        end % end of iLinked
    end % end of iObj
end

relMatrix = HG_learning(H,IniY,mPara);

allresults = zeros(nObject,1);
for iObj = 1:nObject
    if relMatrix(iObj, 1) > relMatrix(iObj, 2)
        allresults(iObj, 1) = 1;
    else
        allresults(iObj,1) = 2;
    end
end

results4test = allresults(nObject-nTest+1:nObject,1);

%% count the experimental results
%                  pos_detected   neg_detected
%   pos_gt              a                     b
%   neg_gt              c                     d
tmpPerf = zeros(2);
for iObj = 1:nTest
    tmpPerf(gt(iObj),results4test(iObj)) = tmpPerf(gt(iObj),results4test(iObj))+1;    
end