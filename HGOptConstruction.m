function vH = HGOptConstruction(vdistM,mPara)
%% this function is to construct the hypergraph
% input: IniY: the label information
%           vdistM: the distance matrix
%           mPara: the parameter
%           mPara.mLamda(mPara.iLamda)
%           mPara.mStarExp(mPara.iStarExp
% output: results4test: the classification results, 1 or 0
 
IS_ProH = mPara.IS_ProH;
nObject = size(vdistM{1,1},1);

mExpFea = mPara.mExpFea{mPara.iExp}; % the feature IDs used here
nHG = size(mExpFea,1); % the number of features used here
    
nEdge = nObject;

%% hypergraph construction

for iFeature = 1:nHG
    tmpH =zeros(nObject,nEdge);%construct tmp H
    feaID = mExpFea(iFeature);% find the feature id for the iFeature th one
    mStarExp = mPara.optPara(feaID,1);%get the optimal mStarExp for feaID
    distM = vdistM{iFeature,1};
    aveDist = mean(mean(distM));
    for iObj = 1:nObject
        vDist = distM(iObj,:);
        [values orders] = sort(vDist,'ascend');
        for iLinked = 1:mStarExp
            if IS_ProH == 0 % if it is not pro H
                tmpH(orders(iLinked),iObj) = 1;
            else
                tmpH(orders(iLinked),iObj) = exp(-values(iLinked)^2/(0.1*aveDist)^2);
            end
        end % end of iLinked
    end % end of iObj
    vH{iFeature,1} = tmpH;
end