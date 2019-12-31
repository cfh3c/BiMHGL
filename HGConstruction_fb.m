
function [IniY H nTest gt mPara] = HGConstruction_fb(mPara,mTrainTestSplitBig)
%% this function is to construct the hypergraph
% input:    mPara: the parameter
%           mTrainTestSplitBig : the splited index of the TrainTest samples
% output: results4test: the classification results, 1 or 0
%% getTTSmallData is to get the training and testing data from the big split data pool.
% select the total training samples and the testing samples
iFoldCV = mPara.iPerm;
tmpPosTrain = mTrainTestSplitBig{iFoldCV,1};
tmpPosTest = mTrainTestSplitBig{iFoldCV,2};
tmpNegTrain = mTrainTestSplitBig{iFoldCV,3};
tmpNegTest = mTrainTestSplitBig{iFoldCV,4};

nPosTrain = length(tmpPosTrain); % number of postive training samples
nNegTrain = length(tmpNegTrain); % number of negative training samples
nPosTest = length(tmpPosTest); % number of postive testing samples
nNegTest = length(tmpNegTest); % number of negative testing samples

nTrain = nPosTrain + nNegTrain;
nTest = nPosTest + nNegTest;
nAll = nTrain + nTest;

gt = zeros(nTest,1);
gt(1:nPosTest,1) = 1;
gt(nPosTest+1:nTest,1) = 2;

tmpAllData = [tmpPosTrain;(tmpNegTrain+mPara.nAllPos);tmpPosTest;(tmpNegTest+mPara.nAllPos)];
mPara.TrueList = tmpAllData;

%mSubW = diag([ones(1,nPosTrain)*mPara.mPosW ones(1,nNegTrain)*mPara.mNegW ones(1,nTest)/nAll]);
%mPara.mSubW = mSubW;


mDimModel = mPara.mDimModel;
nEdge = mDimModel(1,1) + mDimModel(2,1) + mDimModel(3,1);
% IniY = zeros(nEdge,2);
% IniY(1:end,:) = mPara.mWordSentiment;
IniY = zeros(nAll,2);
IniY(1:nPosTrain,1) = 1;% the positive training samples are given 1 in the first column
IniY(nPosTrain+1:nTrain,1) = -1;% 
IniY(1:nPosTrain,2) = -1;% 
IniY(nPosTrain+1:nTrain,2) = 1;% the negative training samples are given 1 in the second column


mExpFea = mPara.mExpFea{mPara.iExp}; % the feature IDs used here
nExpFea = size(mExpFea,1); % the number of features used here

H = [];
for iFeature =1:nExpFea
    feaID = mExpFea(iFeature); % the used feature here
    H = [H mPara.mFeature{feaID,1}(tmpAllData,:)];
end


end