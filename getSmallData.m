function [IniY gt vdistM nTest mPara] = getSmallData(mPara,mTrainTestSplitSmall)
%% getTTSmallData is to get the training and testing data from the small split data pool.
%% The input information is the number of testing round, i.e., iSmallTest
%% The output is the initialized labeled information IniY, the groundtruth data for all the data, and the used distance matrix distM

%% first get the id for the 10*10 results
    i1 = mPara.iPerm;
    i2 = mPara.iSmallTest;                   

    % select the total training samples and the testing samples
    tmpPosTrain = mTrainTestSplitSmall{i1,i2,1};
    tmpPosTest = mTrainTestSplitSmall{i1,i2,2};
    tmpNegTrain = mTrainTestSplitSmall{i1,i2,3};
    tmpNegTest = mTrainTestSplitSmall{i1,i2,4};
    
    nPosTrain = length(tmpPosTrain); % number of postive training samples
    nNegTrain = length(tmpNegTrain); % number of negative training samples
    nPosTest = length(tmpPosTest); % number of postive testing samples
    nNegTest = length(tmpNegTest); % number of negative testing samples
    
    nTrain = nPosTrain + nNegTrain;
    nTest = nPosTest + nNegTest;
    nAll = nTrain + nTest;
    
    IniY = zeros(nAll,2);
    IniY(1:nPosTrain,1) = 1;% the positive training samples are given 1 in the first column
    IniY(nPosTrain+1:nTrain,1) = -1;% 
    IniY(1:nPosTrain,2) = -1;% 
    IniY(nPosTrain+1:nTrain,2) = 1;% the negative training samples are given 1 in the second column
  
    mSubW = diag([ones(1,nPosTrain)*mPara.mPosW ones(1,nNegTrain)*mPara.mNegW ones(1,nTest)/nTest]);
    mPara.mSubW = mSubW;
    
    %% gt for both training and testing
%     gt = zeros(nAll,1);
%     gt(1:nPosTrain,1) = 1;
%     gt(nPosTrain+1:nTrain,1) = 0;
%     gt(nTrain+1:nTrain+nPosTest,1) = 1;
%     gt(nTrain+nPosTest+1:nAll, 1) = 0;
    %% gt for testing only
    gt = zeros(nTest,1);
    gt(1:nPosTest,1) = 1;
    gt(nPosTest+1:nTest,1) = 2;
    
    tmpAllData = [tmpPosTrain;(tmpNegTrain+mPara.nAllPos);tmpPosTest';(tmpNegTest'+mPara.nAllPos)];
    
    mExpFea = mPara.mExpFea{mPara.iExp}; % the feature IDs used here
    nExpFea = size(mExpFea,1); % the number of features used here
    
    
    vdistM = cell(nExpFea,1);    
    for iFeature =1:nExpFea
        validList = find(mPara.mLogFeaFlag{iFeature,mPara.iFea}==1);% get the valid list for this feature
        invalidList = find(mPara.mLogFeaFlag{iFeature,mPara.iFea}==0);
    
        feaID = mExpFea(iFeature); % the used feature here
        distM = zeros(nAll);
        mCount = 0;
        for iImg1 = 1:nAll
             pos1 = tmpAllData(iImg1);
             if iImg1==87
                 k=1;
             end
             if find(validList==pos1)% it is valid
                 mCount = mCount+1;
                mPara.mValidList{iFeature,1}(mCount,1) = iImg1;
             end

             mPara.TrueList(iImg1,1) = pos1;
             tmpFea(iImg1,:) = mPara.mFeature{feaID,mPara.iFea}(pos1,:);% get all fea for this round
        end
        distM = EuDist2(tmpFea);
        clear tmpFea;
        vdistM{iFeature,1} = distM;
    end