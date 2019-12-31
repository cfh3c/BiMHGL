%%learn part
%% including the multiple running with small cross-validation, big cross-validation, evaluation, parameter tunning.

%  First, load required data
%  Second, select the data from the data pool
%  Run the program
%  Evaluation and counting
%  Finally run the program on the big pool
%clear all;


%% parameters (Please see the paper for the definitions):
mPara.IsWeight = 1; % 0: do not learn weight   1: learn weight
mPara.mProbSigmaWeight = [0.3]
mPara.mFea = [1];
mPara.mStarExp = [6]
mPara.mLamda  = [1e2]
mPara.mMu = [1e1]
mPara.mExpFea = {[1;2;3]} % three modalities
mIndexExpFea = [123];
 
mPara.Alpha = [1e0];
mPara.mLamda2 = [1e2];
mPara.mMu2 = [3e4];

mPara.mCostMatrix = [1 1;1 1];

 
mPara.nAllPos = 4196;          mPara.nAllNeg = 1354; % the sample sizes
mDimModel = [2547;49;1553]; % the dictionary sizes of three modalities
mPara.mDimModel = mDimModel;

mPara.nIter = 10;
mPara.nIterL1 = 10;
mPara.nIterL2 = 10;
mPara.IS_ProH = 0;

mPara.nExpFea = length(mPara.mExpFea);
mPara.nData = 1;%10;%10;
mPara.nBigFold = 10; % the big cv fold number
mPara.nSmallFold = 5; % the small (inner cv fold number)

load ../0Datas/20150710/mImageTextFeaSel.mat;
mPara.mFeature = mImageTextFeaSel;
clear mImageTextFeaSel;
% load datas/mLogFeaFlag.mat;
% mPara.mLogFeaFlag = mLogFeaFlag;
% clear mLogFeaFlag;
load datas/mWordSentimentCell_2547_49_1553.mat;
mPara.mWordSentiment = mWordSentimentCell;
clear mWordSentiment;

mPara.tmpCount  = 0;
%timeCount = zeros(nExp,nData);


mPara.iProbSigmaWeight = 1;
mPara.iFea = 1;
mPara.iStarExp = 1;
mPara.iLamda = 1;
mPara.iMu = 1;
mPara.iAlpha = 1;
mPara.iLamda2 = 1;
mPara.iMu2 = 1;

nProbSigmaWeight = length(mPara.mProbSigmaWeight);
nFea = length(mPara.mFea);
nStarExp = length(mPara.mStarExp);
nLamda = length(mPara.mLamda);
nMu = length(mPara.mMu);
nExp = length(mPara.mExpFea);
nMu2 = length(mPara.mMu2);

outputFile = 'results/record_f.txt';
fp = fopen(outputFile,'a+');
fprintf(fp,'##########################################################\n');
fprintf(fp,'\t Bi-level hypergraph learning\n');
fprintf(fp,'##########################################################\n\n');
fclose(fp);

for iRate = 1:1
    if iRate ==1
                mPara.mPosW = 1/1; mPara.mNegW = 1/1; 
                mPara.iRate = 1;
    elseif iRate == 2
                mPara.mPosW = 1/1; mPara.mNegW = 1/1; 
                mPara.iRate = 2;
    elseif iRate == 3
                mPara.mPosW = 1; mPara.mNegW = 1; 
                mPara.iRate = 3;
    end
    for iProbSigmaWeight = 1:nProbSigmaWeight
        mPara.iProbSigmaWeight = iProbSigmaWeight;
        for iFea = 1:nFea
            mPara.iFea = iFea;
            for iStarExp = 1:nStarExp % the star expansion in the hypergraph construction
                mPara.iStarExp = iStarExp;
                for iLamda = 1:nLamda
                    mPara.iLamda = iLamda;
                    for iMu = 1:nMu 
                        mPara.iMu = iMu;
                    for iMu2 = 1:nMu2 
                        mPara.iMu2 = iMu2;
                    
                        fp = fopen(outputFile,'a+');
                        fprintf(fp,'******************* [ %d %d %d %d %d %d %d %d] **********************\n',...
                            mPara.iProbSigmaWeight,mPara.iFea,mPara.iStarExp,mPara.iLamda,mPara.iMu,mPara.iAlpha,mPara.iLamda2,mPara.iMu2);
                        fprintf(fp,'**********\nProbSigmaWeight = %0.9f\nFea = %0.9f\nStarExp = %0.9f\nLamda = %0.9f\nMu = %0.9f\nAlpha = %0.9f\nLamda2 = %0.9f\nMu2 = %0.9f\n**********\n',...
                            (mPara.mProbSigmaWeight(mPara.iProbSigmaWeight) ),(mPara.mFea(mPara.iFea)),(mPara.mStarExp(mPara.iStarExp)),(mPara.mLamda(mPara.iLamda)),...
                            (mPara.mMu(mPara.iMu)),(mPara.Alpha(mPara.iAlpha)),(mPara.mLamda2(mPara.iLamda2)),(mPara.mMu2(mPara.iMu2)));
                        %% first level: the used feature
                        for iExp = 1:mPara.nExpFea
                            fprintf(fp,'============ %d -th Conbination ============\n',mIndexExpFea(iExp));
                            mPara.iExp = iExp;
                            %% second level: the 10 data
                            for iData = 1:10%mPara.nData %#####################################################################
                                fprintf('## %d-th group\n',iData);
                                mPara.iData = iData;
                                filename = ['../0Datas/Splits/mTrainTestSplitBig' num2str(iData)];
                                load(filename);
                                mPara.mTrainTestSplitBig = mTrainTestSplitBig; 
                                filename = ['datas/mTrainTestSplitSmall' num2str(iData)];
                                load(filename);
                                mPara.mTrainTestSplitSmall = mTrainTestSplitSmall;

                                %% the thrid level: 10 cross validation
                                tmpPerf = zeros(10,4);
                                for iPerm = 1:mPara.nBigFold 
                                    fprintf(' # %d-th fold\n',iPerm);
                                    mPara.iPerm = iPerm; 
                                    mPara.iBigTest = iPerm;
                                    %% 1 get the small 10 cv for parameter selection
                                    %% 2 select the  testing data and training data 
                                    %% 3 use the selected parameter for the iPerm-th test
                                    % small cv
                                    % first calculate the overall performance for each
                                    % parameter setting, and then select the best one 
                                    BestPara = zeros(1,5);        

                                    BestPara(1,1) = mPara.iProbSigmaWeight;
                                    BestPara(1,2) = mPara.iFea;
                                    BestPara(1,3) = mPara.iStarExp;
                                    BestPara(1,4) = mPara.iLamda;
                                    BestPara(1,5) = mPara.iMu;

                                    [tmpPerf(iPerm,:) tmpBad] = HGClassify(mPara,BestPara);

                                    mBad{iExp,iData}{iPerm,1} = tmpBad;
                                    tmpacc = (tmpPerf(iPerm,1)+tmpPerf(iPerm,4))/sum(tmpPerf(iPerm,:));
                                    tmpsen = tmpPerf(iPerm,1)/(tmpPerf(iPerm,1)+tmpPerf(iPerm,3));
                                    tmpspec = tmpPerf(iPerm,4)/(tmpPerf(iPerm,2)+tmpPerf(iPerm,4));
                                    tmpbac = 0.5*(tmpsen+tmpspec); 
                                    tmpppv = tmpPerf(iPerm,1)/(tmpPerf(iPerm,1)+tmpPerf(iPerm,2));
                                    tmpnpv =  tmpPerf(iPerm,4)/(tmpPerf(iPerm,3)+tmpPerf(iPerm,4));

                                   ['iExp=' num2str(iExp) ' iData=' num2str(iData) ' iPerm=' num2str(iPerm) ' acc = ' num2str(tmpacc) ' sen = ' num2str(tmpsen) ' spec = ' num2str(tmpspec)  ' bac = ' num2str(tmpbac)  ' ppv = ' num2str(tmpppv)  ' npv = ' num2str(tmpnpv) ]
                                    mBad{iExp,iData}{iPerm,1}
                                    vBestPara{iExp,iData}(iPerm,:) = BestPara';
                                    mACCAll{iExp,1}(iData,iPerm) = tmpacc;
                                    %}
                                end
                                %
                                sumTmpPerf = sum(tmpPerf);

                                mASS{iExp,1}(iData,1) = (sumTmpPerf(1,1)+sumTmpPerf(1,4))/sum(sumTmpPerf);
                                mASS{iExp,1}(iData,2) = sumTmpPerf(1,1)/(sumTmpPerf(1,1)+sumTmpPerf(1,3));
                                mASS{iExp,1}(iData,3) = sumTmpPerf(1,4)/(sumTmpPerf(1,2)+sumTmpPerf(1,4));
                                mASS{iExp,1}(iData,4)  = 0.5*(mASS{iExp,1}(iData,2) +mASS{iExp,1}(iData,3) ); 
                                mASS{iExp,1}(iData,5)  = sumTmpPerf(1,1)/(sumTmpPerf(1,1)+sumTmpPerf(1,2));
                                mASS{iExp,1}(iData,6)  =  sumTmpPerf(1,4)/(sumTmpPerf(1,3)+sumTmpPerf(1,4));
                                ['iExp=' num2str(iExp) ' iData=' num2str(iData) ' the acc = ' num2str(mASS{iExp,1}(iData,1)) ' the sen = ' num2str(mASS{iExp,1}(iData,2)) ' spec = ' num2str(mASS{iExp,1}(iData,3)) ' bac = ' num2str(mASS{iExp,1}(iData,4)) ' ppv = ' num2str(mASS{iExp,1}(iData,5)) ' nvp = ' num2str(mASS{iExp,1}(iData,6))] 
                                fprintf(fp,'%0.5f %0.5f %0.5f %0.5f %0.5f %0.5f\n',mASS{iExp,1}(iData,1),mASS{iExp,1}(iData,2),mASS{iExp,1}(iData,3),mASS{iExp,1}(iData,4),mASS{iExp,1}(iData,5),mASS{iExp,1}(iData,6));

                                mBadUnion{iExp, iData}{1,1} = mBad{iExp,iData}{iPerm,1}(1,:);
                                mBadUnion{iExp, iData}{2,1} = mBad{iExp,iData}{iPerm,1}(2,:);                
                                for iPerm = 2:mPara.nBigFold 
                                    mBadUnion{iExp, iData}{1,1} = [mBadUnion{iExp, iData}{1,1}, mBad{iExp,iData}{iPerm,1}(1,:)];
                                    mBadUnion{iExp, iData}{2,1} = [mBadUnion{iExp, iData}{2,1}, mBad{iExp,iData}{iPerm,1}(2,:)];                   
                                end

                                %}
                            end %end of iData
                            %
                            [mMeanASS(iExp,:) mStdASS(iExp,:)] = getMeanStd(mASS{iExp,1});

                            fprintf(fp,'\n');
                            fprintf(fp,'%0.5f %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f \n',...
                                mMeanASS(iExp,:), mStdASS(iExp,:));
                            fprintf(fp,'=============================\n');
                            %}
                        end% end of iExpFea
                        fprintf(fp,'**********************************************************************************\n');
                        fclose(fp);
                    end
                    end
                end
            end
        end
    end
    
    %{
    mResults{iRate,1}.mMeanASS = mMeanASS;
    mResults{iRate,1}.mStdASS = mStdASS;
    mResults{iRate,1}.mASS = mASS;
    mResults{iRate,1}.mACCAll = mACCAll;
    %}
end
%save datas/mResults.mat;