%%learn part
%% including the multiple running with small cross-validation, big cross-validation, evaluation, parameter tunning.
%% By Yue Gao, 09/05/2014

%  First, load required data
%  Second, select the data from the data pool
%  Run the program
%  Evaluation and counting
%  Finally run the program on the big pool
clear all;0

load mDist;
nData = 50;
nFea = 1;
% parameters:
mPara.mStarExp = [8 10 15 20 30 40 50]%;[4 6 8 10 15 20 25 30];% star expansion number
mPara.mLamda  = [1e-2 1e-1 1e0 1e1 1e2 1e3 1e4]% 1e2 1e3 1e4 1e5 1e6]% [0.001 0.01 0.1 0.5 1 5 10 50 100 1000]%0.5 1 5 10];% parameter in the SSL
%mPara.mMu =   [0.001]% 0.01 0.1 0.5 1 5 10 50 100 1000]%0.5 1 5 10];;% 1e-3 1e-2 1e-1 1   [1e-5 1e-4 1e-3 1e-2 1e-1 10 ];%0.0001 0.001 0.01 0.1  1 10 100 1000];% parameter in the weight learning
mPara.mMu =   [1e-2 1e-1 1e0 1e1 1e2 1e3 1e4]% 0.001 0.01 0.1 0.5 1 5 10 50 100 1000]%0.5 1 5 10];;% 1e-3 1e-2 1e-1 1   [1e-5 1e-4 1e-3 1e-2 1e-1 10 ];%0.0001 0.001 0.01 0.1  1 10 100 1000];% parameter in the weight learning
mPara.IsWeight =1; % 0: do not learn weight   1: learn weight
mPara.mExpFea = {[1]; [2]; [3]; [4]; [1;2]; [1;3]; [1;4]; [2;3]; [2;4]; [3;4]; [1;2;3]; [1;2;4]; [1;3;4]; [2,3,4]; [1;2;3;4]};

mPara.optPara=[30 1000;50 100;15 0.1;30 1000];%for StarExp and Lambda

nStarExp = length(mPara.mStarExp);
nLamda = length(mPara.mLamda);
nMu = length(mPara.mMu);
nExp = length(mPara.mExpFea);

for iIter = 1:1
    mPara.nIter = iIter;% the number of iterations in the learning part
    for PH = 2:2
        mPara.IS_ProH = PH-1;% whether H is probabilitic or 1.
        for iData = 1:nData
            filename = ['mTrainTestSplitBig' num2str(iData)];
            load(filename);
            filename = ['mTrainTestSplitSmall' num2str(iData)];
            load(filename);
            nAllPos = 56;                     nAllNeg = 69;
            mPara.nAllPos = nAllPos;          mPara.nAllNeg = nAllNeg;

            %% make data selection, parameter settings
            % we have different settings: iFea (1-2), iStarExp,  iLamda, iExp (1-5)( using which feature(s)) 
            % and iBigTest (1-10) or iSmallTest (1-100)(to be counted)

            for iFea = 1:nFea %iFea 1: bu   iFea 2: wu
                mPara.iFea = iFea;
                for iStarExp = 1:nStarExp % the star expansion in the hypergraph construction
                    mPara.iStarExp = iStarExp;
                    for iLamda = 1:nLamda% the parameter in the SSL
                        mPara.iLamda = iLamda;
                        %====================
                       for iMu = 1:nMu 
                           mPara.iMu = iMu;
                           for iExp = 1:nExp% iExp: 1-4: the four features, 5:all four features
                                mPara.iExp = iExp; 
                                xPerm = zeros(2);% the performance counting for all 100 small tests
                                for iBigTest = 1:10% select the 10 Big testing datasets
                                    mPara.iBigTest = iBigTest; 
                                    %% the 10 big test is conducted in this loop                
                                    %% get the  training data and the testing data
                                    [IniY gt vdistM nTest] = getTTBigData(mDist,mPara,mTrainTestSplitBig);% get the required data for the current learning procedure
                                    vH = HGOptConstruction(vdistM,mPara);
                                    relMatrix = MHG_learning(vH,IniY,mPara);
                                    tmpPerf = evaluate(relMatrix,gt,nTest);
                                    xPerm = xPerm + tmpPerf; 
                                end% end for the iBigTest                
                                %% get the overall results for all 100 small tests
                                %% count the experimental results
                                    %                  pos_detected   neg_detected
                                    %   pos_gt              a                     b
                                    %   neg_gt              c                     d
                                acc = (xPerm(1,1)+xPerm(2,2))/sum(sum(xPerm));%accuracy: the percentage of all correctedly classified results
                                sen = xPerm(1,1)/sum(xPerm(1,:));%sensitivity:  the percentage of correctedly classified positive samples in all positive samples
                                spec = xPerm(2,2)/sum(xPerm(2,:)); %Specificity:  the percentage of correctedly classified negative samples in all negative samples

                                mBigResults_acc{PH,1}{iData,iFea}{iExp,iStarExp}(iLamda,iMu) = acc;
                                mBigResults_sen{PH,1}{iData,iFea}{iExp,iStarExp}(iLamda,iMu) = sen;
                                mBigResults_spec{PH,1}{iData,iFea}{iExp,iStarExp}(iLamda,iMu) = spec;     
                                ['iIter = ' num2str(iIter) 'PH = ' num2str(PH-1) ' iData = ' num2str(iData)  ' BIG iFea = ' num2str(iFea) ' iStarExp = ' num2str(iStarExp)  ' iLamda = ' num2str(iLamda) ' iMu = ' num2str(iMu) ' iExp = ' num2str(iExp) 'acc = ' num2str(acc) ' sen = ' num2str(sen) ' spec = ' num2str(spec)]
                           end

                       end% end of iMu
                    end% end of iLamda
                end % end of iStarExp
            end % end of iFea

            %% find the max results for big test
            for iExp = 1:nExp
                for iStarExp = 1:nStarExp
                    for iFea = 1:nFea
                        Bigacc{PH,1}{iData,iFea}(iExp,iStarExp) = max(max(mBigResults_acc{PH,1}{iData,iFea}{iExp,iStarExp}));
                        Bigsen{PH,1}{iData,iFea}(iExp,iStarExp) = max(max(mBigResults_sen{PH,1}{iData,iFea}{iExp,iStarExp}));
                        Bigspec{PH,1}{iData,iFea}(iExp,iStarExp) = max(max(mBigResults_spec{PH,1}{iData,iFea}{iExp,iStarExp}));
                    end
                end
            end



            if 1>2
                for iFea = 1:nFea%2 %iFea 1: bu   iFea 2: wu
                    mPara.iFea = iFea;
                    for iStarExp = 1:nStarExp % the star expansion in the hypergraph construction
                        mPara.iStarExp = iStarExp;
                        for iLamda = 1:nLamda% the parameter in the SSL
                            mPara.iLamda = iLamda;
                            %====================
                               for iMu = 1:nMu 
                                   mPara.iMu = iMu;
                                %====================
                                for iExp = 1:4% iExp: 1-4: the four features, 5:all four features

                                    mPara.iExp = iExp; 
                                    xPerm = zeros(2);% the performance counting for all 100 small tests
                                    for iSmallTest = 1:100% select the 100 small testing dataset
                                        mPara.iSmallTest = iSmallTest;
                                        %% the 100 small test is conducted in this loop

                                        %% get the  training data and the testing data
                                        [IniY gt distM nTest] = getTTSingleSmallData(mDist,mPara,mTrainTestSplitSmall);% get the required data for the current learning procedure
                                        tmpPerf = runHGSingleFeature_w(IniY,distM,mPara,nTest,gt);% conduct hypergraph learning. predictM is the generated relevance from each samples to the two classes                    
                                        xPerm = xPerm + tmpPerf;
                                    end% end for the iSmallTest                
                                    %% get the overall results for all 100 small tests
                                    %% count the experimental results
                                        %                  pos_detected   neg_detected
                                        %   pos_gt              a                     b
                                        %   neg_gt              c                     d
                                    acc = (xPerm(1,1)+xPerm(2,2))/sum(sum(xPerm));%accuracy: the percentage of all correctedly classified results
                                    sen = xPerm(1,1)/sum(xPerm(1,:));%sensitivity:  the percentage of correctedly classified positive samples in all positive samples
                                    spec = xPerm(2,2)/sum(xPerm(2,:)); %Specificity:  the percentage of correctedly classified negative samples in all negative samples
                                    mSmallResults_acc{iFea,iExp}(iStarExp,iLamda) = acc;
                                    mSmallResults_sen{iFea,iExp}(iStarExp,iLamda) = sen;
                                    mSmallResults_spec{iFea,iExp}(iStarExp,iLamda) = spec;                
                                    ['SMALL iFea = ' num2str(iFea) ' iStarExp = ' num2str(iStarExp)  ' iLamda = ' num2str(iLamda) ' iExp = ' num2str(iExp) 'acc = ' num2str(acc) ' sen = ' num2str(sen) ' spec = ' num2str(spec)]
                                end% end of for iExp = 1:4

                                for iExp = 5:5% iExp: 1-4: the four features, 5:all four features
                                    mPara.iExp = iExp; 
                                    xPerm = zeros(2);% the performance counting for all 100 small tests
                                    for iSmallTest = 1:100% select the 100 small testing dataset
                                        mPara.iSmallTest = iSmallTest;
                                        %% the 100 small test is conducted in this loop                
                                        %% get the  training data and the testing data
                                        [IniY gt vdistM nTest] = getTTMultipleSmallData(mDist,mPara,mTrainTestSplitSmall);% get the required data for the current learning procedure
                                        tmpPerf = runHGMultipleFeature_w(IniY,vdistM,mPara,nTest,gt);% conduct hypergraph learning. predictM is the generated relevance from each samples to the two classes                    
                                        xPerm = xPerm + tmpPerf;
                                    end% end for the iSmallTest                


                                    acc = (xPerm(1,1)+xPerm(2,2))/sum(sum(xPerm));%accuracy: the percentage of all correctedly classified results
                                    sen = xPerm(1,1)/sum(xPerm(1,:));%sensitivity:  the percentage of correctedly classified positive samples in all positive samples
                                    spec = xPerm(2,2)/sum(xPerm(2,:)); %Specificity:  the percentage of correctedly classified negative samples in all negative samples
                                   if iFea == 1
                                        mBigResults3_acc{iExp,iStarExp}(iLamda,iMu) = acc;
                                        mBigResults3_sen{iExp,iStarExp}(iLamda,iMu) = sen;
                                        mBigResults3_spec{iExp,iStarExp}(iLamda,iMu) = spec;    
                                    else
                                        mBigResults4_acc{iExp,iStarExp}(iLamda,iMu) = acc;
                                        mBigResults4_sen{iExp,iStarExp}(iLamda,iMu) = sen;
                                        mBigResults4_spec{iExp,iStarExp}(iLamda,iMu) = spec;    
                                    end
                                    ['BIG iFea = ' num2str(iFea) ' iStarExp = ' num2str(iStarExp)  ' iLamda = ' num2str(iLamda) ' iMu = ' num2str(iMu) ' iExp = ' num2str(iExp) 'acc = ' num2str(acc) ' sen = ' num2str(sen) ' spec = ' num2str(spec)]
                                end% end of for iExp = 5:5
                                %====================
                               end%end of iMu

                        end% end of iLamda
                    end % end of iStarExp
                end % end of iFea


                %% find the max results for small test
                for iExp = 1:5
                    for iStarExp = 1:nStarExp
                        Big3acc(iExp,iStarExp) = max(max(mBigResults3_acc{iExp,iStarExp}));
                        Big3sen(iExp,iStarExp) = max(max(mBigResults3_sen{iExp,iStarExp}));
                        Big3spec(iExp,iStarExp) = max(max(mBigResults3_spec{iExp,iStarExp}));
                    end
                end

                save Bigacc Bigacc;
                save Bigsen Bigsen;
                save Bigspec Bigspec;
                save Smallacc Smallacc;
                save Smallsen Smallsen;
                save Smallspec Smallspec;
                save mBigResults_acc mBigResults_acc;
                save mBigResults_sen mBigResults_sen;
                save mBigResults_spec mBigResults_spec;
                save mSmallResults_acc mSmallResults_acc;
                save mSmallResults_sen mSmallResults_sen;
                save mSmallResults_spec mSmallResults_spec;

            end
        end
            %% calculate the mean, max, variance of each value

        %% find the max results for big test
        for iFea = 1:nFea
            for iExp = 1:nExp
                for iStarExp = 1:nStarExp
                    tacc =0;       tsen=0;         tspec=0;
                    tmpD = zeros(3,nData);
                   for iData = 1:nData
                       tmpD(1,iData) = Bigacc{PH,1}{iData,iFea}(iExp,iStarExp);
                       tmpD(2,iData) = Bigsen{PH,1}{iData,iFea}(iExp,iStarExp);
                       tmpD(3,iData) = Bigspec{PH,1}{iData,iFea}(iExp,iStarExp);
                   end
                   meanBigacc{iIter,1}{PH,iFea}(iExp,iStarExp) = mean(tmpD(1,:));
                   meanBigsen{iIter,1}{PH,iFea}(iExp,iStarExp) = mean(tmpD(2,:));
                   meanBigspec{iIter,1}{PH,iFea}(iExp,iStarExp) = mean(tmpD(3,:));
                   maxBigacc{iIter,1}{PH,iFea}(iExp,iStarExp) = max(tmpD(1,:));
                   maxBigsen{iIter,1}{PH,iFea}(iExp,iStarExp) = max(tmpD(2,:));         
                   maxBigspec{iIter,1}{PH,iFea}(iExp,iStarExp) = max(tmpD(3,:));
                   varBigacc{iIter,1}{PH,iFea}(iExp,iStarExp) = var(tmpD(1,:));
                   varBigsen{iIter,1}{PH,iFea}(iExp,iStarExp) = var(tmpD(2,:));         
                   varBigspec{iIter,1}{PH,iFea}(iExp,iStarExp) = var(tmpD(3,:));          
                end
            end
        end
        
      label = [1 2 3 4 124 1234];

      
       for iBox = 1:nStarExp
          tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+1,1+iBox) = mPara.mStarExp(iBox);
          tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+1,nStarExp+3+iBox) = mPara.mStarExp(iBox);
          tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+1,nStarExp*2+5+iBox) = mPara.mStarExp(iBox);                  
       end
       
       for iBox = 1:nExp
          nHere =length(mPara.mExpFea{iBox});
          v = 0 ;
          for iv = 1:nHere
             v= v+ mPara.mExpFea{iBox}(iv)*10^iv;
          end
          tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+iBox+1,1) = v;
          tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+iBox+1,nStarExp+3) = v;
          tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+iBox+1,nStarExp*2+5) = v;
       
       end
       tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+2:(iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+nExp+1,2:nStarExp+1) = meanBigacc{iIter,1}{PH,iFea};
       tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+2:(iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+nExp+1,nStarExp+4:nStarExp*2+3) = maxBigacc{iIter,1}{PH,iFea};
       tmpM((iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+2:(iIter-1)*((nExp*2+6))+(PH-1)*(nExp+2)+nExp+1,nStarExp*2+6:nStarExp*3+5) = varBigacc{iIter,1}{PH,iFea};                              

    end
end

mPerf.meanBigacc = meanBigacc;
mPerf.meanBigsen = meanBigsen;
mPerf.meanBigspec = meanBigspec;
mPerf.maxBigacc = maxBigacc;
mPerf.maxBigsen = maxBigsen;
mPerf.maxBigspec = maxBigspec;
mPerf.varBigacc = varBigacc;
mPerf.varBigsen = varBigsen;
mPerf.varBigspec = varBigspec;

save mPerf mPerf;




% for PH=1:2
%     for iFea = 1:nFea
%             for iExp = 1:nExp
%                 for iStarExp = 1:nStarExp
%                     for iLamda = 1:8
%                         t = 0 ;
%                         for iData = 1:50
%                             t= mBigResults_acc{PH,1}{iData,iFea}{iExp,iStarExp}(iLamda,1)+t;
%                         end
%                         t=t/50;
%                         perfLamda{PH,iFea}{iExp,iStarExp}(iLamda,1) =t;
%                     end
%                 end
%             end
%         end
% end