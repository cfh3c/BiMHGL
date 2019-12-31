%% learn on the hypergraph given the training samples and the testing samples

%% given: 1) the test samples and the training samples

clear all

%% 10-fold separation
% postive: 463
% negative: 133
% total: 596
% 10 cv
for iData = 1:20
    % divide positive into 10 pieces:  First 9: 5, the tenth: 11
    PosPosition = 1:463;
    NegPosition = 1:133;
    mSplit = cell(2,10);
    t1 = rand(1,463);
    [values orders1] = sort(t1);
    t2 = rand(1,133);
    [values orders2] = sort(t2);

    for i = 1:9
        mSplit{1,i} = PosPosition(orders1((i-1)*46+1:i*46));
        mSplit{2,i} = NegPosition(orders2((i-1)*13+1:i*13));
    end
    mSplit{1,10} = PosPosition(orders1(414+1:463));
    mSplit{2,10} = NegPosition(orders2(117+1:133));

    mTrainTestSplitBig = cell(10,4);%train pos, test pos, train neg, test neg

    %% each time, drop one pair of positive and negaive samples:
    for iTime = 1:10
        if iTime==10
            x=1;
        end
        iTime
        %nPosHere = 56-length(mSplit{1,iTime});
        %nNegHere = 69-length(mSplit{2,iTime});
        % which one is employed
        % for postive:
        iCount1 = 0;
        iCount2 = 0;
        %^clear mFold;
        for itmp = 1:10 % get the current Pos samples and Neg samples for training
            if itmp==iTime
            else
                tmpnPos = length(mSplit{1,itmp});
                tmpnNeg = length(mSplit{2,itmp});
                for j = 1:tmpnPos
                    iCount1 = iCount1 + 1;
                    mFold{1,iTime}(iCount1,1) = mSplit{1,itmp}(j); % this is the training ID
                end
                for j = 1:tmpnNeg
                    iCount2 = iCount2 + 1;
                    mFold{2,iTime}(iCount2,1) = mSplit{2,itmp}(j);% this is the testing ID
                end
            end
        end% end of i = 1:10 get the current Pos samples and Neg samples

        % get the big training/testing sample
        mTrainTestSplitBig{iTime,1} = mFold{1,iTime};
        mTrainTestSplitBig{iTime,2} = mSplit{1,iTime}';
        mTrainTestSplitBig{iTime,3} = mFold{2,iTime};
        mTrainTestSplitBig{iTime,4} = mSplit{2,iTime}';

        %% get the small training/testing sample from each indivisual group

           % for iTime= 1:9
           % nPosHere 51 : 5...5 6
           % nNegHere 62: 6...6 8
           % for iTime = 10     
           % nPosHere 46 : 5...5 1
           % nNegHere 63: 6...6 9

           % divide positive into 10 pieces:  First 9: 5, the tenth: 6
            mSubSplit = cell(2,10);
            if iTime == 10
                npHere = 463 - 49;
                nnHere = 133 - 16;
                nEachPosPiece = 83;
                nEachNegPiece = 24;
                lastP = 82; %83*4 + 82 = 463 - 49 = 414
                lastN = 21;
            else
                npHere = 463 - 46;
                nnHere = 133 - 13;
                nEachPosPiece = 83;
                nEachNegPiece = 24;
                lastP = 85; %83*4 + 85 = 463 - 46 = 417
                lastN = 24;
            end

            t1 = rand(1,npHere);
            [values orders1] = sort(t1);
            t2 = rand(1,nnHere);
            [values orders2] = sort(t2);

            for i = 1:4
                mSubSplit{1,i} = mFold{1,iTime}(orders1((i-1)*nEachPosPiece+1:i*nEachPosPiece));
                mSubSplit{2,i} = mFold{2,iTime}(orders2((i-1)*nEachNegPiece+1:i*nEachNegPiece));
            end
            mSubSplit{1,5} = mFold{1,iTime}(orders1(nEachPosPiece*4+1:(nEachPosPiece*4+lastP)));
            mSubSplit{2,5} = mFold{2,iTime}(orders2(nEachNegPiece*4+1:(nEachNegPiece*4+lastN)));

            for iSubTime = 1:5
                nSubPosHere = npHere - length(mSubSplit{1,iSubTime});
                nSubNegHere = nnHere - length(mSubSplit{2,iSubTime});
                % which one is employed
                % for postive:
                iSubCount1 = 0;
                iSubCount2 = 0;
                clear mSubFold;
                for iSub = 1:5 % get the current Pos samples and Neg samples
                    if iSub==iSubTime
                    else
                        tmpnSubPos = length(mSubSplit{1,iSub});
                        tmpnSubNeg = length(mSubSplit{2,iSub});
                        for j = 1:tmpnSubPos
                            iSubCount1 = iSubCount1 + 1;
                            mSubFold{1,1}(iSubCount1,1) = mSubSplit{1,iSub}(j);
                        end
                        for j = 1:tmpnSubNeg
                            iSubCount2 = iSubCount2 + 1;
                            mSubFold{2,1}(iSubCount2,1) = mSubSplit{2,iSub}(j);
                        end                                       
                    end

                end% end of i = 1:10 get the current Pos samples and Neg samples

                    % get the big training/testing sample
                    mTrainTestSplitSmall{iTime,iSubTime,1} = mSubFold{1,1};
                    mTrainTestSplitSmall{iTime,iSubTime,2} = mSubSplit{1,iSubTime}';
                    mTrainTestSplitSmall{iTime,iSubTime,3} = mSubFold{2,1};
                    mTrainTestSplitSmall{iTime,iSubTime,4} = mSubSplit{2,iSubTime}';                

            end %iSubTime


  
    end % iTime
    filename = ['datas/mTrainTestSplitBig' num2str(iData)];
    save(filename, 'mTrainTestSplitBig');
    filename = ['datas/mTrainTestSplitSmall' num2str(iData)];
    save(filename, 'mTrainTestSplitSmall');
end