function     [Perf tmpBad] =  HGClassify(mPara,BestPara)
%% Perf is a 1 time 6 vector
mTrainTestSplitBig = mPara.mTrainTestSplitBig;

mPara.iProbSigmaWeight = BestPara(1,1);
mPara.iFea = BestPara(1,2);
mPara.iStarExp = BestPara(1,3);
mPara.iLamda = BestPara(1,4);
mPara.iMu = BestPara(1,5);

[IniY IniYT H HT nTest gt mPara] = BiHGConstruction_fb(mPara,mTrainTestSplitBig);
[relMatrix mPara] = BiHG_learning2(H,HT,IniY,IniYT,mPara);

[tmpPerf tmpBad] = evaluate(relMatrix,gt,nTest,mPara);

Perf = tmpPerf(:);
%}
end