function [f mPara] = BiHG_learning2(H,HT,IniY,IniYT,mPara)
%% conduct hypergraph learning in H

[nObject nEdge] = size(H);
lamda = mPara.mLamda(mPara.iLamda); % the parameter in SSL, i.e., lamda
mu = mPara.mMu(mPara.iMu);
IsWeight = mPara.IsWeight;

[nObjectT nEdgeT] = size(HT);
alpha = mPara.Alpha(mPara.iAlpha);
lamda2 = mPara.mLamda2(mPara.iLamda2);
mu2 = mPara.mMu2(mPara.iMu2);

W = eye(nEdge)/nEdge; 
F = eye(nEdgeT)/nEdgeT; 
%mSubW = mPara.mSubW; % the subjet weights to each subjects (it is based on the number of positive samples, negative samples and test samples.
%H = H.*repmat(diag(mSubW),1,nEdge);
%innerH = H.*repmat(diag(mSubW),1,nEdge);
innerH = H;
innerHT = HT;


%% learning on the hypergraph
if IsWeight == 0 % if no weight learnign is required
    %% DV DE INVDE calculation
    DV = eye(nObject);
    for iObject = 1:nObject
       DV(iObject,iObject) = sum(H(iObject,:).*diag(W)'); 
    end

    DE = eye(nEdge);
    for iEdge = 1:nEdge
        DE(iEdge,iEdge)=sum(H(:,iEdge));
    end

    DV2 = DV^(-0.5);
    INVDE = inv(DE);

    T = DV2*H*INVDE*H'*DV2;
    eta = 1/(1+lamda);
    L2 = eye(nObject) - eta*T;
    f = inv(L2) * IniY;
elseif IsWeight == 1
    nIter = mPara.nIter;
    nIterL1 = mPara.nIterL1;
    nIterL2 = mPara.nIterL2;
    dif_obj = 0;           
    flag = 0;
    fRecord{1,1} = IniY;   
    fRecord{2,1} = IniY;    
    fRecord{3,1} = IniY;
    tmpW3 = W;
    tmpW2 = W;
    tmpW1 = W;
    count = 0; % count >1, fail.  count =0 or 1: continue
    count0 = 0; % count0 >=3, end.  else, continue
%    mParaW = mSubW;
    mParaW = eye(nObject);
    mParaF = eye(nObjectT);
    for iIter = 1:nIter+1
        
        count1 = 0;
        count01 = 0;
        dif_obj1 = 0;
        %% update 1st-level HG
        for iIterL1 = 1:nIterL1+1
            %% update f
            DV = eye(nObject);
            for iObject = 1:nObject
              DV(iObject,iObject) = sum(H(iObject,:).*diag(W)'); 
            end
            DVtemp = diag(DV);
            DVtemp(find(DVtemp==0))=1e-10;
            DV = diag(DVtemp);
            DE = eye(nEdge);
            for iEdge = 1:nEdge
               %DE(iEdge,iEdge)=sum(H(:,iEdge).*diag(F)); 
               DE(iEdge,iEdge)=sum(H(:,iEdge)); 
            end
            DEtemp = diag(DE);
            DEtemp(find(DEtemp==0))=1e-10;
            DE = diag(DEtemp);

            DV2 = DV^(-0.5);
            INVDE = inv(DE);

            if mPara.iRate == 2%% 2: using the sample rate    1: no rate
                Theta = DV2*innerH*W*INVDE*innerH'*DV2;
                mParaW = mSubW;%diag(sum(innerH,2).*diag(mSubW));
            elseif mPara.iRate == 1
                Theta = DV2*H*W*INVDE*H'*DV2;
            end

            eta = 1/(1+lamda);
            L2 = mParaW-eta*Theta;
            f = (lamda/(1+lamda))*inv(L2) * mParaW*IniY.*repmat(diag(F),1,2);
            %tmpF{iIter,1} = f; % need change

            %% calculate the objective function
            tmp_sum_w1 = 0;
            tmp_sum_w2 = 0;
            for iEdge  = 1:nEdge
               tmp_sum_w1 = tmp_sum_w1 + W(iEdge,iEdge);
               tmp_sum_w2 = tmp_sum_w2 + W(iEdge,iEdge)*W(iEdge,iEdge);
            end
            laplacian1 = f(:,1)' * (mParaW - Theta) * f(:,1) + f(:,2)' * (mParaW - Theta) * f(:,2);
            IniYF1 = IniY(:,1).*diag(F);
            IniYF2 = IniY(:,2).*diag(F);
            exploss1 =  lamda * ((f(:,1) - IniYF1)'*(f(:,1) - IniYF1)+(f(:,2) - IniYF2)'*(f(:,2) - IniYF2));
            m_obj1(iIterL1,1) = laplacian1 + exploss1 + mu*tmp_sum_w2;
            %L(iIter,1) = laplacian;
            %L(iIter,2) = exploss;
            %L(iIter,3) = laplacian+exploss;
            if iIterL1 > 1                   
               dif_obj1 = m_obj1(iIterL1,1) - m_obj1(iIterL1-1,1);
               %objrecord(iIter,1) = dif_obj;
            end
        
            %% judge if to end the iteration
            [num2str(iIter) '    level_1    ' num2str(iIterL1) '     ' num2str(dif_obj1)]
            if dif_obj1 > 0
               count1 = count1+1;
            elseif dif_obj1 == 0
               count01 = count01+1;
            else
               count1 = 0;
               count01 = 0;
            end
            if (iIterL1 > nIterL1)&&(m_obj1(iIterL1,1) - m_obj1(1,1)<0)  || (count1 > 0) || count01>=3%((iIter > 1)&&(m_obj(iIter,1) - m_obj(iIter-1,1)< -1e-4))%||& (m_obj(iIter,1) - m_obj(1,1)<0)%%if dif is too small, stop iteration      
                %iIter
                %f = tmpF{iIter-1,1};
                break;
            end
            
            %% update w_hginnerH
            if mPara.iRate == 2%% 2: using the sample rate    1: no rate
                DV2H = DV2*innerH; % DV2H : Tau
                %INVDEHDV2 = INVDE*innerH'*DV2;
            elseif mPara.iRate == 1
                DV2H = DV2*H; % DV2H : Tau
                %INVDEHDV2 = INVDE*H'*DV2;               
            end

            %DV2HINVDEHDV2 = DV2H*W*INVDEHDV2;

            %obj4AllEdges = (f(:,1)'*DV2HINVDEHDV2*f(:,1) + f(:,2)'*DV2HINVDEHDV2*f(:,2));

            %watch = zeros(nEdge,1);
            %tmpWold = W;
            for iEdge = 1:nEdge
                tmp_left = DV2H(:,iEdge);
                %tmp_right = INVDEHDV2(iEdge,:);
                %obj4OneEdge(iEdge,1) = (f(:,1)'*(tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left')*f(:,1) + f(:,2)'*(tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left')*f(:,2));
                obj4OneEdge(iEdge,1) = (f(:,1)'*(tmp_left*INVDE(iEdge,iEdge)*tmp_left')*f(:,1) + f(:,2)'*(tmp_left*INVDE(iEdge,iEdge)*tmp_left')*f(:,2));
                %clear tmp_left tmp_right;
                clear tmp_left;
                %['obj = ' num2str(dif_obj) ' 1/nEdge = ' num2str(1/nEdge)  ' obj4AllEdges/(2*mu*nEdge) = ' num2str(obj4AllEdges/(2*mu*nEdge)) ' obj4OneEdge = '  num2str(i) '  ' num2str(obj4OneEdge)]
                %watch(iEdge,1) = obj4OneEdge(iEdge,1);watch(iEdge,2) = obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge;
                %watch(iEdge,3) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu); watch(iEdge,4) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
                %A_watch = [dif_obj 1/nEdge obj4AllEdges/(2*nEdge) obj4OneEdge(iEdge,1)/2];%                                                                       
            end

            obj4AllEdges = sum(obj4OneEdge);
            for iEdge = 1:nEdge                  
                tmpW(iEdge,iEdge) = 1/nEdge+(obj4OneEdge(iEdge,1)*nEdge - obj4AllEdges)/(2*mu*nEdge);%better-0  
                %watch(iEdge,5) = 1/nEdge+(obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
            end

            %in case some values are below 0
            minValue = min(diag(tmpW));
            if minValue > 0
                W = tmpW;
            else
                diagW = diag(tmpW);
                diagW = diagW - minValue + 1e-10;
                diagW = diagW/(1 - nEdge*minValue + nEdge*1e-10);
                W = diag(diagW);
            end

            %watch(1,6) = obj4AllEdges;
            tmpW3 = tmpW2; tmpW2 = tmpW1;  tmpW1 = W;
            clear DV2H INVDEHDV2 FTF L1 L2 L3 T;
            %Record_W{i_class,iteration}{i_para_lamda,1} = W;             
        end
        
        count2 = 0;
        count02 = 0;
        dif_obj2 = 0;
        %% update 2nd-level HG
        for iIterL2 = 1:nIterL2+1
            %% update g
            %F = diag(abs(f(:,1)));
            DVT = eye(nObjectT);
            for iObject = 1:nObjectT
              %DV(iObject,iObject) = sum(HT(iObject,:).*abs(f(:,1))'); 
              DV(iObject,iObject) = sum(HT(iObject,:).*diag(F)'); 
            end
            DVtempT = diag(DVT);
            DVtempT(find(DVtempT==0))=1e-10;
            DVT = diag(DVtempT);
            DET = eye(nEdgeT);
            for iEdge = 1:nEdgeT
               %DET(iEdge,iEdge)=sum(HT(:,iEdge).*diag(W)); 
               DET(iEdge,iEdge)=sum(HT(:,iEdge)); 
            end
            DEtempT = diag(DET);
            DEtempT(find(DEtempT==0))=1e-10;
            DET = diag(DEtempT);

            DVT2 = DVT^(-0.5);
            INVDET = inv(DET);

            if mPara.iRate == 2%% 2: using the sample rate    1: no rate
                ThetaT = DVT2*innerHT*F*INVDET*innerHT'*DVT2;
                mParaF = mSubWT;%diag(sum(innerH,2).*diag(mSubW));
            elseif mPara.iRate == 1
                ThetaT = DVT2*HT*F*INVDET*HT'*DVT2;
            end

            eta2 = alpha/(alpha+lamda2);
            L2T = mParaF-eta2*ThetaT;
            g = (lamda2/(alpha+lamda2))*inv(L2T) * mParaF*IniYT;
            %tmpG{iIter,1} = g;

            %% calculate the objective function in level 2
            tmp_sum_F1 = 0;
            tmp_sum_F2 = 0;
            for iEdge  = 1:nEdgeT
               tmp_sum_F1 = tmp_sum_F1 + F(iEdge,iEdge);
               tmp_sum_F2 = tmp_sum_F2 + F(iEdge,iEdge)*F(iEdge,iEdge);
            end
            laplacian2 = alpha * g(:,1)' * (mParaF - ThetaT) * g(:,1) + g(:,2)' * (mParaF - ThetaT) * g(:,2);
            exploss2 =  lamda2 * ((g(:,1) - IniYT(:,1))'*(g(:,1) - IniYT(:,1))+(g(:,2) - IniYT(:,2))'*(g(:,2) - IniYT(:,2)));
            m_obj2(iIterL2,1) = laplacian2 + exploss2 + mu2*tmp_sum_F2;
            if iIterL2 > 1                   
               dif_obj2 = m_obj2(iIterL2,1) - m_obj2(iIterL2-1,1);
               %objrecord(iIter,1) = dif_obj;
            end
            
            %% judge if to end the iteration
            [num2str(iIter) '    level_2    ' num2str(iIterL2) '     ' num2str(dif_obj2)]
            if dif_obj2 > 0
               count2 = count2+1;
            elseif dif_obj2 == 0
               count02 = count02+1;
            else
               count2 = 0;
               count02 = 0;
            end
            if (iIterL2 > nIterL2)&&(m_obj2(iIterL2,1) - m_obj2(1,1)<0)  || (count2 > 0) || count02>=3%((iIter > 1)&&(m_obj(iIter,1) - m_obj(iIter-1,1)< -1e-4))%||& (m_obj(iIter,1) - m_obj(1,1)<0)%%if dif is too small, stop iteration      
                %iIter
                %f = tmpF{iIter-1,1};
                break;
            end
            
            %% update F_hginnerH
            if mPara.iRate == 2%% 2: using the sample rate    1: no rate
                DVT2HT = DVT2*innerHT; % DV2H : Tau
                %INVDETHTDVT2 = INVDET*innerHT'*DVT2;
            elseif mPara.iRate == 1
                DVT2HT = DVT2*HT; % DV2H : Tau
                %INVDETHTDVT2 = INVDET*HT'*DVT2;               
            end

            %DVT2HTINVDETHTDVT2 = DVT2HT*F*INVDETHTDVT2;

            %obj4AllEdges = (f(:,1)'*DV2HINVDEHDV2*f(:,1) + f(:,2)'*DV2HINVDEHDV2*f(:,2));

            %watch = zeros(nEdgeT,1);
            %tmpFold = F;
            for iEdge = 1:nEdgeT
                tmp_leftT = DVT2HT(:,iEdge);
                %tmp_rightT = INVDETHTDVT2(iEdge,:);
                %obj4OneEdgeT(iEdge,1) = (g(:,1)'*(tmp_leftT*F(iEdge,iEdge)*INVDET(iEdge,iEdge)*tmp_leftT')*g(:,1) + g(:,2)'*(tmp_leftT*F(iEdge,iEdge)*INVDET(iEdge,iEdge)*tmp_leftT')*g(:,2));
                obj4OneEdgeT(iEdge,1) = (g(:,1)'*(tmp_leftT*INVDET(iEdge,iEdge)*tmp_leftT')*g(:,1) + g(:,2)'*(tmp_leftT*INVDET(iEdge,iEdge)*tmp_leftT')*g(:,2));
                %clear tmp_leftT tmp_rightT;
                clear tmp_leftT;
                %['obj = ' num2str(dif_obj) ' 1/nEdge = ' num2str(1/nEdge)  ' obj4AllEdges/(2*mu*nEdge) = ' num2str(obj4AllEdges/(2*mu*nEdge)) ' obj4OneEdge = '  num2str(i) '  ' num2str(obj4OneEdge)]
                %watch(iEdge,1) = obj4OneEdge(iEdge,1);watch(iEdge,2) = obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge;
                %watch(iEdge,3) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu); watch(iEdge,4) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
                %A_watch = [dif_obj 1/nEdge obj4AllEdges/(2*nEdge) obj4OneEdge(iEdge,1)/2];%                                                                       
            end

            obj4AllEdgesT = sum(obj4OneEdgeT);
            lamdaYF = lamda * IniY(:,1).*f(:,1);
            lamdaYFs = sum(lamdaYF,1);
            for iEdge = 1:nEdgeT
                tmpF(iEdge,iEdge) = 1/nEdgeT+(-alpha * obj4AllEdgesT - 2*lamdaYFs + 2*lamdaYF(iEdge,1)* nEdgeT + alpha *obj4OneEdgeT(iEdge,1)*nEdgeT)/(2*(mu2+lamda)*nEdgeT);%better-0  
                %watch(iEdge,5) = 1/nEdgeT+(obj4AllEdges - obj4OneEdge(iEdge,1)*nEdgeT)/(2*mu*nEdgeT);
            end

            %in case some values are below 0
            minValueT = min(diag(tmpF));
            if minValueT > 0
                F = tmpF;
            else
                diagF = diag(tmpF);
                diagF = diagF - minValueT + 1e-10;
                diagF = diagF/(1 - nEdgeT*minValueT + nEdgeT*1e-10);
                F = diag(diagF);
            end

            %watch(1,6) = obj4AllEdges;
            %tmpW3 = tmpW2; tmpW2 = tmpW1;  tmpW1 = W;
            clear DVT2HT INVDETHTDVT2 FTF L1 L2 L3 T;
            %Record_W{i_class,iteration}{i_para_lamda,1} = W;       
        end
        % if iIter ~= (nIter+1)
        %% calculate the objective function
%         tmp_sum_w1 = 0;
%         tmp_sum_w2 = 0;
%         for iEdge  = 1:nEdge
%            tmp_sum_w1 = tmp_sum_w1 + W(iEdge,iEdge);
%            tmp_sum_w2 = tmp_sum_w2 + W(iEdge,iEdge)*W(iEdge,iEdge);
%         end
%         tmp_sum_F1 = 0;
%         tmp_sum_F2 = 0;
%         for iEdge  = 1:nEdgeT
%            tmp_sum_F1 = tmp_sum_F1 + F(iEdge,iEdge);
%            tmp_sum_F2 = tmp_sum_F2 + F(iEdge,iEdge)*F(iEdge,iEdge);
%         end
%         laplacian = f(:,1)' * (mParaW - Theta) * f(:,1) + f(:,2)' * (mParaW - Theta) * f(:,2);
%         laplacian2 = alpha * g(:,1)' * (mParaF - ThetaT) * g(:,1) + g(:,2)' * (mParaF - ThetaT) * g(:,2);
%         %exploss =  lamda * (norm((f(:,1) - IniY(:,1)).*diag(sqrt(mSubW))') + norm((f(:,2) - IniY(:,2)).*diag(sqrt(mSubW)))');
%         %exploss =  lamda * ((f(:,1) - IniY(:,1))'*mParaW*(f(:,1) - IniY(:,1))+(f(:,2) - IniY(:,2))'*mParaW*(f(:,2) - IniY(:,2)));
%         exploss =  lamda * ((f(:,1) - IniY(:,1))'*(f(:,1) - IniY(:,1))+(f(:,2) - IniY(:,2))'*(f(:,2) - IniY(:,2)));
%         exploss2 =  lamda2 * ((g(:,1) - IniYT(:,1))'*(g(:,1) - IniYT(:,1))+(g(:,2) - IniYT(:,2))'*(g(:,2) - IniYT(:,2)));
        m_obj(iIter,1) = laplacian1 + laplacian2 + exploss1 + exploss2 + mu*tmp_sum_w2 + mu2*tmp_sum_F2;
        %L(iIter,1) = laplacian;
        %L(iIter,2) = exploss;
        %L(iIter,3) = laplacian+exploss;
        if iIter > 1                   
           dif_obj = m_obj(iIter,1) - m_obj(iIter-1,1);
           %objrecord(iIter,1) = dif_obj;
        end
        %% judge if to end the iteration
        [num2str(iIter) '    Entire    ' num2str(dif_obj)]
        if dif_obj > 0
           count = count+1;
        elseif dif_obj == 0
           count0 = count0+1;
        else
           count = 0;
           count0 = 0;
        end
        if (iIter > nIter)&&(m_obj(iIter,1) - m_obj(1,1)<0)  || (count > 0) || count0>=3%((iIter > 1)&&(m_obj(iIter,1) - m_obj(iIter-1,1)< -1e-4))%||& (m_obj(iIter,1) - m_obj(1,1)<0)%%if dif is too small, stop iteration      
            %iIter
            %f = tmpF{iIter-1,1};
            break;
        end
    end % end iteration
end