function [f mPara] = HG_learning(H,IniY,mPara)
%% conduct hypergraph learning in H

[nObject nEdge] = size(H);
lamda = mPara.mLamda(mPara.iLamda); % the parameter in SSL, i.e., lamda
mu = mPara.mMu(mPara.iMu);
IsWeight = mPara.IsWeight;

W = eye(nEdge)/nEdge; 

%mSubW = mPara.mSubW; % the subjet weights to each subjects (it is based on the number of positive samples, negative samples and test samples.
%H = H.*repmat(diag(mSubW),1,nEdge);
%innerH = H.*repmat(diag(mSubW),1,nEdge);
innerH = H;


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
    for iIter = 1:nIter+1
        if flag == 0
           %% DV DE INVDE calculation
           DV = eye(nObject);
           for iObject = 1:nObject
              DV(iObject,iObject) = sum(H(iObject,:).*diag(W)'); 
           end
           DVtemp = diag(DV);
           DVtemp(find(DVtemp==0))=1e-10;
           DV = diag(DVtemp);
           DE = eye(nEdge);
           for iEdge = 1:nEdge
               DE(iEdge,iEdge)=sum(H(:,iEdge)); 
           end
           DEtemp = diag(DE);
           DEtemp(find(DEtemp==0))=1e-10;
           DE = diag(DEtemp);
           
           DV2 = DV^(-0.5);
           INVDE = inv(DE);

           %% change for new formualtion
           if mPara.iRate == 2%% 2: using the sample rate    1: no rate
                Theta = DV2*innerH*W*INVDE*innerH'*DV2;
                mParaW = mSubW;%diag(sum(innerH,2).*diag(mSubW));
           elseif mPara.iRate == 1
                 Theta = DV2*H*W*INVDE*H'*DV2;
           end
           
           %% end of change for new formulation
           
           eta = 1/(1+lamda);
           L2 = mParaW-eta*Theta;
           f = (lamda/(1+lamda))*inv(L2) * mParaW*IniY;
           tmpF{iIter,1} = f;

           %%calculate the objective function
           tmp_sum_w1 = 0;
           tmp_sum_w2 = 0;
           for iEdge  = 1:nEdge
               tmp_sum_w1 = tmp_sum_w1 + W(iEdge,iEdge);
               tmp_sum_w2 = tmp_sum_w2 + W(iEdge,iEdge)*W(iEdge,iEdge);
           end
           laplacian = f(:,1)' * (mParaW - Theta) * f(:,1) + f(:,2)' * (mParaW - Theta) * f(:,2);
           %exploss =  lamda * (norm((f(:,1) - IniY(:,1)).*diag(sqrt(mSubW))') + norm((f(:,2) - IniY(:,2)).*diag(sqrt(mSubW)))');
           %exploss =  lamda * ((f(:,1) - IniY(:,1))'*mParaW*(f(:,1) - IniY(:,1))+(f(:,2) - IniY(:,2))'*mParaW*(f(:,2) - IniY(:,2)));
           exploss =  lamda * ((f(:,1) - IniY(:,1))'*(f(:,1) - IniY(:,1))+(f(:,2) - IniY(:,2))'*(f(:,2) - IniY(:,2)));
           m_obj(iIter,1) = laplacian + exploss + mu*tmp_sum_w2;
           L(iIter,1) = laplacian;
           L(iIter,2) = exploss;
           L(iIter,3) = laplacian+exploss;
           if iIter > 1                   
               dif_obj = m_obj(iIter,1) - m_obj(iIter-1,1);
               objrecord(iIter,1) = dif_obj;
           end
           %% update w_hginnerH

           [num2str(iIter) '     ' num2str(dif_obj)]
           if dif_obj > 0
               count = count+1;
           elseif dif_obj == 0
               count0 = count0+1;
           else
               count = 0;
               count0 = 0;
           end

           if (iIter > nIter)&&(m_obj(iIter,1) - m_obj(1,1)<0)  || (count > 0) || count0>=3%((iIter > 1)&&(m_obj(iIter,1) - m_obj(iIter-1,1)< -1e-4))%||& (m_obj(iIter,1) - m_obj(1,1)<0)%%if dif is too small, stop iteration      
                flag = 1;
                %iIter
                f = tmpF{iIter-1,1};
           else
                 if mPara.iRate == 2%% 2: using the sample rate    1: no rate
                   DV2H = DV2*innerH; % DV2H : Tau
                   INVDEHDV2 = INVDE*innerH'*DV2;
                elseif mPara.iRate == 1
                   DV2H = DV2*H; % DV2H : Tau
                   INVDEHDV2 = INVDE*H'*DV2;               
                 end
               
               DV2HINVDEHDV2 = DV2H*W*INVDEHDV2;
               if iIter ~= (nIter+1)
                   obj4AllEdges = (f(:,1)'*DV2HINVDEHDV2*f(:,1) + f(:,2)'*DV2HINVDEHDV2*f(:,2));

                    watch = zeros(nEdge,1);
                    tmpWold = W;
                    for iEdge = 1:nEdge
                        tmp_left = DV2H(:,iEdge);
                        tmp_right = INVDEHDV2(iEdge,:);
                        obj4OneEdge(iEdge,1) = (f(:,1)'*(tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left')*f(:,1) + f(:,2)'*(tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left')*f(:,2));
                        clear tmp_left tmp_right;
                         %['obj = ' num2str(dif_obj) ' 1/nEdge = ' num2str(1/nEdge)  ' obj4AllEdges/(2*mu*nEdge) = ' num2str(obj4AllEdges/(2*mu*nEdge)) ' obj4OneEdge = '  num2str(i) '  ' num2str(obj4OneEdge)]
                        watch(iEdge,1) = obj4OneEdge(iEdge,1);watch(iEdge,2) = obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge;
                        watch(iEdge,3) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu); watch(iEdge,4) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
                        A_watch = [dif_obj 1/nEdge obj4AllEdges/(2*nEdge) obj4OneEdge(iEdge,1)/2];%                                                                       
                    end
                    
                    obj4AllEdges = sum(obj4OneEdge);
                    for iEdge = 1:nEdge                  
                        tmpW(iEdge,iEdge) = 1/nEdge+(obj4OneEdge(iEdge,1)*nEdge - obj4AllEdges)/(2*mu*nEdge);%better-0  
                        watch(iEdge,5) = 1/nEdge+(obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
                    end
                    
                    %% in case some values are below 0
                    minValue = min(diag(tmpW));
                    if minValue > 0
                        W = tmpW;
                    else
                       diagW = diag(tmpW);
                       diagW = diagW - minValue + 1e-10;
                       diagW = diagW/(1 - nEdge*minValue + nEdge*1e-10);
                       W = diag(diagW);
                    end

                    watch(1,6) = obj4AllEdges;
                    tmpW3 = tmpW2; tmpW2 = tmpW1;  tmpW1 = W;
                    clear DV2H INVDEHDV2 FTF L1 L2 L3 T;
                    %Record_W{i_class,iteration}{i_para_lamda,1} = W;             
               end
           end
       end
    end % end iteration
end