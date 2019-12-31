
function getImgTxtFea()
clear all;
mImageTextFea = cell(1,1); %暂先考虑one type of feature: Image

%***********************  get ImageFea ******************************
ImageFilesPath = 'VSO\source\result\';
PreImageNum = 810;
ImageNum = 596; %810中只有596ge
FeaDim = 1200; 
mImageTextFea{1,1} = zeros(ImageNum,FeaDim); %pos: 1 - 463 ; neg: 463:596
load([ImageFilesPath 'gvalue.mat']);

iP = 0;
for i = 1:PreImageNum
    if gvalue(i,1)==1
        infilename = [ImageFilesPath num2str(i) '-biconcept.mat'];
        sgc_exist = exist(infilename, 'file');
        if(sgc_exist==0)
            continue;
        end
        load(infilename);
        iP = iP +1;
        mImageTextFea{1,1}(iP,:) = biconceptVector';
    end
end

iN = iP;
for i = 1:PreImageNum
    if gvalue(i,1)==0
        infilename = [ImageFilesPath num2str(i) '-biconcept.mat'];
        sgc_exist = exist(infilename, 'file');
        if(sgc_exist==0)
            continue;
        end
        load(infilename);
        iN = iN +1;
        mImageTextFea{1,1}(iN,:) = biconceptVector';
    end
end
['posImageNum = ' num2str(iP) '   NegImageNum = ' num2str(iN-iP)]


%***********************  get TextFea ******************************
% have not had
save datas/mImageTextFea ;

end