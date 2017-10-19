function [] = FaceRecognition(Base0,Test0,BaseXc,TestXc,BaseLabels,TestLabels,m,k,a)

%*****************************Section 3.1*********************************%
%Here we find the distinguishable features of our training set, our
%eigenvectors, V. We then find how prominent these features are in each of 
%our test images.

%Find the eigenvectors of our covariance matrix, aka distinguishing feature
%vectors, V, from our base data set (These vectors are automatically put in
%order of increasing eigenvalues, i.e. increasing importance)
[V,~] = eig(BaseXc*BaseXc');
%Find the weightings of each of our images (base set and test set) for each
%of our eigenvectors. This gives us how prominent the features are in each picture.
Xf=V'*BaseXc; Yf=V'*TestXc;
%Then shorten these vectors so we are only using the m most important
%features to distinguish images.
Xf=flipud(Xf); Xf=Xf(1:m,:); Yf=flipud(Yf); Yf=Yf(1:m,:);

%Then, for each test picture, find the k base pictures which most closely
%match the test in terms of these weightings
KNearest=knnsearch(Xf',Yf','K',k,'dist','cityblock');
IKNearest=BaseLabels(KNearest);
%Then see, out of the closest pictures, which identifier is most common.
if k==1
    %(No need for a vote if we only finding the closest picture)
    BestGuess=IKNearest;
else
BestGuess=mode(IKNearest')';
end
%Output how successful the classification was.
SuccessRate=sum(BestGuess==TestLabels)/length(TestLabels);
fprintf('Success rate = %.3g %%\n',100*SuccessRate)



%****************************Section 3.2**********************************%
%This section gives you different ways of visualising the results.

%                      ********Option 1********                           %
Reply = input('Would you like to see an example matched image? (y/n)','s');
%This option shows a random test picture with its k-nearest-neighbours
while Reply=='y' 
    close all
    %Pick a random picture from the test set.
    R=randi(length(BestGuess));
    %Recall the k-nearest-neighbours to that image
    KNearestR=KNearest(R,:);
    %The images are currently vectors, need to reshape into square matrices
    [l,~]=size(Test0);
    ExampleTestPic=uint8(reshape(Test0(:,R),[sqrt(l),sqrt(l)]));
    %Plot all the neighbours,
    for i=1:k
    subplot(2,k,k+i)
    imshow(uint8(reshape(Base0(:,KNearestR(i)),[sqrt(l),sqrt(l)])))
    end
    %and plot the original test image at the top.
    subplot(2,k,floor((k+1)/2));
    imshow(ExampleTestPic)
    title('Example Image from Test Set')
    annotation('textbox', [0 0.52 1 0],'String', ...
       ' Most Similar Images in Training Set','FontSize',12,'FontWeight',...
       'bold','EdgeColor', 'none','HorizontalAlignment', 'center')
    %Output whether or not the displayed match was a successful match
    if BestGuess(R)==TestLabels(R)
        fprintf('This was a correct match\n')
    else
        fprintf('This was an incorrect match\n')
    end
        Reply = input('Would you like to see another? (y/n)','s');
end

%                      ********Option 2********                           %
Reply = input('Would you like to see an example of clustering of the base data set? (y/n)\n','s');
%This option displays a scatter graph of some of the base data set in space 
%made by two of the eigenvectors. This should give you an idea how similar
%the faces of the same identifier are to each other compared to other
%identifiers
while Reply == 'y'
    close all
    %We need to pick 2 feature vectors to make up the axes of our
    %scatter graph so can't have m=1.
    if m<2
        disp('You need more than 1 feature to plot the clusters. (Need m > 1)')
        break
    end
    %Pick the 2 random feature vectors from our m most important ones.
    R=randperm(m);
    R=[R(1),R(2)];
    
    %Recall the weightings for each base image for these vectors.
    Scat=[Xf(R(1),:)',Xf(R(2),:)'];
    hold on
    IxR=unique(BaseLabels);
    %If you have specified a lot of people or are using the MNIST Database,
    %the scatter plot will look very cluttered and unclear. So we will only
    %plot a random 4 of these.
    %Incase there are too many identifiers, pick up to 4 random numbers from the set 
    r=IxR(sort(randsample(1:length(a),min(4,length(a)))));
    Legend=[];
    b=a;
    while isempty(Scat)==0
    %Only plot data for theidentifiers with the random numbers picked above
        if isempty(r)==0 && r(1)==IxR(1)
    %Scatter plot each identifier in a different colour
    s=scatter(Scat(1:b(1),1),Scat(1:b(1),2),'filled');
    s.SizeData=20;
    Legend=[Legend;IxR(1)];
    r(1)=[];
        end
    Scat(1:b(1),:)=[];
    IxR(1)=[];
    b(1)=[];
    end
    legend(strcat('Number',num2str(Legend)))
    Reply = input('This was made by using two random features. Would you like to see another? (y/n)','s');
    
end
end