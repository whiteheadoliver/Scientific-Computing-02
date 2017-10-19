%Choose which image data base to look at
DataBase=input(['Would you like to look at the Yale face or MNIST Number data base?'...
    'Type "Yale" or "Numbers" to specify. '...
    '\n'],'s');


%Request number of eigenfaces to be used when comparing pictures
m=input(['How many facial/image features would you like the analyse? (up to 784 for the MNIST database'...
    '\n or 1024 for the Yale database) '...
    '\n']);
%Request how many nearest neighbours to use when running k-nearest neighbours    
k=input(['How big would you like k when running k-nearest neighbours? (1 to 5 usually suffices)'...
    '\n']);

if strcmp(DataBase,'Yale')==1


    X0=fea'; %Transpose fea'

    %******************* Section1 ************************************  
    %*****************************************************************
    %Produces two matrices of pixel data and two corresponding index vectors
    %The matrix Base0 contains vectorised images in each column. The user can
    %choose which people (any of the 1-38) they want in the analysis and 
    %how many pictures of each person (n) they want in the base set. I.e. the
    %number of columns in Base0 will be n x (no. ofpeople chosen)
    %The n pictures of each person to be used in the base set are chosen at
    %random.
    %Test0 contains the same people- with all remaining pictures of
    %those people not used in Base0




    %Requests number of pictures of each person which are to used in the training set
    n= input(['How many photos of each person would you like in the training set? (up to 59)'...
        '\nAny remaining photos of that person will be used in the test set'...
        '\n']) ;
 
    
   %Requests vector indicating which people indentifiers from 1-38 are included in the analysis
    people=input(['Please select which people you would like to use in the analysis (from 1-38).',...
        '\nE.g. For persons 1 and 6 type [1,6]. Or for persons 1,2 and 20 to 38 type [1,2,20:38] including square brackets..',...
        '\nRanges should be specified with the colon operator (:).',...
        '\n']);

    StartIndices=find(gnd-[0; gnd(1:(length(gnd)-1))]); %First column instance of a person in gnd index
    EndIndices=[(StartIndices(2:38)-1); length(gnd)]; %End column instance of a person in gnd index
    IndexMatrix0=[(1:38)' StartIndices EndIndices]; %[Person label, start col index, end col index]
    IndexMatrix1=IndexMatrix0(people,:); %As per IndexMatrix0 but only for 'people' specified

    BaseColumnIndex=zeros(n*length(people),1); %Store indices for columns we need to extract for base set
    TestColumnIndex=zeros((64-n)*length(people),1); %Store indices for columns we need to extract for test set

    %Loop through each person i that we want to analyse.
    %Find the indices for the columns of X0 which correspond to pictures of
    %person i.
    %Randomly choose n of these indices to be used to create the base set.
    %The other remaining column indices for pictures of person i are to be
    %used in the test set.
    for i=1:(length(people))
        Repeats= (IndexMatrix1(i,3)-IndexMatrix1(i,2)+1);%Number of repeated pictures of person i
        Indices=randsample(IndexMatrix1(i,2):IndexMatrix1(i,3),Repeats); %Randomly permuted index vector of
        %columns for person i in X0=fea
        BaseColumnIndex(((i-1)*n+1):i*n,1)=sort(Indices(1:n)); %Contains n column indices (of X0) for corresponding
        %to person i
        TestColumnIndex(((i-1)*(Repeats-n)+1):(i*(Repeats-n)))=sort(Indices((n+1):Repeats)); %Contains all other
        %column indices (of X0) for person i

    end

    BaseColumnIndex=BaseColumnIndex(find(BaseColumnIndex)); %remove extra zeros
    TestColumnIndex=TestColumnIndex(find(TestColumnIndex)); %remove extra zeros

    %Extract columns of X0 by using the index vectors created above. 
    Base0=X0(:,BaseColumnIndex); %Base set of pictures
    BaseLabels=gnd(BaseColumnIndex); %Person label for each column in Base0
    Test0=X0(:,TestColumnIndex); %Test set of people
    TestLabels=gnd(TestColumnIndex); %Person label for each column in Test0



    %******************* Section 2 *********************************** 
    %*****************************************************************
    %Subtract the 'mean' image (given by averaging columns of Base0) of the
    %base set, from both the Base0 and Test0 matrices

    Xm=sum(Base0,2)/size(Base0,2);
    BaseXc = Base0 - repmat(Xm,[1 size(Base0,2)]);
    TestXc = Test0 - repmat(Xm,[1 size(Test0,2)]);
    %Vector giving number of pictures of each different number
    [a,b]=hist(BaseLabels,unique(BaseLabels));


    %******************* Section 3 *********************************** 
    %*****************************************************************
    %Calls facial recignition function (see separate function m-file for
    %details of function)
    
    FaceRecognition(Base0,Test0,BaseXc,TestXc,BaseLabels,TestLabels,m,k,a)

else
    %******************* Section1 ************************************  
    %*****************************************************************
    %Produces two matrices of pixel data and two corresponding index vectors
    %The matrix Base0 contains vectorised images in each column.
    %Test0 contains the same number pictures- with all remaining pictures of
    %those numbers not used in Base0


    [BaseLabels,idx] = sort(MNIST_baseLabels(:,1));
    Base0=MNIST_baseImages(:,idx)*256;
    [TestLabels,idx2] = sort(MNIST_testLabels(:,1));
    Test0=MNIST_testImages(:,idx2)*256;
    
    %******************* Section 2 *********************************** 
    %*****************************************************************
    %Subtract the 'mean' image (given by averaging columns of Base0) of the
    %base set, from both the Base0 and Test0 matrices

    Xm=sum(Base0,2)/size(Base0,2);
    BaseXc = Base0 - repmat(Xm,[1 size(Base0,2)]);
    TestXc = Test0 - repmat(Xm,[1 size(Test0,2)]);
    
    %Vector giving number of pictures of each different number
    [a,~]=hist(BaseLabels,unique(BaseLabels));
    
    %******************* Section 3 *********************************** 
    %*****************************************************************
    %Calls facial recignition function (see separate function m-file for
    %details of function)
    FaceRecognition(Base0,Test0,BaseXc,TestXc,BaseLabels,TestLabels,m,k,a)
    
end