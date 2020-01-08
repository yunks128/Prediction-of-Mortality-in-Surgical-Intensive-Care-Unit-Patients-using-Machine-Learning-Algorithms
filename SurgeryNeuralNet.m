load data.mat
% input1 = diagnosis
% target = 0 live, 1 die

%% compute medicine_input
% medicine = 1~8 medicines
medicine_input=zeros(max(size(GS_target)),8);   % 8 different medicines

for i=1:max(size(medicine))  %39919
    for j=1:max(size(GS_target))    %1352
        for k=1:8 %medicine
            if medicine_index(i)==j && medicine(i)==k
                medicine_input(j,k)=medicine_input(j,k)+1;
            end
        end
    end
end
%%%%%%%%%
%% inhospital days summation
inhospital_input=zeros(max(size(GS_target)),1);

for i=1:max(size(inhospital_days))
    for j=1:max(size(GS_target))
        if inhospital_index(i)==j
            inhospital_input(j)=inhospital_input(j)+inhospital_days(i);
        end
    end
end

%% number of chemo
chemo_input=zeros(max(size(GS_target)),1);

for i=1:max(size(chemo_index))
    for j=1:max(size(GS_target))
        if chemo_index(i)==j
            chemo_input(j)=chemo_input(j)+1;
        end
    end
end

%% diagnosis
diagnosis_input=zeros(max(size(GS_target)),12);

for i=1:max(size(diagnosis))  
    for j=1:max(size(GS_target))    
        counter=1;
        for k=[11,15,16,18,19,20,21,30,31,34,35,36] % diagnosis code
            if diagnosis_index(i)==j && diagnosis(i)==k
                diagnosis_input(j,counter)=diagnosis_input(j,counter)+1;
            end
            counter=counter+1;
        end
    end
end

%% bloodtest
bloodtest_input=zeros(max(size(GS_target)),19);

for i=1:max(size(bloodtest_index))
    for j=1:max(size(GS_target))
        for k=1:19
        if bloodtest_index(i)==j && bloodtest(i,1)==k
            bloodtest_input(j,k)=bloodtest(i,2);
        end
        end
    end
end

%% input build
GS_input = [age, gender, inhospital_input, chemo_input, diagnosis_input, medicine_input, bloodtest_input];

load GS_input.mat
load GS_target.mat

GS_data = [GS_input, GS_target];

inputs = GS_input';
targets = GS_target';

% Create a Pattern Recognition Network
hiddenLayerSize = 200;
net = patternnet([hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize hiddenLayerSize]);


% Set up Division of Data for Training, Validation, Testing

%%%%%
AUC=[];
for i=1:10 % 10-fold cross validation
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% View the Network
view(net)

%figure;
%plotroc(targets,outputs);
%title('targets');
%figure;
%plotconfusion(targets,outputs);
%title('targets');
eval(['targets',num2str(i),'=targets;']);
eval(['outputs',num2str(i),'=outputs;']);

[X1,Y1,T,AUC1]=perfcurve(targets(1,:),outputs(1,:),1);
%[X2,Y2,T,AUC2]=perfcurve(targets(2,:),outputs(2,:),1);

figure
plot(X1,Y1)
hold on
%plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off

AUC=[AUC, AUC1];
end
mean(AUC)
[i j]=max(AUC);
eval(['targets=targets',num2str(j),';']);
eval(['outputs=outputs',num2str(j),';']);
figure;
plotroc(targets,outputs);
title('targets');

%hgsave(gcf,'ROC5layer500neuron.fig')

%%
CROSSENTROPY=[];
ALLINPUT=inputs';
OUTCOME=targets';

for i=1:43
    importanceALLINPUT=ALLINPUT;
    importanceALLINPUT(:,i)=[];
inputs = ALLINPUT';
targets = OUTCOME';

% Create a Pattern Recognition Network
hiddenLayerSize = 100;
net = patternnet([ hiddenLayerSize]);


% Set up Division of Data for Training, Validation, Testing

%%%%%
    

    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)
CROSSENTROPY=[CROSSENTROPY;performance];
i

end

% decision tree classification
figure
Model=fitctree(GS_input,GS_target)
Model=fitctree(GS_input,GS_target,'CrossVal','on')
Model=fitctree(GS_input,GS_target,'OptimizeHyperparameters','auto')

[~,score]=resubPredict(Model);
classError = kfoldLoss(Model)

diffscore = score(:,2) - max(score(:,1),score(:,1));
[X,Y,T,AUC,OPTROCPT,suby,subnames] = perfcurve(GS_target,diffscore,1);
plot(X,Y)
hold on
%plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off

view(Model,'Mode','graph');

%best level compute
resubcost = resubLoss(Model,'Subtrees','all');
[cost,secost,ntermnodes,bestlevel] = cvloss(Model,'Subtrees','all');
%plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
plot(ntermnodes,cost,'b-')
figure(gcf);
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')

[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
%plot([0 100], [cutoff cutoff], 'k:')
%plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
%legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
legend('Cross-validation')
hold off

bestlevel=17
pt = prune(Model,'Level',bestlevel);
view(pt,'Mode','graph')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% logstic regression
pred=GS_input;
resp=GS_target;
mdl = fitglm(pred,resp,'Distribution','binomial','Link','logit');
scores = mdl.Fitted.Probability;
[X,Y,T,AUC] = perfcurve(species(51:end,:),scores,'virginica');
AUC
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')
