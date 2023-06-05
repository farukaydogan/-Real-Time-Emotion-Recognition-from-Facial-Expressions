trainFerKlasorler = {'/Users/pc/Documents/NisaMatlab/datas/train/angry', '/Users/pc/Documents/NisaMatlab/datas/train/happy', '/Users/pc/Documents/NisaMatlab/datas/train/sad','/Users/pc/Documents/NisaMatlab/datas/train/surprise'}; % Klasör isimlerinizi girmeniz gerekmektedir.
testFerKlasorler = {'/Users/pc/Documents/NisaMatlab/datas/test/angry', '/Users/pc/Documents/NisaMatlab/datas/test/happy', '/Users/pc/Documents/NisaMatlab/datas/test/sad','/Users/pc/Documents/NisaMatlab/datas/test/surprise'}; % Klasör isimlerinizi girmeniz gerekmektedir.
[x_train, y_train] = importFolders(trainFerKlasorler);
[x_test, y_test] = importFolders(testFerKlasorler);


 

model = [
    imageInputLayer([48 48 1], 'Name', 'input')

    convolution2dLayer(3, 64, 'Padding', 'same', 'WeightsInitializer', 'glorot', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    dropoutLayer(0.6, 'Name', 'dropout1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    dropoutLayer(0.6, 'Name', 'dropout2')

    fullyConnectedLayer(128, 'Name', 'fc')
    batchNormalizationLayer('Name', 'bn6')
    reluLayer('Name', 'relu6')
    dropoutLayer(0.6, 'Name', 'dropout3')

    fullyConnectedLayer(4, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];



options = trainingOptions('adam', ...
    'Plots', 'training-progress', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 100, ...
    'ValidationData', {x_test,y_test}, ...
    'ValidationFrequency', 30, ...
    'Shuffle','every-epoch',...
    'Verbose', true);

 net = trainNetwork(x_train,y_train, model, options);




% CNN'yi eğit
disp('Model eğitimi başlatıldı');

YPred = classify(net, x_test);

accuracy = sum(YPred == y_test)/numel(y_test);  

fprintf('Doğruluk Oranı: %f\n', accuracy);

% Confusion Matrix
confusionMatrix = confusionmat(y_test, YPred);

% Confusion Matrix'in Gösterimi
confusionchart(confusionMatrix);

%model kaydediliyor
path="/Users/pc/Documents/NisaMatlab/savingModel/lastTrainingModel.mat";
save(path, 'net');
fprintf('%s modeliniz başarıyla kaydedilmiştir \n', path);