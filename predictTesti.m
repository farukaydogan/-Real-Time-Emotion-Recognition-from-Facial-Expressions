labels={'kizgin','mutlu','uzgun','sasirmis'};

% Model yükleme
load('/Users/pc/Documents/NisaMatlab/savingModel/tesekkurler.mat', 'net');  % 'model_path.mat' model dosyanızın yolu olmalıdır

%Görüntüyü okuyun
img = imread('/Users/pc/Documents/NisaMatlab/sasirmis.png');  % 'image.jpg' okunacak görüntünün ismi olmalı

img = rgb2gray(img);

% Görüntüyü modelin girdi boyutuna yeniden boyutlandırın
inputSize = [48 48];  % Modelin giriş boyutu
resizedImage = imresize(img, inputSize);

% Modeli kullanarak görüntüyü sınıflandırın
label = classify(net, resizedImage);

% Sınıf etiketini gösterin
disp(labels{label});