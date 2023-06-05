% Video giriş nesnesini oluşturun
vid = videoinput('macvideo', 1,'YCbCr422_192x192'); 

% Model için gerekli boyut ve kanal sayısını belirleyin
inputSize = [48 48];
numChannels = 1;  % Gri tonlamalı görüntüler için


labels={'kizgin','mutlu','uzgun','sasirmis'};

% Model yükleme
load('/Users/pc/Documents/NisaMatlab/savingModel/tesekkurler.mat', 'net');  % 'model_path.mat' model dosyanızın yolu olmalıdır

% onCleanup işlevini ayarlayın
cleanupObj = onCleanup(@() stop(vid));

% Çerçeve boyutunu ayarlayın
set(vid, 'FramesPerTrigger', Inf);


% Yüz algılayıcıyı oluşturun
faceDetector = vision.CascadeObjectDetector;


% Video nesnesini başlatın
start(vid)

% Figür penceresi oluşturun
hFig = figure('Menubar', 'none');

% Canlı video görüntüsü oluşturun
hImage = image(zeros(192, 192), 'Parent', gca, 'CDataMapping', 'scaled');

% Frame sayacını oluşturun
frameCounter = 0;

% Video akışını güncelleyin
while ishandle(hFig)
    img = getdata(vid, 1, 'uint8');
        
    % Görüntüyü gri tonlamalı yapın
    img = rgb2gray(img);

      if mod(frameCounter, 10) == 0
         
        % Yüzleri algılayın
        bboxes = faceDetector(img);
     
        % Eğer yüz algılandıysa, yüzün görüntüsünü al ve yeniden boyutlandır
        if ~isempty(bboxes)
            faceImage = img(bboxes(1,2):bboxes(1,2)+bboxes(1,4), bboxes(1,1):bboxes(1,1)+bboxes(1,3), :);
            
            % Görüntüyü modelin girdi boyutuna yeniden boyutlandırın
            resizedImage = imresize(faceImage, inputSize);
            
            label = classify(net, resizedImage);
            
            % Algılanan yüzleri görüntüye çizin
            img = insertObjectAnnotation(img, 'rectangle', bboxes, labels(label));
        end
    end
        
    set(hImage, 'CData', img);
    drawnow
end

% Video nesnesini durdurun
stop(vid);
delete(vid);
clear(vid);