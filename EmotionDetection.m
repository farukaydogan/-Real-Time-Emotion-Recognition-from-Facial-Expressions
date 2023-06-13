% Kamerayı oluştur
cam = webcam; 

% Figure'ü oluştur
figure;

% Yüz algılayıcıyı oluştur
faceDetector = vision.CascadeObjectDetector();

% Sınıflandırıcıyı yükle
% Burada modelinizi 'myModel' olarak adlandırdığımı varsayıyorum.
load('/Users/pc/Documents/NisaMatlab/savingModel/tesekkurler.mat','net');

% Etiketleri tanımlayın
labels = {'Angry','Happy','Sad','Surprised'};

% Döngüyü try-catch bloğu içinde çalıştır
try
    while true
        % Kameradan bir frame al
        img = snapshot(cam);
        
        % Görüntüyü siyah beyaz yap
        img = rgb2gray(img);
   
        img = imresize(img, 0.5);        
        % Yüzleri algıla
        bboxes = step(faceDetector, img);
        
        % Algılanan her yüz için
      % Algılanan her yüz için
        for i = 1:size(bboxes, 1)
            % Yüzü çıkar
            face = img(bboxes(i,2):bboxes(i,2)+bboxes(i,4), bboxes(i,1):bboxes(i,1)+bboxes(i,3));
        
            % Yüzü sınıflandırıcıya uyacak şekilde yeniden boyutlandır
            face = imresize(face, [48 48]);
        
            % Yüzü sınıflandır
            label = classify(net, face);
        
            % Sınıf etiketini bboxes'e ekle
            bboxes(i, 5) = label;
        end

        % Eğer yüzler algılandıysa, yüzlerin üzerinde etiketlerle birlikte dikdörtgen çiz
        if ~isempty(bboxes)
            % Etiketleri metne çevir
            labels = cell(size(bboxes, 1), 1);
            for i = 1:size(bboxes, 1)
                switch bboxes(i, 5)
                    case 1
                        labels{i} = 'Angry';
                    case 2
                        labels{i} = 'Happy';
                    case 3
                        labels{i} = 'Sad';
                    case 4
                        labels{i} = 'Surprised';
                end
            end

            % Algılanan yüzleri ve etiketlerini dikdörtgen ile belirt
            img = insertObjectAnnotation(img, 'rectangle', bboxes(:, 1:4), labels);
        end
        
        % Frame'i görüntüle
        imshow(img);
        
        % MATLAB'ın diğer işlemlerini gerçekleştirebilmesi için biraz duraklat
        pause(0.2); % Bu değer yüksek çerçeve hızları için artırılabilir
    end
catch ME
    % Hata durumunda (örneğin döngü kapatıldığında) burası çalışacak
    disp('Döngü kapatıldı, kamera serbest bırakılıyor...');
    fprintf('Bir hata meydana geldi:\n');
    fprintf('%s\n', getReport(ME));
end

% Kamerayı kapat ve serbest bırak
clear('cam');
