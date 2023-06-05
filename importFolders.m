function [x_train, y_train] = importFolders(klasorler)
    
    numTotalImages=0;
    for i = 1:length(klasorler)
          numTotalImages = numTotalImages+ length(dir(fullfile(klasorler{i}, '*.jpg')));
    end

    x_train = zeros( 48, 48,1,numTotalImages); % Görüntüler burada saklanır. 
    y_train = zeros(numTotalImages, 1); % Etiketler burada saklanır.

    idx = 1;
    for i = 1:length(klasorler)
        klasor = klasorler{i};
        dosyaBilgisi = dir(fullfile(klasor, '*.jpg')); % Bu kod, belirli bir klasördeki tüm .jpg dosyalarını bulur.
        for j = 1:length(dosyaBilgisi)
            fprintf("%d / %d %s veri cekiliyor \n",j,length(dosyaBilgisi),klasor)
            tamYol = fullfile(klasor, dosyaBilgisi(j).name);
            resim = imread(tamYol); % Görseli oku
            x_train(:, :,:,idx) = resim; % Görseli diziye ekle
            y_train(idx) = i; % Görselin etiketini (klasör sırasını) dizilere ekle
            idx = idx + 1;
        end
    end

    y_train = categorical(y_train); % Etiketleri kategorik değişkene dönüştürür.
end