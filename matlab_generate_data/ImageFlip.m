clear;
close all;
clc;
%% image flip
file_path = 'Train/';
Img_path_list = dir(strcat(file_path, '*.bmp'));
img_num = length(Img_path_list);
% mkdir('train1');
% mkdir('train2');
% mkdir('train3');
if img_num>0    
    for i=1:img_num       
        image_name = Img_path_list(i).name;        
        im = imread(strcat(file_path, image_name));
%         imwrite(im, ['Train/11_' num2str(i), '.bmp'], 'bmp');
        im1 = flip(im, 2);
        imwrite(im1, ['Train/22_' num2str(i), '.bmp'], 'bmp');
        im2 = flip(im);
        imwrite(im2, ['Train/33_' num2str(i), '.bmp'], 'bmp');
        im3 = flip(im1);
        imwrite(im3, ['Train/44_' num2str(i), '.bmp'], 'bmp');
%         figure;imshow(im);
%         figure;imshow(im1);
%         figure;imshow(im2);
%         figure;imshow(im3);
    end
end