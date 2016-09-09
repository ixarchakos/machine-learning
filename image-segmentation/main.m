clear all;
clc;
%Initialize the K vector.
K = input('Give the K parameter: ');
%Load the image.
img = imread('im.jpg');
%get image properties.
[height, width, D] = size(img);
%%
%Get the image as a N x D matrix.
initialImage = zeros(height * width, D);
for w = 1 : width
    for h = 1 : height
      n = h + (w - 1) * height;
      initialImage(n, 1) = img(h, w, 1);
      initialImage(n, 2) = img(h, w, 2);
      initialImage(n, 3) = img(h, w, 3);
    end
end
%%
%Call the EM algorithm for the selected k.
[clusteredImage, gamma, m, sigma, p] = em(K , initialImage);
%%
%Compute the reconstruction error.
error = (norm(initialImage - clusteredImage)^2) / size(initialImage, 1);
display(['Reconstruction Error = ', num2str(error)]);
%%
%Display the original image.
figure;
image(img);
%%
%create an image from a N X D matrix.
newImage = zeros(h, w, D, 'uint8');
for n = 1 : height * width
    w = fix(n / h);
    if mod(n, h) ~= 0
        w = w + 1;
    end
    he = n - (w - 1) * h;
    newImage(he, w, :) = clusteredImage(n, :); 
end
%%
%Display the new image.
figure;
image(newImage);