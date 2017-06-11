% ICML paperID29: demo code
clear;clc;close all;
% you should install matconvnet first: http://www.vlfeat.org/matconvnet/
% run(fullfile('matconvnet','matlab','vl_setupnn.m'));
opts.dataDir = pwd; % data path
imdbfile = fullfile(opts.dataDir,'imdb.mat');
if ~exist(imdbfile,'file')
    imdb = getMnistImdb(opts); % MNIST dataset
    save(imdbfile,'-struct','imdb');
else
    imdb = load(imdbfile);
end
load LeNet.mat; % The accuracy of the original network is 99.20%
net.layers{end}.type = 'softmax';
gpuindex = 2;
h = gpuDevice(gpuindex);
reset(h);
%% display size of conv filters
layernum = length(net.layers);
ind = 0;
for i = 1:layernum
    if strcmp(net.layers{i}.type,'conv')
        ind = ind+1;
        fprintf('Filter size of convolutional layer %d is :\n',ind);
        disp(size(net.layers{i}.weights{1}));
    end
end
%% evaluating
test_ind = find(imdb.images.set == 3);
net = vl_simplenn_move(net, 'gpu'); % 'cpu'
imdb.images.data = gpuArray(imdb.images.data);
res = vl_simplenn(net,imdb.images.data(:,:,1,test_ind));
[~,cl] = max(squeeze(res(end).x));
correct = sum(cl==imdb.images.labels(test_ind))/length(test_ind);
fprintf('The classification accuracy on MNIST is %.2f%%.\n',correct*100);