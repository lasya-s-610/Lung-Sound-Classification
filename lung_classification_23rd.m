% Directory paths
fileDir = 'Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files';
audioDir = fileDir;
textDir = fileDir;

%all audio files
audioFiles = dir(fullfile(audioDir, '*.wav'));
textFiles=dir(fullfile(textDir,'*.txt'));
% Initialize cell arrays to hold data and labels
MelSpectrograms = {};
labels_train_resampled = {};
for i=1:length(audioFiles)
    filename=audioFiles(i).name;

    %patient number
    parts=split(filename,"_");
    Patient_Number=str2double(parts{1});
    Recording_Index=parts{2};
    Chest_Location=parts{3};
    Acquisition_Mode=parts{4};
    Recording_Equipment=parts{5}(1:end-4);  %to remove .wav

    %load audio,txt files
    audio_path=fullfile(audioDir,filename);
    text_path=fullfile(textDir,strrep(filename,'.wav','.txt'));
    [sig,fs]=audioread(audio_path);
    annotations=readmatrix(text_path);

    %resampling
    fs_req=4000;
    resampled_sig=resample(sig,fs_req,fs);

    %snippet generation
    for j=1:size(annotations,1)
        sample_start=round(annotations(j,1)*fs_req);
        sample_end=round(annotations(j,2)*fs_req);

        % Ensure the indices are within the valid range
        if sample_start<fs_req/60
            sample_start=67;
        end
        snippet=resampled_sig(sample_start:sample_end);
        
        %DFT baseline wander removal
        dft_snippet=fft(snippet);
        dft_snippet(1)=0;   %removing dc component
        dft_removal_snippet=ifft(dft_snippet,"symmetric");

        %Amplitude normalization
        normalized_snippet=dft_removal_snippet/max(abs(dft_removal_snippet));

        %mel spectrograms
        window_length=256;
        [melspec,f,t]=melSpectrogram(normalized_snippet,fs_req,"Window",hann(window_length,"periodic"),"OverlapLength",128);
        melspec=imresize(melspec,[32,15]);
        melspec_rgb = repmat(log(melspec), [1, 1, 3]);
        MelSpectrograms{end+1}=abs(melspec_rgb);
        labels_train_resampled{end+1,1}=Patient_Number;
        labels_train_resampled{end,2}=annotations(j,3);  %crackles
        labels_train_resampled{end,3}=annotations(j,4);  %wheezes
    end
end
num_files=length(MelSpectrograms);
melspectrogram_size=size(MelSpectrograms{1});
save('pre_processed_data.mat',"MelSpectrograms","labels_train_resampled");
%%
figure;
subplot(2, 2, 1);
plot(resampled_sig);
title('After Resampling');
xlabel('Samples');
ylabel('Amplitude');

subplot(2, 2, 2);
plot(snippet);
title('Snippet generation');
xlabel('Samples');
ylabel('Amplitude');

% Step 2: DFT Baseline Wander Removal

subplot(2, 2, 3);
plot(dft_removal_snippet);
title('After DFT Baseline Wander Removal');
xlabel('Samples');
ylabel('Amplitude');

subplot(2, 2, 4);
plot(normalized_snippet);
title('After Amplitude Normalization');
xlabel('Samples');
ylabel('Normalized Amplitude');
%%
% Visualize the first 4 Mel-spectrograms
figure;

for i = 1:4
    subplot(2, 2, i);
    imagesc(f, t, log(abs(MelSpectrograms{i})));
    axis xy;
    xlabel('Frequency (Hz)');
    ylabel('Time (s)');
    title(['Log scaled Mel-Spectrogram ' num2str(i)]);
    colorbar;
end
%%
load("pre_processed_data.mat","MelSpectrograms","labels_train_resampled");

%diagnosis data
data=readtable("Respiratory_Sound_Database\Respiratory_Sound_Database\lung_diagnosis.csv");

%array to store melspectrogram labels
X=zeros([melspectrogram_size,1,num_files]);
diagnosis_labels=strings(num_files,1);

%mapping labels for spectrograms
for i=1:num_files
    X(:,:,:,i)=MelSpectrograms{i};
    Patient_Num=labels_train_resampled{i,1};
    index=find(data.Var1==Patient_Num);
    if ~isempty(index)
    diagnosis_labels(i)=data.Var2{index};
    end
end
categories=["Asthma", "Bronchiectasis", "Bronchiolitis", "COPD", "URTI", "Pneumonia", "Healthy"];
diagnosis_labels=categorical(diagnosis_labels,categories);
%%
load("pre_processed_data.mat","MelSpectrograms","labels_train_resampled");
training_percent=0.8;
validation_percent=0.1;
testing_percent=0.1;
num_train=floor(training_percent*num_files);
num_valid=floor(validation_percent*num_files);
num_test=floor(testing_percent*num_files);

%assigning data to 3 sets
arr_train=X(:,:,:,(1:num_train));
arr_valid=X(:,:,:,(num_train+1:num_train+num_valid));
arr_test=X(:,:,:,(num_valid+num_train+1:end));

%labels
labels_train=diagnosis_labels((1:num_train));
labels_valid=diagnosis_labels((num_train+1:num_train+num_valid));
labels_test=diagnosis_labels((num_valid+num_train+1:end));
tabulate(labels_train)
%%
% Perform oversampling for underrepresented classes
desiredCount = 5*median(histcounts(labels_train));  % Define the desired sample count per class

% Perform oversampling
[arr_train_resampled,labels_train_resampled] = randomOverSampler(arr_train,labels_train,desiredCount);
tabulate(labels_train_resampled)
% Ensure labels are categorical
labels_train_resampled = categorical(labels_train_resampled);

% Create augmented image datastore
imageAugmenter = imageDataAugmenter('RandRotation',[-10,10], 'RandXTranslation',[-3,3], 'RandYTranslation',[-3,3], 'RandXScale',[0.9,1.1], 'RandYScale',[0.9,1.1]);
augimdsTrain = augmentedImageDatastore([32 15 1], arr_train_resampled, labels_train_resampled, 'DataAugmentation', imageAugmenter);
%%
%Network
%options
options=trainingOptions("adam","MaxEpochs",25,"MiniBatchSize",32,"Shuffle","every-epoch",'InitialLearnRate',0.0001,'LearnRateSchedule','piecewise','LearnRateDropFactor', 0.5,'LearnRateDropPeriod', 10,"Plots","training-progress","Verbose",false,"ValidationData",{arr_valid,labels_valid},"ValidationFrequency",35,"L2Regularization",0.0001);

% Define the layers for the proposed lightweight inception network: RDLINet
layers = [
    imageInputLayer([32 15 1], 'Name', 'input', 'Normalization', 'none')  
    
    % Conv 2D [3x3]
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1','WeightL2Factor', 0.0002)
    leakyReluLayer(0.3,'Name', 'leaky_relu1')

    % Max pool 2D
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'max_pool1')
    
    % Depth-wise separable Conv (replaced with grouped convolution)
    groupedConvolution2dLayer(3,16,1, 'Padding', 'same', 'Name', 'depth_conv1','WeightL2Factor', 0.0002)
    leakyReluLayer(0.3,'Name', 'leaky_relu2')
    convolution2dLayer(1, 32, 'Padding', 'same', 'Name', 'pointwise_conv1','WeightL2Factor', 0.0002)
    leakyReluLayer(0.3,'Name', 'leaky_relu3')

    % Max pool 2D
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'max_pool2')
    
    % First MFLI block

    % Branch 1
    convolution2dLayer(1, 16, 'Padding', 'same', 'Name', 'conv4','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu5')
    maxPooling2dLayer([4 3], 'Stride', [4 3], 'Name', 'adjust_pool1')
    % Branch 2
    groupedConvolution2dLayer(3,16,1, 'Padding', 'same', 'Name', 'depth_conv2','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu6')
    convolution2dLayer(1, 48, 'Padding', 'same', 'Name', 'conv5','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu7')
     maxPooling2dLayer([4 3], 'Stride', [4 3], 'Name', 'adjust_pool2')
    % Branch 3
    groupedConvolution2dLayer(5,16,1, 'Padding', 'same', 'Name', 'depth_conv3','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu8')
    convolution2dLayer(1, 48, 'Padding', 'same', 'Name', 'conv6','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu9')
     maxPooling2dLayer([4 3], 'Stride', [4 3], 'Name', 'adjust_pool3')
    % Branch 4
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'max_pool3')
    convolution2dLayer(1, 32, 'Padding', 'same', 'Name', 'conv7','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu10')
     maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'adjust_pool4')
    % Concatenate branches
    depthConcatenationLayer(4, 'Name', 'concat1')
    
    % Second MFLI block

    % Branch 1
    convolution2dLayer(1, 16, 'Padding', 'same', 'Name', 'conv8','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu11')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'adjust_pool5')
    % Branch 2
    groupedConvolution2dLayer(3,16,1, 'Padding', 'same', 'Name', 'depth_conv4','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu12')
    convolution2dLayer(1, 48, 'Padding', 'same', 'Name', 'conv9','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu13')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'adjust_pool6')
    % Branch 3
    groupedConvolution2dLayer(5,16,1, 'Padding', 'same', 'Name', 'depth_conv5','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu14')
    convolution2dLayer(1, 48, 'Padding', 'same', 'Name', 'conv10','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu15')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'adjust_pool7')
    % Branch 4
    maxPooling2dLayer([2 1], 'Stride', 1, 'Name', 'max_pool4')
    convolution2dLayer(1, 32, 'Padding', 'same', 'Name', 'conv11','WeightL2Factor', 0.0003)
    leakyReluLayer(0.3,'Name', 'leaky_relu16')
    maxPooling2dLayer([1 1], 'Stride', [1 1], 'Name', 'adjust_pool8')
    % Concatenate branches
    depthConcatenationLayer(4, 'Name', 'concat2')
    
    % Max pool 2D
    maxPooling2dLayer([1 1], 'Stride',[1 1], 'Name', 'max_pool5')
    
    % Global average pool 2D
    globalAveragePooling2dLayer('Name', 'global_avg_pool')
    
    % GLU Classifier module
    fullyConnectedLayer(5, 'Name', 'fc1','WeightL2Factor', 0.0003)
    sigmoidLayer('Name', 'sigmoid')
    fullyConnectedLayer(5, 'Name', 'fc2','WeightL2Factor', 0.0003)
    multiplicationLayer(2, 'Name', 'multiplication')
    fullyConnectedLayer(7, 'Name', 'fc3','WeightL2Factor', 0.0003)

    % Softmax layer
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];


% Create a layer graph
lgraph = layerGraph(layers);
%MFLI block connections
if ismember('adjust_pool3', {lgraph.Layers.Name})
    lgraph = disconnectLayers(lgraph, 'adjust_pool3', 'max_pool3/in');
end
lgraph=connectLayers(lgraph,'max_pool2','max_pool3/in');
if ismember('adjust_pool1', {lgraph.Layers.Name})
    lgraph = disconnectLayers(lgraph, 'adjust_pool1', 'depth_conv2/in');
end
lgraph=connectLayers(lgraph,'max_pool2','depth_conv2/in');
if ismember('adjust_pool2', {lgraph.Layers.Name})
    lgraph = disconnectLayers(lgraph, 'adjust_pool2', 'depth_conv3/in');
end
lgraph=connectLayers(lgraph,'max_pool2','depth_conv3/in');
if ismember('adjust_pool7', {lgraph.Layers.Name})
    lgraph = disconnectLayers(lgraph, 'adjust_pool7', 'max_pool4/in');
end
lgraph=connectLayers(lgraph,'concat1','max_pool4/in');
if ismember('adjust_pool5', {lgraph.Layers.Name})
    lgraph = disconnectLayers(lgraph, 'adjust_pool5', 'depth_conv4/in');
end
lgraph=connectLayers(lgraph,'concat1','depth_conv4/in');
if ismember('adjust_pool6', {lgraph.Layers.Name})
    lgraph = disconnectLayers(lgraph, 'adjust_pool6', 'depth_conv5/in');
end
lgraph=connectLayers(lgraph,'concat1','depth_conv5/in');
% Connect branches of first MFLI block to concatenation layer
lgraph = connectLayers(lgraph, 'adjust_pool1', 'concat1/in2');
lgraph = connectLayers(lgraph, 'adjust_pool2', 'concat1/in3');
lgraph = connectLayers(lgraph, 'adjust_pool3', 'concat1/in4');

% Connect branches of second MFLI block to concatenation layer
lgraph = connectLayers(lgraph, 'adjust_pool5', 'concat2/in2');
lgraph = connectLayers(lgraph, 'adjust_pool6', 'concat2/in3');
lgraph = connectLayers(lgraph, 'adjust_pool7', 'concat2/in4');

% Connect layers for GLU 
if ismember('sigmoid', {lgraph.Layers.Name})
    lgraph = disconnectLayers(lgraph, 'sigmoid', 'fc2/in');
end
lgraph=connectLayers(lgraph,'global_avg_pool','fc2/in');
lgraph = connectLayers(lgraph, 'sigmoid', 'multiplication/in2');
% Display network
figure;
plot(lgraph);
title('RDLINet Architecture');
%
net=trainNetwork(augimdsTrain,lgraph,options);

% Predict labels for test data
labels_Pred=classify(net,arr_test);

% Calculate accuracy
accuracy=sum(labels_Pred==labels_test)/num_test;
disp("Test Accuracy: " + accuracy);
%%
% Plot confusion matrix
figure;
confusionchart(labels_test, labels_Pred);
title('Confusion Matrix for Test Data');
%%
function [X_resampled, y_resampled] = randomOverSampler(X, y, desiredCount)
    % Initialize the resampled data and labels
    X_resampled = X;
    y_resampled = y;
    
    % Get unique classes and their counts
    classes = categories(y);
    classCounts = countcats(y);
    
    % Loop through each class and perform oversampling
    for i = 1:numel(classes)
        class = classes(i);
        currentCount = classCounts(i);
        
        if currentCount < desiredCount
            augmentCount = desiredCount - currentCount;
            classData = X(:,:,:,y == class);
            
            % Randomly sample with replacement
            for j = 1:augmentCount
                index = randi(size(classData, 4));
                X_resampled(:,:,:,end+1) = classData(:,:,:,index);
                y_resampled(end+1) = class;
            end
        end
    end
end