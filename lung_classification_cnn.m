% Directory paths
fileDir = 'Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files';
audioDir = fileDir;
textDir = fileDir;

%all audio files
audioFiles = dir(fullfile(audioDir, '*.wav'));
textFiles=dir(fullfile(textDir,'*.txt'));
% Initialize cell arrays to hold data and labels
MelSpectrograms = {};
labels = {};

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
        snippet=ifft(dft_snippet,"symmetric");

        %Amplitude normalization
        normalized_snippet=snippet/max(abs(snippet));

        %mel spectrograms
        window_length=256;
        [melspec,f,t]=melSpectrogram(normalized_snippet,fs_req,"Window",hann(window_length,"periodic"),"OverlapLength",128);
        melspec=imresize(melspec,[32,15]);
        MelSpectrograms{end+1}=abs(melspec);
        labels{end+1,1}=Patient_Number;
        labels{end,2}=annotations(j,3);  %crackles
        labels{end,3}=annotations(j,4);  %wheezes
    end
end
num_files=length(MelSpectrograms);
melspectrogram_size=size(MelSpectrograms{1})
save('pre_processed_data.mat',"MelSpectrograms","labels");
%%
load("pre_processed_data.mat","MelSpectrograms","labels");

%diagnosis data
data=readtable("Respiratory_Sound_Database\Respiratory_Sound_Database\lung_diagnosis.csv");

%array to store melspectrogram labels
X=zeros([melspectrogram_size,1,num_files]);
diagnosis_labels=strings(num_files,1);

%mapping labels for spectrograms
for i=1:num_files
    X(:,:,:,i)=MelSpectrograms{i};
    Patient_Num=labels{i,1};
    index=find(data.Var1==Patient_Num);
    if ~isempty(index)
    diagnosis_labels(i)=data.Var2{index};
    end
end
categories=["Asthma", "Bronchiectasis", "Bronchiolitis", "COPD", "URTI", "Pneumonia", "Healthy"];
diagnosis_labels=categorical(diagnosis_labels,categories);
%%
load("pre_processed_data.mat","MelSpectrograms","labels");
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
%%
total_arr=X(:,:,:,(1:end));
total_labels=diagnosis((1:end));
%options
options=trainingOptions("adam","MaxEpochs",20,"MiniBatchSize",32,"InitialLearnRate",0.0001,"Plots","training-progress","Verbose",false,"ValidationData",{arr_valid,labels_valid},"ValidationFrequency",25,"ExecutionEnvironment","auto");
% Define the input size
inputSize = [32,15,1];

% Define the layers of the CNN
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    
    % First convolutional layer
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    % Second convolutional layer
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    % Third convolutional layer
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    
    % Fully connected layers
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(7, 'Name', 'fc2') % Assuming 7 classes
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];
% Create the layer graph
lgraph = layerGraph(layers);
%%
total_arr=X(:,:,:,(1:end));
total_labels=diagnosis_labels((1:end));
net=trainNetwork(arr_train,labels_train,lgraph,options);

% Predict labels for test data
total_Pred=classify(net,total_arr);
%%
% Calculate accuracy
accuracy=sum(total_Pred==total_labels)/num_files;
disp("Total Accuracy: " + accuracy);

% Plot confusion matrix
figure;
confusionchart(total_labels, total_Pred);
title('Confusion Matrix for Total Data');

% Predict labels for test data
total_Pred=classify(net,total_arr);
%%
trainlabels_Pred=classify(net,arr_train);
% Calculate accuracy
accuracy=sum(trainlabels_Pred==labels_train)/num_train;
disp("Training Accuracy: " + accuracy);

% Plot confusion matrix
figure;
confusionchart(labels_train,trainlabels_Pred);
title('Confusion Matrix for Train Data');
%%
predicted_labels=[1,0,0,1,0,0,0;
    0,9,0,2,0,0,0;
    0,0,22,0,2,0,0;
    0,0,0,506,1,9,7;
    0,0,0,2,15,2,1;
    0,0,0,17,0,39,8;
    0,0,0,10,2,0,26];
categories=["Asthma", "Bronchiectasis", "Bronchiolitis", "COPD", "URTI", "Pneumonia", "Healthy"];
confusionchart(predicted_labels,categories)
title('Confusion Matrix for Test Data');