close all; clear all; clc

%access all names: meta(:).filename  
%access one name: meta(5).filename
%access one self score: meta(5).skill_self
%access corresponding kinematics values: K(5)
%access transcriptions: T(5)

%add all subfolders to path
folder = fileparts(which(mfilename)); 
addpath(genpath(folder));

%% Import meta stuff as a structure array
meta = readtable('JIGSAWS/Suturing/meta_file_Suturing.txt');
meta = table2struct(meta);
meta = rmfield(meta, 'Var2');
meta = rmfield(meta, 'Var5');
meta = cell2struct( struct2cell(meta), {'filename', 'experience', 'GRS', 'tissue','suture','motion','flow','overall','quality'});
num_surgeons = size(meta,1);

%% Import kinematics
K = containers.Map('KeyType','double','ValueType','any');
i = 1;
for name = {meta(:).filename}
   disp('Importing kinematics: ' + string(name));
   K(i) = importdata('JIGSAWS/Suturing/kinematics/AllGestures/'+string(name)+'.txt');
   i = i + 1;
end

%% Import transcriptions
T = containers.Map('KeyType','double','ValueType','any');
i = 1;
for name = {meta(:).filename}
   disp('Importing transcriptions: ' + string(name));
   T(i) = table2cell(readtable('JIGSAWS/Suturing/transcriptions/'+string(name)+'.txt'));
   i = i + 1;
end
%% Generate CSV for one gesture
gesture_number = 6;
Gesture = 'G' + string(gesture_number);  %enter number of gesture

Expert_Samples = [];  %Outputs
Novice_Samples = [];

for i = 1:num_surgeons
       Kinematics_i = K(i);   %Get surgeon's kinematics file
       
       %smooth out velocities
       for z = 13:18
           Kinematics_i(:,z) = smoothdata(Kinematics_i(:,z));
       end
       
       Transcription_i = T(i);   %Get surgeon's transcription file
       index = find(strcmp(Transcription_i(:,3), Gesture));   % Get frame stamps for the gesture
       
       for j = 1:size(index,1)   %for every frame pair get the sequence between
          frame_start = cell2mat(Transcription_i(index(j),1));
          frame_end = cell2mat(Transcription_i(index(j),2));
          
          % convert ROT to euler angles (better feature?)
          EUL = [];
          for k = frame_start:frame_end
            eul = rotm2eul([Kinematics_i(k,4:6);Kinematics_i(k,7:9);Kinematics_i(k,10:12)]);
            EUL = [EUL;eul];
          end
          
          %only looking at the left hand
          Sample = [Kinematics_i(frame_start:frame_end,1:3),EUL,Kinematics_i(frame_start:frame_end,13:18)];
          %right hand: Kinematics_i(frame_start:frame_end,20:22),Kinematics_i(frame_start:frame_end,32:38)
          if(meta(i).experience == 'E')     %append sequence to output with a NaN row at the end
              Expert_Samples = [Expert_Samples;Sample];
              Expert_Samples = [Expert_Samples;nan(1,size(Expert_Samples,2))];  %delineate each sample
          else
              Novice_Samples = [Novice_Samples;Sample];
              Novice_Samples = [Novice_Samples;nan(1,size(Novice_Samples,2))];
          end     
       end
end

writematrix(Expert_Samples,'ExpertSamples' + Gesture + '.csv')
writematrix(Novice_Samples,'NoviceSamples' + Gesture + '.csv')
