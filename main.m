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
meta = cell2struct( struct2cell(meta), {'filename', 'skill_self', 'skill_GRS', 'Score1','Score2','Score3','Score4','Score5','Score6'});
num_surgeons = size(meta,1);

%% Import kinematics data
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
%% Visualize something

%plot one kinematics of one gesture for all relevant surgeons

gesture_number = 2;
Gesture = 'G' + string(gesture_number);  %enter number of gesture

for i = 1:num_surgeons
       Kinematics_i = K(i);   %Get ith kinematics file
       Transcription_i = T(i);   %Get ith transcription file
       index = find(strcmp(Transcription_i(:,3), Gesture));   % Get frame stamps for the gesture
       xyz_l = [];
       vel_l = [];
       xyz_r = [];
       vel_r = [];
       for j = 1:size(index,1)
          frame_start = cell2mat(Transcription_i(index(j),1));
          frame_end = cell2mat(Transcription_i(index(j),2));
          
          xyz_l_j = Kinematics_i(frame_start:frame_end,39:41)-repmat(Kinematics_i(frame_start,39:41),frame_end-frame_start+1,1);
          xyz_r_j = Kinematics_i(frame_start:frame_end,58:60)-repmat(Kinematics_i(frame_start,58:60),frame_end-frame_start+1,1);
          
          %try to rotate gestures to line up
%           axis = cross([1 0 0],xyz_j(10,:));
%           angle = real(acos(max(min(dot([1 0 0],xyz_j(10,:))/(norm([1 0 0])*norm(xyz_j(10,:))),1),-1)));
%           R = Rot(transpose(axis),angle);
%         xyz_j = transpose(R*transpose(xyz_j));
          
          xyz_l = [xyz_l;xyz_l_j];
          vel_l = [vel_l;Kinematics_i(frame_start:frame_end,51:53)];
          xyz_r = [xyz_r;xyz_r_j];
          vel_r = [vel_r;Kinematics_i(frame_start:frame_end,70:72)];
       end
            
       vel_scale = 0.1;
       if(meta(i).Score2 >=4 && size(xyz_l,1) > 0)  %if expert
           subplot(2, 3, 1);
           plot3(xyz_l(:,1),xyz_l(:,2),xyz_l(:,3),'g');
           hold on
           subplot(2, 3, 2);
           plot3(vel_l(:,1),vel_l(:,2),vel_l(:,3),'g');
           hold on
           subplot(2, 3, 3);
           quiver3(xyz_l(:,1),xyz_l(:,2),xyz_l(:,3),vel_l(:,1)*vel_scale,vel_l(:,2)*vel_scale,vel_l(:,3)*vel_scale,'g','AutoScale','on');
           hold on
           subplot(2, 3, 4);
           plot3(xyz_r(:,1),xyz_r(:,2),xyz_r(:,3),'g');
           hold on
           subplot(2, 3, 5);
           plot3(vel_r(:,1),vel_r(:,2),vel_r(:,3),'g');
           hold on
           subplot(2, 3, 6);
           quiver3(xyz_r(:,1),xyz_r(:,2),xyz_r(:,3),vel_r(:,1)*vel_scale,vel_r(:,2)*vel_scale,vel_r(:,3)*vel_scale,'g','AutoScale','on');
           hold on          
       end
       
       if(meta(i).Score2 <= 2 && size(xyz_l,1) > 0)  %if novice
           subplot(2, 3, 1);
           plot3(xyz_l(:,1),xyz_l(:,2),xyz_l(:,3),'r');
           hold on
           subplot(2, 3, 2);
           plot3(vel_l(:,1),vel_l(:,2),vel_l(:,3),'r');
           hold on
           subplot(2, 3, 3);
           quiver3(xyz_l(:,1),xyz_l(:,2),xyz_l(:,3),vel_l(:,1)*vel_scale,vel_l(:,2)*vel_scale,vel_l(:,3)*vel_scale,'r','AutoScale','off');
           hold on
           subplot(2, 3, 4);
           plot3(xyz_r(:,1),xyz_r(:,2),xyz_r(:,3),'r');
           hold on
           subplot(2, 3, 5);
           plot3(vel_r(:,1),vel_r(:,2),vel_r(:,3),'r');
           hold on
           subplot(2, 3, 6);
           quiver3(xyz_r(:,1),xyz_r(:,2),xyz_r(:,3),vel_r(:,1)*vel_scale,vel_r(:,2)*vel_scale,vel_r(:,3)*vel_scale,'r','AutoScale','off');
           hold on                     
       end
       sgtitle("Experts vs Novice Slave Tooltip Kinematics on Gesture " + string(gesture_number));
       subplot(2, 3, 1);
       title('L Position');
       subplot(2, 3, 2);
       title('L Velocity (Translational)');
       subplot(2, 3, 3);
       title('L Position & Velocity');
       subplot(2, 3, 4);
       title('R Position');
       subplot(2, 3, 5);
       title('R Velocity (Translational)');
       subplot(2, 3, 6);
       title('R Position & Velocity');
end




% N = [];
% I = [];
% E = [];
% for i = 1:num_surgeons
%     if meta(i).skill_self == 'N'
%         N = [N, meta(i).skill_GRS];
%     end
%     if meta(i).skill_self == 'I'
%         I = [I, meta(i).skill_GRS];
%     end
%     if meta(i).skill_self == 'E'
%         E = [E, meta(i).skill_GRS];
%     end
% end
% 
% hAxes = axes('NextPlot','add',...           %# Add subsequent plots to the axes,
%              'DataAspectRatio',[1 1 1],...  %#   match the scaling of each axis,
%              'XLim',[0 30],...               %#   set the x axis limit,
%              'YLim',[0 eps],...             %#   set the y axis limit (tiny!),
%              'Color','none');               %#   and don't use a background color
% plot(N,0,'r+',I,0,'gx',E,0,'bo','MarkerSize',10);  %# Plot data set 1
% hold on
% title("Self Proclaimed Skill vs GRS Skill");
% %legend('Novice','Intermediate','Expert');
