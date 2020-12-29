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
%% Plot stuff
close all
%plot one kinematics of one gesture for all relevant surgeons
gesture_number = 4;
Gesture = 'G' + string(gesture_number);  %enter number of gesture

balance = 40;
gestures_E = 0;
gestures_N = 0;

for i = 1:num_surgeons
       color1 = [0.2+rand(1)*0.3 0.8+rand(1)*0.2 0.2+rand(1)*0.3];
       color2 = [0.8+rand(1)*0.2 0.2+rand(1)*0.3 0.2+rand(1)*0.3];
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
          
                    %try to rotate gestures to line up
%           axis = cross([1 0 0],xyz_j(10,:));
%           angle = real(acos(max(min(dot([1 0 0],xyz_j(10,:))/(norm([1 0 0])*norm(xyz_j(10,:))),1),-1)));
%           R = Rot(transpose(axis),angle);
%         xyz_j = transpose(R*transpose(xyz_j));
          

          xyz_l = Kinematics_i(frame_start:frame_end,1:3)-repmat(Kinematics_i(frame_start,1:3),frame_end-frame_start+1,1);
          xyz_r = Kinematics_i(frame_start:frame_end,20:22)-repmat(Kinematics_i(frame_start,20:22),frame_end-frame_start+1,1);
          vel_l = Kinematics_i(frame_start:frame_end,13:15);
          vel_r = Kinematics_i(frame_start:frame_end,32:34);
          vel_l_rot = Kinematics_i(frame_start:frame_end,16:18);
          vel_r_rot = Kinematics_i(frame_start:frame_end,35:37);  
          
           vel_scale = 0.1;

           if(meta(i).experience == 'E' && size(xyz_l,1) > 0 && gestures_E < balance)  %if expert
               subplot(2, 3, 1);
               plot3(xyz_l(:,1)+0.05,xyz_l(:,2),xyz_l(:,3),'Color',color1);
               hold on
               subplot(2, 3, 2);
               plot3(vel_l(:,1)+0.2,vel_l(:,2),vel_l(:,3),'Color',color1);
               hold on
               subplot(2, 3, 3);
               plot3(vel_l_rot(:,1)+8,vel_l_rot(:,2),vel_l_rot(:,3),'Color',color1);
               %quiver3(xyz_l(:,1)+0.05,xyz_l(:,2),xyz_l(:,3),vel_l(:,1)*vel_scale,vel_l(:,2)*vel_scale,vel_l(:,3)*vel_scale,'Color',color1,'AutoScale','on');
               hold on
               subplot(2, 3, 4);
               plot3(xyz_r(:,1)+0.05,xyz_r(:,2),xyz_r(:,3),'Color',color1);
               hold on
               subplot(2, 3, 5);
               plot3(vel_r(:,1)+0.2,vel_r(:,2),vel_r(:,3),'Color',color1);
               hold on
               subplot(2, 3, 6);
               plot3(vel_r_rot(:,1)+8,vel_r_rot(:,2),vel_r_rot(:,3),'Color',color1);
               %quiver3(xyz_r(:,1)+0.05,xyz_r(:,2),xyz_r(:,3),vel_r(:,1)*vel_scale,vel_r(:,2)*vel_scale,vel_r(:,3)*vel_scale,'Color',color1,'AutoScale','on');
               hold on          
               gestures_E = gestures_E + 1;
           end

            
           
           if(meta(i).experience =='N' && size(xyz_l,1) > 0 && gestures_N < balance)  %if novice

               subplot(2, 3, 1);
               plot3(xyz_l(:,1),xyz_l(:,2),xyz_l(:,3),'Color',color2);
               hold on
               subplot(2, 3, 2);
               plot3(vel_l(:,1),vel_l(:,2),vel_l(:,3),'Color',color2);
               hold on
               subplot(2, 3, 3);
               plot3(vel_l_rot(:,1),vel_l_rot(:,2),vel_l_rot(:,3),'Color',color2);
               %quiver3(xyz_l(:,1),xyz_l(:,2),xyz_l(:,3),vel_l(:,1)*vel_scale,vel_l(:,2)*vel_scale,vel_l(:,3)*vel_scale,'Color',color2,'AutoScale','off');
               hold on
               subplot(2, 3, 4);
               plot3(xyz_r(:,1),xyz_r(:,2),xyz_r(:,3),'Color',color2);
               hold on
               subplot(2, 3, 5);
               plot3(vel_r(:,1),vel_r(:,2),vel_r(:,3),'Color',color2);
               hold on
               subplot(2, 3, 6);
               plot3(vel_r_rot(:,1),vel_r_rot(:,2),vel_r_rot(:,3),'Color',color2);
               %quiver3(xyz_r(:,1),xyz_r(:,2),xyz_r(:,3),vel_r(:,1)*vel_scale,vel_r(:,2)*vel_scale,vel_r(:,3)*vel_scale,'Color',color2,'AutoScale','off');
               hold on 
               gestures_N = gestures_N + 1;
           end
       end
end

disp(gestures_N - gestures_E);

sgtitle("Experts vs Novice Master Tooltip Kinematics on Gesture " + string(gesture_number));
subplot(2, 3, 1);
title('L Position');
subplot(2, 3, 2);
title('L Velocity (Translational)');
subplot(2, 3, 3);
title('L Velocity (Rotational)');
subplot(2, 3, 4);
title('R Position');
subplot(2, 3, 5);
title('R Velocity (Translational)');
subplot(2, 3, 6);
title('R Velocity (Rotational)');


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
