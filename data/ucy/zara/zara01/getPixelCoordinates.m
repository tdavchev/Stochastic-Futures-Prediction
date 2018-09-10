fid = fopen('crowds_zara01.vsp');

% First line contains the number of splines (or pedestrians?)
tline = fgetl(fid);
components = strsplit(tline, ' - ');
numSplines = str2num(cell2mat(components(1)));

pixel_pos_frame_ped = [];

minDiff = Inf;
for spline=1:numSplines
   
    % First line for each spline contains the number of control
    % points
    tline = fgetl(fid);
    components = strsplit(tline, ' - ');
    numPoints = str2num(cell2mat(components(1)));
    oldframe=0;
    for point=1:numPoints                        
        % Each line contains the information x, y, frame_number,
        % gaze(not needed)
        tline = fgetl(fid);
        components = strsplit(tline, ' ');
        x = str2num(cell2mat(components(1)));
        y = str2num(cell2mat(components(2)));
        frame = str2num(cell2mat(components(3)));
        
        if point ~= 1
            minDiff = min(minDiff, frame-oldframe);
        end
        oldframe = frame;
        % Putting x, y in reverse order
        vec = [frame; spline; 288-y; x+360];
        pixel_pos_frame_ped = [pixel_pos_frame_ped vec];
        
    end    
end
fprintf('The min diff between subsequent frames is %f\n', minDiff);

% NOTE x, y coordinates are recorded assuming center of the frame
% is (0,0)

[Y,I] = sort(pixel_pos_frame_ped(1,:));
pixel_pos_frame_ped_sorted = pixel_pos_frame_ped(:,I);
pixel_pos_frame_ped = pixel_pos_frame_ped_sorted;


pixel_pos_frame_ped(1, :) = floor(pixel_pos_frame_ped(1, :) / 8);

pixel_pos_frame_ped(3,:) = pixel_pos_frame_ped(3,:) / 576;
pixel_pos_frame_ped(4,:) = pixel_pos_frame_ped(4,:) / 720;

save('pixel_pos.mat', 'pixel_pos_frame_ped');
csvwrite('pixel_pos.csv', pixel_pos_frame_ped);
