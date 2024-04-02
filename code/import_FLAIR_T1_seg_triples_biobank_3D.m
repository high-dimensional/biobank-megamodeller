%Code by Dr Robert Gray

% Import matched T1s, FLAIRs and segmentations of white matter lesions
% -> All volumes are affine-registered to MNI space
% -> Middle axial slice only
% -> Resized and cropped to 128x128

clear
clc
parpool(10)

% Hard code the paths
output_dir = '/media/robert/Data2/T1_FLAIR_seg_triples/';
temp_dir = '/media/robert/Data2/temp/';
flair_dir = '/home/robert/remote_mounts/brc_server_new/robert/biobank_FLAIR/';
t1_dir = '/home/robert/remote_mounts/brc_server_new/robert/biobank_T1/';

% List the files to extract
original_file_paths = dirrec(flair_dir, '.zip');
number_of_files = length(original_file_paths);
original_filenames = regexp(original_file_paths, "/", "split");
original_filenames = cellfun(@(x) x(8),original_filenames);

% List the T1s
t1_file_paths = dirrec(t1_dir, '.zip');
t1_file_paths = regexp(t1_file_paths, "/", "split");
t1_file_paths = cellfun(@(x) x(8),t1_file_paths);

% List the files that have already been extractedstandardise_flair_with_t1_stats
extracted_file_paths = dirrec(output_dir, '.nii');
number_of_extracted_files = length(extracted_file_paths);
extracted_filenames = regexp(extracted_file_paths, "/", "split");
extracted_filenames = cellfun(@(x) x(6),extracted_filenames);

% idx_usable = cell(number_of_files, 1);
% idx_usable_segs = cell(number_of_files, 1);

% standardise_flair_with_t1_stats = true;

parfor_progress(number_of_files); % Initialize
parfor i = 1:number_of_files
    current_filename = original_filenames{i};
    current_zip = original_file_paths{i};
    a = strfind(extracted_filenames, current_filename);
     
    if length(extracted_filenames) > 0 && length([a{:}]) > 0
%         disp("Skipping: file already extracted!");
    else 
        % disp(num2str(i));
        temp_dir_flair = [temp_dir, num2str(i), '_T2_FLAIR/'];
        temp_dir_t1 = [temp_dir, num2str(i), '_T1/'];
        temp_dir_seg = [temp_dir, num2str(i), '_SEG/'];

        % Empty the temp folder
        if exist(temp_dir_flair, 'dir')
            rmdir(temp_dir_flair, 's');
        end
        mkdir(temp_dir_flair);
        if exist(temp_dir_t1, 'dir')
            rmdir(temp_dir_t1, 's');
        end
        mkdir(temp_dir_t1);
        if exist(temp_dir_seg, 'dir')
            rmdir(temp_dir_seg, 's');
        end
        mkdir(temp_dir_seg);

        % Unzip the current FLAIR
        unzip(current_zip, temp_dir_flair);
   
        % Get eid
        temp = regexp(current_zip, '/', 'split');
        eid_with_ext = regexp(temp{8}, '_', 'split');
        eid = eid_with_ext{1};
        eids{i} = eid;

        % Check what FLAIR data was extracted
        contents_of_temp_dir_flair = dir([temp_dir_flair, 'T2_FLAIR']);
        contents_of_temp_dir_flair = {contents_of_temp_dir_flair.name};
        contents_of_temp_dir_flair(1:2)=[];

        if length(contents_of_temp_dir_flair) < 8
            % disp("-> Unusable: FLAIR missing");
            idx_usable{i} = 0;
            idx_usable_segs{i} = 0;
        else
            % Find corresponding T1
            t1_idx = find(contains(t1_file_paths, eid));
            if isempty(t1_idx)
                % disp("-> Unusable: T1 missing");
                idx_usable{i} = 0;
                idx_usable_segs{i} = 0;
            else
                current_t1_path = [t1_dir, t1_file_paths{t1_idx(1)}];
                unzip(current_t1_path, temp_dir_t1);

                % Check what T1 data was extracted
                contents_of_temp_dir_t1 = dir([temp_dir_t1, 'T1']);
                contents_of_temp_dir_t1 = {contents_of_temp_dir_t1.name};
                contents_of_temp_dir_t1(1:2)=[];

                if length(contents_of_temp_dir_t1) < 8
                    % disp("-> Unusable: T1 missing");
                    idx_usable{i} = 0;
                    idx_usable_segs{i} = 0;
                else
                    current_affine = [temp_dir_t1, 'T1/transforms/T1_to_MNI_linear.mat'];

                    % (1) Apply affine component of MNI registration to the FLAIR
                    current_flair_path = [temp_dir_flair, 'T2_FLAIR/', 'T2_FLAIR_unbiased_brain.nii.gz'];
                    current_flair_ref_path = [temp_dir_flair, 'T2_FLAIR/', 'T2_FLAIR_brain_to_MNI.nii.gz'];
                    current_flair_out_path = [temp_dir_flair, 'T2_FLAIR/', 'mni_T2_FLAIR_unbiased_brain.nii.gz'];
                    fsl_command_flair = ['flirt -in ', current_flair_path, ' -ref ', ...
                        current_flair_ref_path, ' -applyxfm -init ', current_affine, ...
                        ' -o ', current_flair_out_path];
                    status_vol = system(fsl_command_flair);
%                     system(fsl_command_flair);

                    % (2) Resize, crop and slice the FLAIR
                    flair = load_untouch_nii([[current_flair_out_path]]);
                    flair = flair.img;
                    flair( isnan(flair) ) = min(flair(:));
                    flair = single(flair);
                    flair = imresize3(flair, 0.7);
                    flair = squeeze(flair(:,1+12:128+12,:));

                    % Repeat (1) and (2), but for the T1
                    current_t1_path = [temp_dir_t1, 'T1/', 'T1_unbiased_brain.nii.gz'];
                    current_t1_ref_path = [temp_dir_t1, 'T1/', 'T1_brain_to_MNI.nii.gz'];
                    current_t1_out_path = [temp_dir_t1, 'T1/', 'mni_T1_unbiased_brain.nii.gz'];
                    fsl_command_t1 = ['flirt -in ', current_t1_path, ' -ref ', ...
                        current_t1_ref_path, ' -applyxfm -init ', current_affine, ...
                        ' -o ', current_t1_out_path];
                    status_vol = system(fsl_command_t1);
%                     system(fsl_command_t1);

                    t1 = load_untouch_nii([[current_t1_out_path]]);
                    t1 = t1.img;
                    t1( isnan(t1) ) = min(t1(:));
                    t1 = single(t1);
                    t1 = imresize3(t1, 0.7);
                    t1 = squeeze(t1(:,1+12:128+12,:));
                    
%                     % Now write both volumes to disk
%                     m = mean(t1(:));
%                     s = std(t1(:));
%                     t1 = (t1 - m) / s;
%                     if ~standardise_flair_with_t1_stats
%                         m = mean(flair(:));
%                         s = std(flair(:));
%                     end
%                     flair = (flair - m) / s;

                    niftiwrite(t1, [output_dir, current_filename, '_', num2str(i), '_t1.nii']);
                    niftiwrite(flair, [output_dir, current_filename, '_', num2str(i), '_flair.nii']); 
                    idx_usable{i} = 1;

                    % Now look for the segmentation
                    current_seg_path = [temp_dir_flair, 'T2_FLAIR/', 'lesions/final_mask.nii.gz'];
                    if isfile(current_seg_path)
                        % Repeat (1) and (2), but for the segmentation
                        current_seg_out_path = [temp_dir_flair, 'T2_FLAIR/', 'mni_final_mask.nii.gz'];
                        fsl_command_seg = ['flirt -in ', current_seg_path, ' -ref ', ...
                        current_flair_ref_path, ' -applyxfm -init ', current_affine, ...
                        ' -o ', current_seg_out_path];
                        status_vol = system(fsl_command_seg);
%                         system(fsl_command_seg);

                        current_seg = load_untouch_nii([[current_seg_out_path]]);
                        current_seg = current_seg.img;
                        current_seg( isnan(current_seg) ) = min(current_seg(:));
                        current_seg = abs(single(current_seg));
                        current_seg(current_seg > 0) = 1;
                        current_seg(current_seg < 1) = 0;
                        current_seg = imresize3(current_seg, 0.7, 'nearest');
                        current_seg = squeeze(current_seg(:,1+12:128+12,:));
                        current_seg(current_seg > 0) = 1;
                        current_seg(current_seg < 1) = 0;
                        niftiwrite(current_seg, [output_dir, current_filename, '_', num2str(i), '_seg.nii']);
                        idx_usable_segs{i} = 1;
                    end
                end
            end
        end
        if exist(temp_dir_flair, 'dir')
            rmdir(temp_dir_flair, 's');
        end
        if exist(temp_dir_t1, 'dir')
            rmdir(temp_dir_t1, 's');
        end
        if exist(temp_dir_seg, 'dir')
            rmdir(temp_dir_seg, 's');
        end
    end

parfor_progress; % Count
end
parfor_progress(0); % Clean up
delete(gcp('nocreate'));
