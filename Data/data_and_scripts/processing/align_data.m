function [ aligned_data, aligned_bin ] = align_data( data_struct )
%align_data aligns data to arbitrary bin on a trial-by-trial basis
%   Input: data_struct structure with fields 'data' and 'align_bin'. each
%   element of data_struct is a trial/row/instance to align

% Check inputs


%% Intialize
num_rows = length(data_struct);

% Find longest trial
longest_trial = 0;
for row = 1:num_rows
    trial_length = sum(~isnan(data_struct(row).data));
    if trial_length > longest_trial
        longest_trial = trial_length;
    end
end

aligned_data_temp = nan(num_rows,2*longest_trial); % just to pad ends for now. will trim later


%% Do it

% stick aligning bin right in the middle of aligned_data, then fill
% everything else in

for row = 1:num_rows
    align_bin = data_struct(row).align_bin;
    trial_length = sum(~isnan(data_struct(row).data));
    middle_bin = longest_trial + 1;
    start_bin = middle_bin - align_bin + 1;
    end_bin = start_bin + trial_length - 1;
    
    aligned_data_temp(row,start_bin:end_bin) = data_struct(row).data;    
end

%% Trim

column_sums = sum(isnan(aligned_data_temp),1);
first_column = find(column_sums < num_rows,1,'first');
last_column = find(column_sums < num_rows,1,'last');
temp = find(column_sums < num_rows);
aligned_data = aligned_data_temp(:,first_column:last_column);
aligned_bin = longest_trial + 1 - (first_column - 1);

end

