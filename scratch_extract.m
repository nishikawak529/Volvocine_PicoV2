% scratch_extract.m
% This script extracts separable profiles (a and b) for target agent 7,
% source agent 8 from the SVD cache and saves them to a CSV file.

try
    load('EstimateL/Round/low_rank_analysis/M10/svd_cache_8f8f8f523f8f614f8f8f5e13568f4558.mat');
    % Find target 7, source 8 with rank 5 (contains 5 SVD components)
    idx = find([all_results.target_agent_id] == 7 & [all_results.source_agent_id] == 8 & [all_results.rank] == 5, 1);
    if isempty(idx)
        error('Could not find SVD result for target 7, source 8 with rank 5.');
    end
    res = all_results(idx);
    comps = res.components;
    
    phi = comps(1).phi_grid(:);
    r1_a = real(comps(1).a_values(:));
    r1_b = real(comps(1).b_values(:));
    r2_a = real(comps(2).a_values(:));
    r2_b = real(comps(2).b_values(:));
    r3_a = real(comps(3).a_values(:));
    r3_b = real(comps(3).b_values(:));
    r4_a = real(comps(4).a_values(:));
    r4_b = real(comps(4).b_values(:));
    r5_a = real(comps(5).a_values(:));
    r5_b = real(comps(5).b_values(:));
    
    T = table(phi, r1_a, r1_b, r2_a, r2_b, r3_a, r3_b, r4_a, r4_b, r5_a, r5_b, ...
        'VariableNames', {'phi', ...
                          'rank1_a', 'rank1_b', ...
                          'rank2_a', 'rank2_b', ...
                          'rank3_a', 'rank3_b', ...
                          'rank4_a', 'rank4_b', ...
                          'rank5_a', 'rank5_b'});
                          
    out_dir = 'EstimateL/Round/low_rank_analysis/M10/pairwise_reconstructions/7-8';
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    csv_path = fullfile(out_dir, 'agent7_from_agent8_separable_profiles.csv');
    writetable(T, csv_path);
    fprintf('[INFO] Separable profiles successfully saved to: %s\n', csv_path);
catch ME
    fprintf('[ERROR] %s\n', ME.message);
end
