addpath('D:\codes\Volvocine_PicoV2');
cd('D:\codes\Volvocine_PicoV2');

base_dir = fullfile('VolBotVideo');
conditions = {'baseline', 'sinz', 'm5sinz', 'm10sinz', 'moptz'};

% Load SVD weights once
svd_weights_file = fullfile('EstimateL', 'SStick', 'low_rank_analysis', 'M10', ...
    'global_joint_cp_rank1_profile_free_network_svd', 'agent_svd_contributions.csv');
weights_table = readtable(svd_weights_file);
agent_id_map = [9,8; 8,10; 11,7; 12,9];

summary_records = [];

% Check if existing matlab summary exists
out_csv = fullfile(base_dir, 'matlab_phase_energy_summary.csv');
if exist(out_csv, 'file')
    try
        T_existing = readtable(out_csv);
        summary_records = table2struct(T_existing);
        fprintf('[INFO] Loaded %d existing records from %s\n', numel(summary_records), out_csv);
    catch
        summary_records = [];
    end
end

fprintf('\n========================================================\n');
fprintf('  STARTING BATCH RELATIVE PHASE & ENERGY ANALYSIS\n');
fprintf('========================================================\n\n');

for c_idx = 1:numel(conditions)
    cond = conditions{c_idx};
    cond_dir = fullfile(base_dir, cond);
    if ~isfolder(cond_dir)
        continue;
    end
    
    subdirs = dir(cond_dir);
    subdirs = subdirs([subdirs.isdir] & ~startsWith({subdirs.name}, '.'));
    
    for s_idx = 1:numel(subdirs)
        t_name = subdirs(s_idx).name;
        if ~startsWith(t_name, 'GX')
            continue;
        end
        trial_dir = fullfile(cond_dir, t_name);
        
        csv_check = fullfile(trial_dir, 'phase_energy_analysis.csv');
        if exist(csv_check, 'file')
            fprintf('[SKIP] [%s] %s: phase_energy_analysis.csv already exists.\n', cond, t_name);
            continue;
        end
        
        fprintf('\n>>> Processing [%s] %s in %s ...\n', cond, t_name, trial_dir);
        close all;
        
        try
            % Call plot_relative_phase
            % Args: dirpath, rank=1, cut=5.0s, dur=300s, filter=true, win=1, saveFig=false, n_sync=1, m_sync=1, window=[15,25]
            [cluster_info, phase_series, energy_info] = plot_relative_phase( ...
                trial_dir, 1, 5.0, 300, true, 1, false, 1, 1, [15, 25], false, true, true);
            
            % Save figures
            figs = findall(0, 'Type', 'figure');
            fprintf('    Saving %d figure(s)...\n', numel(figs));
            
            for k = 1:numel(figs)
                fig = figs(k);
                f_title = get(fig, 'Name');
                
                if contains(f_title, 'Cumulative Energy')
                    fig_base = 'cumulative_energy';
                elseif contains(f_title, 'Weighted Order Parameter')
                    fig_base = 'weighted_order_parameter';
                elseif contains(f_title, 'Input Amplitude')
                    fig_base = 'weighted_input_amplitude';
                elseif contains(f_title, 'Chunk:')
                    fig_base = 'relative_phase';
                else
                    fig_base = sprintf('figure_%d', k);
                end
                
                png_path = fullfile(trial_dir, [fig_base, '.png']);
                exportgraphics(fig, png_path, 'Resolution', 200);
            end
            close all;
            
            % Compute First Mode Z1(t) time series
            agents = [phase_series{1}.agent_id];
            N = numel(agents);
            L = 4;
            V = zeros(N, L);
            for i = 1:N
                real_ag = agents(i);
                idx = find(agent_id_map(:,1) == real_ag, 1);
                mode_ag = agent_id_map(idx, 2);
                row_w = (weights_table.agent_id == mode_ag);
                for m = 1:L
                    col = sprintf('sender_v_mode%d', m);
                    V(i, m) = weights_table.(col)(row_w);
                end
            end
            
            t_common = phase_series{1}(1).time(:);
            phase_matrix = zeros(numel(t_common), N);
            for i = 1:N
                phase_matrix(:, i) = phase_series{1}(i).absolute_phase_unwrapped(:);
            end
            Z_complex = exp(1i * phase_matrix) * V;
            Z_abs = abs(Z_complex);
            z1 = Z_abs(:, 1);
            
            % Steady-state calculation (final 25% or last 15s of valid data)
            valid_mask = ~isnan(z1);
            t_valid = t_common(valid_mask);
            z1_valid = z1(valid_mask);
            
            if ~isempty(z1_valid)
                t_max_v = t_valid(end);
                steady_t_start = max(t_valid(1), t_max_v - 15.0);
                mask_steady = t_valid >= steady_t_start;
                
                z1_steady_mean = mean(z1_valid(mask_steady));
                z1_steady_std = std(z1_valid(mask_steady));
                z1_final = z1_valid(end);
                z1_overall_mean = mean(z1_valid);
            else
                z1_steady_mean = NaN;
                z1_steady_std = NaN;
                z1_final = NaN;
                z1_overall_mean = NaN;
            end
            
            % Energy metrics
            total_e_J = energy_info.total_energy_J;
            mean_p_W = energy_info.total_mean_power_W;
            dur_sec = energy_info.duration_sec;
            
            % Save trial MAT file
            mat_path = fullfile(trial_dir, 'phase_energy_analysis.mat');
            save(mat_path, 'cluster_info', 'phase_series', 'energy_info', 't_common', 'Z_abs', 'z1');
            
            % Append to summary
            record = struct( ...
                'trial', t_name, ...
                'duration_sec', dur_sec, ...
                'total_energy_J', total_e_J, ...
                'mean_power_W', mean_p_W, ...
                'mean_power_mW', mean_p_W * 1000, ...
                'mode1_converged_mean', z1_steady_mean, ...
                'mode1_steady_std', z1_steady_std, ...
                'mode1_final', z1_final, ...
                'mode1_overall_mean', z1_overall_mean ...
            );
            
            % If trial already exists in summary_records, update it; otherwise append
            existing_idx = [];
            if ~isempty(summary_records)
                existing_trials = {summary_records.trial};
                existing_idx = find(strcmp(existing_trials, t_name), 1);
            end
            if ~isempty(existing_idx)
                summary_records(existing_idx) = record;
            else
                summary_records = [summary_records; record]; %#ok<AGROW>
            end
            
            fprintf('    [OK] %s: Energy=%.2f J, Power=%.4f W, Mode1 Converged=%.4f (final=%.4f)\n', ...
                t_name, total_e_J, mean_p_W, z1_steady_mean, z1_final);
            
        catch ME
            fprintf('    [ERROR] Failed on %s: %s\n', t_name, ME.message);
            disp(getReport(ME));
        end
    end
end

% Save summary table
if ~isempty(summary_records)
    T_summary = struct2table(summary_records);
    writetable(T_summary, out_csv);
    fprintf('\nSaved summary table to: %s\n', out_csv);
end

fprintf('\n========================================================\n');
fprintf('  BATCH RELATIVE PHASE & ENERGY ANALYSIS COMPLETED!\n');
fprintf('========================================================\n');
