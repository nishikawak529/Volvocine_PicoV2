function all_joint_results = joint_svd_analysis_round(round_dir, M, varargin)
% Joint SVD (Low-Rank Joint Approximation) analysis for oscillator coupling matrices.
%
% For a target agent i, collects coupling matrices C_j->i for all source agents j,
% concatenates them column-wise, and performs a single SVD to extract:
%   1) A single target (incoming) profile a_r(phi_i) shared across all partners.
%   2) Partner-specific source (outgoing) profiles b_r^(j)(phi_j) tailored for each partner.
%
% Usage:
%   results = joint_svd_analysis_round();
%   results = joint_svd_analysis_round('EstimateL/Round', 10);
%

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'Round');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end

    opts = parse_options(varargin{:});
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');

    pair_infos = list_pair_folders(round_dir);
    if isempty(pair_infos)
        error('No pair folders like "7-8" were found under %s.', round_dir);
    end

    % Create output directory structure
    analysis_out_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), 'joint_svd_analysis');
    if ~exist(analysis_out_dir, 'dir')
        mkdir(analysis_out_dir);
    end

    % Check Joint SVD cache
    cache_key = compute_joint_cache_key(pair_infos, M, opts);
    cache_path = fullfile(analysis_out_dir, sprintf('joint_svd_cache_%s.mat', cache_key));
    
    is_cached = false;
    if opts.use_cache && isfile(cache_path)
        fprintf('[INFO] Loading cached Joint SVD results from %s\n', cache_path);
        load_data = load(cache_path);
        all_joint_results = load_data.all_joint_results;
        is_cached = true;
    end

    if ~is_cached
        % Start diary logging to a text file
        log_file_path = fullfile(analysis_out_dir, 'joint_svd_summary.txt');
        if exist(log_file_path, 'file'), delete(log_file_path); end
        diary(log_file_path);

        fprintf('[INFO] Starting Joint SVD Analysis on Round Data\n');
        fprintf('  Round directory: %s\n', round_dir);
        fprintf('  Output directory: %s\n', analysis_out_dir);
        fprintf('  Fourier order: M=%d\n', M);
        fprintf('  Remove self-only terms: %s\n', mat2str(opts.RemoveSelfOnly));
        fprintf('  Remove constant term: %s\n', mat2str(opts.RemoveConstant));
        fprintf('  Remove other-only terms: %s\n\n', mat2str(opts.RemoveOtherOnly));

        % Step 1: Collect coupling matrices C_analysis for all pairs
        pair_data_list = struct([]);
        for k = 1:numel(pair_infos)
            info = pair_infos(k);
            csv_pattern = fullfile(info.folder, '*.csv');
            fprintf('[INFO] Loading Pair %d/%d: %d-%d\n', k, numel(pair_infos), info.agent_ids(1), info.agent_ids(2));

            try
                pair_out = plot_phase_dynamics_from_csv(csv_pattern, info.agent_ids, M, ...
                    'analysis_start_sec', opts.analysis_start_sec, ...
                    'analysis_duration_sec', opts.analysis_duration_sec, ...
                    'sample_dt', opts.sample_dt, ...
                    'signal_column', opts.signal_column, ...
                    'normalize_signal', opts.normalize_signal, ...
                    'tail_percent', opts.tail_percent, ...
                    'clip_normalized_signal', opts.clip_normalized_signal, ...
                    'clip_limit', opts.clip_limit, ...
                    'plot_surfaces', false, ...
                    'plot_gamma', false, ...
                    'save_output', false, ...
                    'use_cache', opts.use_cache, ...
                    'cache_dir', opts.cache_dir, ...
                    'file_indices', opts.file_indices);
            catch ME
                warning('Skipping pair folder %s: %s', info.folder, ME.message);
                continue;
            end

            phi1_all = pair_out.point_cloud.phi1;
            phi2_all = pair_out.point_cloud.phi2;

            % Re-estimate coefficient matrices C_full for s1 and s2
            [C1_full, m_values, n_values] = estimate_fourier_coeff_matrix(phi1_all, phi2_all, pair_out.point_cloud.s1, M);
            [C2_full, ~, ~] = estimate_fourier_coeff_matrix(phi1_all, phi2_all, pair_out.point_cloud.s2, M);

            % Remove marginal terms if requested
            [C1_analysis, ~, ~] = remove_phase_marginal_terms(C1_full, phi1_all, phi2_all, pair_out.point_cloud.s1, m_values, n_values, ...
                info.agent_ids(1), info.agent_ids, opts.RemoveSelfOnly, opts.RemoveConstant, opts.RemoveOtherOnly);
            [C2_analysis, ~, ~] = remove_phase_marginal_terms(C2_full, phi1_all, phi2_all, pair_out.point_cloud.s2, m_values, n_values, ...
                info.agent_ids(2), info.agent_ids, opts.RemoveSelfOnly, opts.RemoveConstant, opts.RemoveOtherOnly);

            % Save to list
            entry = struct();
            entry.pair_name = info.name;
            entry.agent_ids = info.agent_ids;
            entry.C1 = C1_analysis; % target: agent_ids(1), source: agent_ids(2)
            entry.C2 = C2_analysis; % target: agent_ids(2), source: agent_ids(1)
            entry.m_values = m_values;
            entry.n_values = n_values;
            
            if isempty(pair_data_list)
                pair_data_list = entry;
            else
                pair_data_list(end+1) = entry; %#ok<AGROW>
            end
        end

        if isempty(pair_data_list)
            error('No valid pair data was loaded.');
        end

        % Get unique agent IDs across all loaded pairs
        all_ids = [];
        for k = 1:numel(pair_data_list)
            all_ids = [all_ids, pair_data_list(k).agent_ids]; %#ok<AGROW>
        end
        unique_agents = unique(all_ids);

        % Step 2: For each agent, build concatenated matrix and run SVD
        all_joint_results = struct([]);
        m_vals = pair_data_list(1).m_values;
        n_vals = pair_data_list(1).n_values;
        
        for a_idx = 1:numel(unique_agents)
            target_aid = unique_agents(a_idx);
            fprintf('[INFO] Processing Joint SVD for Agent %d...\n', target_aid);

            % Collect all matrices C_j->target
            % We keep track of the source agent ID for each concatenated block
            C_blocks = {};
            source_ids = [];
            associated_pairs = {};

            for k = 1:numel(pair_data_list)
                p_data = pair_data_list(k);
                if p_data.agent_ids(1) == target_aid
                    % target is agent_ids(1), source is agent_ids(2), matrix is C1
                    C_blocks{end+1} = p_data.C1; %#ok<AGROW>
                    source_ids(end+1) = p_data.agent_ids(2); %#ok<AGROW>
                    associated_pairs{end+1} = p_data.pair_name; %#ok<AGROW>
                elseif p_data.agent_ids(2) == target_aid
                    % target is agent_ids(2), source is agent_ids(1), matrix is C2
                    % Since C2 has rows = phi1 (source) and cols = phi2 (target),
                    % we must transpose it to make rows = target, cols = source.
                    C_blocks{end+1} = p_data.C2.'; %#ok<AGROW>
                    source_ids(end+1) = p_data.agent_ids(1); %#ok<AGROW>
                    associated_pairs{end+1} = p_data.pair_name; %#ok<AGROW>
                end
            end

            if isempty(C_blocks)
                fprintf('  No interactions found where Agent %d is the target. Skipping.\n', target_aid);
                continue;
            end

            % Concatenate columns: C_concat = [C_s1, C_s2, ...]
            C_concat = cat(2, C_blocks{:});
            
            % Compute Joint SVD
            [U, S, V] = svd(C_concat, 'econ');
            sigma = diag(S);
            energy = sum(sigma.^2);

            % Build separable profiles for rank 1, 2, 3
            phi_grid = linspace(0, 2*pi, 512).';
            components = struct('r', {}, 'a_values', {}, 'b_values_by_source', {}, 'sigma', {}, 'energy_ratio', {});
            
            n_ranks = min(opts.ProfileRank, numel(sigma));
            for r = 1:n_ranks
                sigma_r = sigma(r);
                alpha_r = sqrt(sigma_r) * U(:, r);
                
                % Phase align U based on the maximum real part of the 1st component
                if r == 1
                    a1_temp = exp(1i * phi_grid * m_vals(:).') * alpha_r;
                    [~, idx_max] = max(abs(a1_temp));
                    phase_shift = exp(-1i * angle(a1_temp(idx_max)));
                end
                
                alpha_r = phase_shift * alpha_r;
                a_values = exp(1i * phi_grid * m_vals(:).') * alpha_r;

                % Partition right singular vector V(:, r) into source blocks
                % V(:, r) has length (2M+1)*K, where K is numel(C_blocks)
                v_r = conj(phase_shift) * V(:, r);
                block_size = numel(n_vals);
                
                b_values_by_source = struct([]);
                for p = 1:numel(C_blocks)
                    start_idx = (p-1)*block_size + 1;
                    end_idx = p*block_size;
                    beta_r_p = sqrt(sigma_r) * conj(v_r(start_idx:end_idx)); % conjugate to match u_k * conj(v_k)
                    
                    b_vals_p = exp(1i * phi_grid * n_vals(:).') * beta_r_p;
                    
                    b_entry = struct();
                    b_entry.source_id = source_ids(p);
                    b_entry.pair_name = associated_pairs{p};
                    b_entry.b_values = real(b_vals_p(:));
                    
                    if isempty(b_values_by_source)
                        b_values_by_source = b_entry;
                    else
                        b_values_by_source(end+1) = b_entry; %#ok<AGROW>
                    end
                end

                comp_entry = struct();
                comp_entry.r = r;
                comp_entry.sigma = sigma_r;
                comp_entry.energy_ratio = (sigma_r^2) / energy;
                comp_entry.a_values = real(a_values(:));
                comp_entry.b_values_by_source = b_values_by_source;
                
                if isempty(components)
                    components = comp_entry;
                else
                    components(end+1) = comp_entry; %#ok<AGROW>
                end
            end

            % Log SVD eigenvalues
            fprintf('  Singular values:\n');
            for r = 1:min(5, numel(sigma))
                fprintf('    r=%d: val = %.5f, energy ratio = %.4f%%\n', ...
                    r, sigma(r), (sigma(r)^2 / energy) * 100);
            end
            fprintf('\n');

            % Save result entry
            res = struct();
            res.agent_id = target_aid;
            res.sigma = sigma;
            res.energy = energy;
            res.source_ids = source_ids;
            res.associated_pairs = associated_pairs;
            res.components = components;
            res.phi_grid = phi_grid;
            
            if isempty(all_joint_results)
                all_joint_results = res;
            else
                all_joint_results(end+1) = res; %#ok<AGROW>
            end
        end

        % Save SVD cache
        if opts.use_cache
            save(cache_path, 'all_joint_results');
            fprintf('[INFO] Saved Joint SVD cache to %s\n', cache_path);
        end

        diary('off');
        fprintf('[INFO] Saved summary log to %s\n', log_file_path);
    end

    % Step 3: Always plot and save figures grouped by agent ID
    self_profile_dir = fullfile(analysis_out_dir, 'agent_self_profiles');
    if ~exist(self_profile_dir, 'dir')
        mkdir(self_profile_dir);
    end

    for a_idx = 1:numel(all_joint_results)
        res = all_joint_results(a_idx);
        target_aid = res.agent_id;
        phi_grid = res.phi_grid;
        components = res.components;
        n_pairs = numel(res.source_ids);
        
        % Plotting Separable Profiles Overlay (3x2 tiled layout)
        fig_sep_overlay = figure('Color', 'w', 'Position', [100, 100, 1000, 800], ...
            'Name', sprintf('Joint SVD Separable Profiles - Agent %d', target_aid));
        t_lay = tiledlayout(fig_sep_overlay, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        title(t_lay, sprintf('Joint SVD Separable Profiles Overlay for Agent %d', target_aid), 'FontWeight', 'bold', 'FontSize', 12);
        
        % Left col (col 1): Incoming target profile a_r (phi_target) - same curve, just annotated
        % Right col (col 2): Paired incoming source profiles b_r^(j) (phi_source) for each partner
        
        for r = 1:3
            if r > numel(components)
                continue;
            end
            
            comp = components(r);
            
            % Left Plot: a_r (Target / Receiver)
            ax_in = nexttile(t_lay, (r - 1) * 2 + 1);
            hold(ax_in, 'on');
            grid(ax_in, 'on');
            box(ax_in, 'on');
            
            % Plot the single shared target profile
            plot(ax_in, phi_grid, comp.a_values, 'LineWidth', 2.5, 'Color', [0 0.447 0.741]);
            
            ylabel(ax_in, sprintf('a_%d (Target)', r), 'FontWeight', 'bold');
            if r == 3
                xlabel(ax_in, sprintf('\\phi_{%d} (Target Phase)', target_aid));
            end
            set(ax_in, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
            set(ax_in, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
            xlim(ax_in, [0, 2*pi]);
            legend(ax_in, {'Shared Target Profile'}, 'Location', 'best', 'FontSize', 8);
            title(ax_in, sprintf('%d-order Target Profile a_%d (var = %.1f%%)', r, r, comp.energy_ratio * 100));
            
            % Right Plot: Paired b_r^(j) (Source / Sender)
            ax_out = nexttile(t_lay, (r - 1) * 2 + 2);
            hold(ax_out, 'on');
            grid(ax_out, 'on');
            box(ax_out, 'on');
            
            colors = lines(n_pairs);
            legend_labels = {};
            for k = 1:n_pairs
                b_src = comp.b_values_by_source(k);
                plot(ax_out, phi_grid, b_src.b_values, 'LineWidth', 1.5, 'Color', colors(k, :));
                legend_labels{end+1} = sprintf('from Agent %d (Pair %s)', b_src.source_id, b_src.pair_name); %#ok<AGROW>
            end
            
            ylabel(ax_out, sprintf('b_%d (Paired Source)', r), 'FontWeight', 'bold');
            if r == 3
                xlabel(ax_out, '\phi_{source} (Source Phase)');
            end
            set(ax_out, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
            set(ax_out, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
            xlim(ax_out, [0, 2*pi]);
            if ~isempty(legend_labels)
                legend(ax_out, legend_labels, 'Location', 'best', 'FontSize', 8, 'Interpreter', 'none');
            end
            title(ax_out, sprintf('%d-order Paired Source Profiles b_%d', r, r));
        end
        
        sep_overlay_save_path = fullfile(self_profile_dir, sprintf('agent%d_joint_svd_profiles_overlay.png', target_aid));
        saveas(fig_sep_overlay, sep_overlay_save_path);
        fprintf('[INFO] Saved Joint SVD separable profiles overlay plot to: %s\n', sep_overlay_save_path);
        
        % Export profiles data as CSV
        t_data = table(phi_grid, 'VariableNames', {'phi'});
        for r = 1:min(3, numel(components))
            comp = components(r);
            t_data.(sprintf('a_%d_shared', r)) = comp.a_values(:);
            for k = 1:n_pairs
                b_src = comp.b_values_by_source(k);
                col_name = sprintf('b_%d_from_agent%d', r, b_src.source_id);
                t_data.(col_name) = b_src.b_values(:);
            end
        end
        csv_profile_path = fullfile(self_profile_dir, sprintf('agent%d_joint_svd_profiles_data.csv', target_aid));
        writetable(t_data, csv_profile_path);
        fprintf('[INFO] Saved Joint SVD profile CSV data to: %s\n', csv_profile_path);
        
        if ~opts.keep_figures
            close(fig_sep_overlay);
        end
    end

    fprintf('[INFO] Joint SVD Analysis completed. Output folder: %s\n', analysis_out_dir);
end

function opts = parse_options(varargin)
    p = inputParser;
    % Data extraction parameters
    addParameter(p, 'analysis_start_sec', 10, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
    addParameter(p, 'analysis_duration_sec', 80, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'sample_dt', 0.01, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'signal_column', 'a2', @(x) ischar(x) || isstring(x));
    addParameter(p, 'normalize_signal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'tail_percent', 10, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x < 50);
    addParameter(p, 'clip_normalized_signal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'clip_limit', 0.5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'use_cache', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'cache_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'file_indices', [], @(x) isempty(x) || isnumeric(x));

    % Joint SVD parameters
    addParameter(p, 'ProfileRank', 3, @isnumeric);
    addParameter(p, 'RemoveSelfOnly', true, @islogical);
    addParameter(p, 'RemoveConstant', true, @islogical);
    addParameter(p, 'RemoveOtherOnly', false, @islogical);
    addParameter(p, 'keep_figures', false, @islogical);

    parse(p, varargin{:});
    opts = p.Results;
    opts.signal_column = char(opts.signal_column);
    opts.normalize_signal = logical(opts.normalize_signal);
    opts.clip_normalized_signal = logical(opts.clip_normalized_signal);
    opts.use_cache = logical(opts.use_cache);
    opts.cache_dir = char(opts.cache_dir);
    opts.file_indices = double(opts.file_indices);
    opts.RemoveSelfOnly = logical(opts.RemoveSelfOnly);
    opts.RemoveConstant = logical(opts.RemoveConstant);
    opts.RemoveOtherOnly = logical(opts.RemoveOtherOnly);
    opts.keep_figures = logical(opts.keep_figures);
end

function pair_infos = list_pair_folders(round_dir)
    if isstring(round_dir), round_dir = char(round_dir); end
    pair_infos = struct([]);
    files = dir(round_dir);
    for i = 1:numel(files)
        if files(i).isdir && ~strcmp(files(i).name, '.') && ~strcmp(files(i).name, '..')
            % Check if folder name is like '7-8'
            if ~isempty(regexp(files(i).name, '^\d+-\d+$', 'once'))
                parts = strsplit(files(i).name, '-');
                ids = [str2double(parts{1}), str2double(parts{2})];
                entry = struct('name', files(i).name, 'folder', fullfile(round_dir, files(i).name), 'agent_ids', ids);
                if isempty(pair_infos)
                    pair_infos = entry;
                else
                    pair_infos(end+1) = entry; %#ok<AGROW>
                end
            end
        end
    end
    if ~isempty(pair_infos)
        [~, order] = sort({pair_infos.name});
        pair_infos = pair_infos(order);
    end
end

function [C, m_values, n_values] = estimate_fourier_coeff_matrix(phi1, phi2, y, M)
    m_values = (-M:M).';
    n_values = (-M:M).';

    term_m = repelem(m_values, numel(n_values));
    term_n = repmat(n_values, numel(m_values), 1);

    n_terms = numel(term_m);
    n_samples = numel(y);

    G = complex(zeros(n_terms, n_terms));
    h = complex(zeros(n_terms, 1));
    chunk_size = 20000;

    for start_idx = 1:chunk_size:n_samples
        end_idx = min(start_idx + chunk_size - 1, n_samples);
        sub_phi1 = phi1(start_idx:end_idx);
        sub_phi2 = phi2(start_idx:end_idx);
        sub_y = y(start_idx:end_idx);

        A_chunk = exp(1i * (sub_phi1 * term_m.' + sub_phi2 * term_n.'));
        G = G + A_chunk' * A_chunk;
        h = h + A_chunk' * sub_y;
    end

    lambda = 1e-4 * n_samples;
    coeff = (G + lambda * eye(n_terms)) \ h;
    C = reshape(coeff, numel(n_values), numel(m_values)).';
end

function [C_out, y_rec, info] = remove_phase_marginal_terms(C, phi1, phi2, y, m_values, n_values, target_agent_id, phase_agent_ids, remove_self, remove_constant, remove_other)
    term_m = repelem(m_values(:), numel(n_values));
    term_n = repmat(n_values(:), numel(m_values), 1);
    coeff_vector = reshape(C.', [], 1);
    total_energy = sum(abs(coeff_vector).^2);

    is_self_term = false(size(coeff_vector));
    is_constant_term = (term_m == 0 & term_n == 0);
    is_other_term = false(size(coeff_vector));

    if phase_agent_ids(1) == target_agent_id
        is_self_term = (term_m ~= 0 & term_n == 0);
        is_other_term = (term_m == 0 & term_n ~= 0);
    else
        is_self_term = (term_m == 0 & term_n ~= 0);
        is_other_term = (term_m ~= 0 & term_n == 0);
    end

    mask_to_remove = false(size(coeff_vector));
    if remove_self, mask_to_remove = mask_to_remove | is_self_term; end
    if remove_constant, mask_to_remove = mask_to_remove | is_constant_term; end
    if remove_other, mask_to_remove = mask_to_remove | is_other_term; end

    removed_self_energy = sum(abs(coeff_vector(is_self_term)).^2);
    removed_constant_energy = sum(abs(coeff_vector(is_constant_term)).^2);
    removed_other_energy = sum(abs(coeff_vector(is_other_term)).^2);

    coeff_analysis = coeff_vector;
    coeff_analysis(mask_to_remove) = 0;
    C_out = reshape(coeff_analysis, numel(n_values), numel(m_values)).';

    A = exp(1i * (phi1(:) * term_m.' + phi2(:) * term_n.'));
    y_rec = real(A * coeff_analysis);

    info = struct();
    info.total_coeff_energy = total_energy;
    info.removed_self_energy_ratio = removed_self_energy / max(total_energy, eps);
    info.removed_constant_energy_ratio = removed_constant_energy / max(total_energy, eps);
    info.removed_other_energy_ratio = removed_other_energy / max(total_energy, eps);
    
    removed_total_energy = sum(abs(coeff_vector(mask_to_remove)).^2);
    info.removed_total_coeff_energy = removed_total_energy;
    info.removed_total_energy_ratio = removed_total_energy / max(total_energy, eps);
    info.remaining_energy_ratio = 1 - info.removed_total_energy_ratio;
    info.reconstruction_mismatch_rmse = sqrt(mean((y(:) - real(A * coeff_vector)).^2));
end

function cache_key = compute_joint_cache_key(pair_infos, M, opts)
    parts = {};
    parts{end + 1} = 'version=joint_svd_v1';
    parts{end + 1} = sprintf('M=%d', M);
    parts{end + 1} = sprintf('analysis_start_sec=%.15g', opts.analysis_start_sec);
    parts{end + 1} = sprintf('analysis_duration_sec=%.15g', opts.analysis_duration_sec);
    parts{end + 1} = sprintf('sample_dt=%.15g', opts.sample_dt);
    parts{end + 1} = sprintf('signal_column=%s', opts.signal_column);
    parts{end + 1} = sprintf('normalize_signal=%d', opts.normalize_signal);
    parts{end + 1} = sprintf('tail_percent=%.15g', opts.tail_percent);
    parts{end + 1} = sprintf('clip_normalized_signal=%d', opts.clip_normalized_signal);
    parts{end + 1} = sprintf('clip_limit=%.15g', opts.clip_limit);
    parts{end + 1} = sprintf('file_indices=%s', mat2str(opts.file_indices));
    parts{end + 1} = sprintf('RemoveSelfOnly=%d', opts.RemoveSelfOnly);
    parts{end + 1} = sprintf('RemoveConstant=%d', opts.RemoveConstant);
    parts{end + 1} = sprintf('RemoveOtherOnly=%d', opts.RemoveOtherOnly);
    
    for i = 1:numel(pair_infos)
        csv_files = dir(fullfile(pair_infos(i).folder, '*.csv'));
        for j = 1:numel(csv_files)
            parts{end + 1} = sprintf('%s|%d|%.15g', csv_files(j).name, csv_files(j).bytes, csv_files(j).datenum); %#ok<AGROW>
        end
    end
    
    cache_key = md5_hex(strjoin(parts, '\n'));
end

function hex_text = md5_hex(text_value)
    import java.security.MessageDigest;
    import java.lang.String;
    
    md = MessageDigest.getInstance('MD5');
    bytes_data = md.digest(double(String(text_value).getBytes('UTF-8')));
    
    hex_chars = '0123456789abcdef';
    hex_len = numel(bytes_data) * 2;
    hex_str = char(zeros(1, hex_len));
    for idx = 1:numel(bytes_data)
        b = bytes_data(idx);
        if b < 0
            b = b + 256;
        end
        hi = floor(b / 16) + 1;
        lo = mod(b, 16) + 1;
        hex_str(idx*2 - 1) = hex_chars(hi);
        hex_str(idx*2) = hex_chars(lo);
    end
    hex_text = hex_str;
end
