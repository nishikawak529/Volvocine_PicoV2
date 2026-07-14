function all_global_results = global_joint_svd_analysis(round_dir, M, varargin)
% Global Joint SVD (Low-Rank Joint Approximation for Whole Network via column-wise concatenation).
%
% Concatenates ALL 12 coupling matrices (for 4 agents, 6 pairs, 2 directions each) 
% column-wise to form a single matrix C_global_concat of size (2M+1) x 12*(2M+1):
%
%   C_global_concat = [ C_{8->7}, C_{9->7}, C_{10->7}, C_{7->8}, C_{9->8}, C_{10->8}, ... ]
%
% Performs a single SVD on C_global_concat to extract:
%   - A single system-wide target (receiver) profile a_r(phi) shared across all agents and pairs.
%   - 12 distinct source (sender) profiles b_r^(p)(phi) corresponding to each interaction pair and direction.
%
% Usage:
%   results = global_joint_svd_analysis();
%   results = global_joint_svd_analysis('EstimateL/Round', 10);
%

    % =========================================================================
    % CONFIGURATION: Digraph edge color and thickness limits
    % =========================================================================
    % Fixed upper/lower bounds for coupling strength color mapping.
    % The colormap range will be [-clim_limit, clim_limit] to allow comparison.
    clim_limit = 0.06;
    
    % Fixed maximum weight limit for coupling strength line thickness.
    % The edge line thickness is normalized in [0, linewidth_limit].
    linewidth_limit = 0.06;
    % =========================================================================

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'SStick');
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
    analysis_out_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M), 'global_joint_svd');
    if ~exist(analysis_out_dir, 'dir')
        mkdir(analysis_out_dir);
    end

    % Start diary logging to a text file
    log_file_path = fullfile(analysis_out_dir, 'global_svd_summary.txt');
    if exist(log_file_path, 'file'), delete(log_file_path); end
    diary(log_file_path);

    fprintf('[INFO] Starting Global Joint SVD (Column Concatenation) Analysis on Network Data\n');
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
            [C2_full, ~, ~] = estimate_fourier_coeff_matrix(phi2_all, phi1_all, pair_out.point_cloud.s2, M);

            % Remove marginal terms if requested
            [C1_analysis, ~, ~] = remove_phase_marginal_terms(C1_full, phi1_all, phi2_all, pair_out.point_cloud.s1, m_values, n_values, ...
                info.agent_ids(1), info.agent_ids, opts.RemoveSelfOnly, opts.RemoveConstant, opts.RemoveOtherOnly);
            [C2_analysis, ~, ~] = remove_phase_marginal_terms(C2_full, phi2_all, phi1_all, pair_out.point_cloud.s2, m_values, n_values, ...
                info.agent_ids(2), [info.agent_ids(2), info.agent_ids(1)], opts.RemoveSelfOnly, opts.RemoveConstant, opts.RemoveOtherOnly);

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

        % Step 2: Build Global Concatenated Matrix C_global_concat
        % Collect all 12 blocks, and keep track of meta-information
        C_blocks = {};
        interaction_meta = struct('pair_name', {}, 'target_id', {}, 'source_id', {}, 'direction', {});

        for k = 1:numel(pair_data_list)
            p_data = pair_data_list(k);
            
            % Block 1: C1 (target: agent_ids(1), source: agent_ids(2))
            C_blocks{end+1} = p_data.C1; %#ok<AGROW>
            meta = struct('pair_name', p_data.pair_name, 'target_id', p_data.agent_ids(1), ...
                          'source_id', p_data.agent_ids(2), 'direction', 's1');
            if isempty(interaction_meta)
                interaction_meta = meta;
            else
                interaction_meta(end+1) = meta; %#ok<AGROW>
            end

            % Block 2: C2 (target: agent_ids(2), source: agent_ids(1))
            % Since C2 is already estimated with rows = target, cols = source,
            % we do not transpose it.
            C_blocks{end+1} = p_data.C2; %#ok<AGROW>
            meta = struct('pair_name', p_data.pair_name, 'target_id', p_data.agent_ids(2), ...
                          'source_id', p_data.agent_ids(1), 'direction', 's2');
            interaction_meta(end+1) = meta; %#ok<AGROW>
        end

        % Concatenate all columns
        C_global_concat = cat(2, C_blocks{:});
        n_blocks = numel(C_blocks);

        % Extract all unique agent IDs
        unique_agents = unique([ [interaction_meta.target_id], [interaction_meta.source_id] ]);

        % Step 3: Compute SVD on C_global_concat
        [U, S, V] = svd(C_global_concat, 'econ');
        sigma = diag(S);
        energy = sum(sigma.^2);

        % Log SVD Singular Values
        fprintf('[INFO] Global Concatenated SVD Singular Values:\n');
        for r = 1:min(10, numel(sigma))
            fprintf('  r=%d: val = %.5f, energy ratio = %.4f%%\n', ...
                r, sigma(r), (sigma(r)^2 / energy) * 100);
        end
        fprintf('\n');

        % Step 4: Reconstruct shared receiver profile a_r(phi) and interaction-specific b_r^(p)(phi)
        phi_grid = linspace(0, 2*pi, 512).';
        m_vals = pair_data_list(1).m_values;
        n_vals = pair_data_list(1).n_values;
        block_size = numel(n_vals);
        n_ranks = min(opts.ProfileRank, numel(sigma));

        components = struct('r', {}, 'a_values', {}, 'b_values_by_interaction', {}, 'sigma', {}, 'energy_ratio', {});

        for r = 1:n_ranks
            sigma_r = sigma(r);
            alpha_r = sqrt(sigma_r) * U(:, r);

            % Phase align U based on the maximum real part of the r-th component
            a_temp = exp(1i * phi_grid * m_vals(:).') * alpha_r;
            [~, idx_max] = max(abs(a_temp));
            phase_shift_r = exp(-1i * angle(a_temp(idx_max)));

            alpha_r = phase_shift_r * alpha_r;
            a_values = exp(1i * phi_grid * m_vals(:).') * alpha_r;

            b_values_by_interaction = struct([]);

            for p = 1:n_blocks
                start_idx = (p-1)*block_size + 1;
                end_idx = p*block_size;
                
                % Direct phase-compensated SVD projection to match U_r * V_r^H
                v_block = V(start_idx:end_idx, r);
                beta_r_p = sqrt(sigma_r) * conj(phase_shift_r) * conj(v_block);

                b_vals_p = exp(1i * phi_grid * n_vals(:).') * beta_r_p;

                b_entry = struct();
                b_entry.pair_name = interaction_meta(p).pair_name;
                b_entry.target_id = interaction_meta(p).target_id;
                b_entry.source_id = interaction_meta(p).source_id;
                b_entry.direction = interaction_meta(p).direction;
                b_entry.b_values = real(b_vals_p(:));
                b_entry.beta_coeff = beta_r_p(:); % Save raw coefficients for 2nd-stage SVD

                if isempty(b_values_by_interaction)
                    b_values_by_interaction = b_entry;
                else
                    b_values_by_interaction(end+1) = b_entry; %#ok<AGROW>
                end
            end

            comp_entry = struct();
            comp_entry.r = r;
            comp_entry.sigma = sigma_r;
            comp_entry.energy_ratio = (sigma_r^2) / energy;
            comp_entry.a_values = real(a_values(:));
            comp_entry.alpha_coeff = alpha_r(:); % Save complex coefficients
            comp_entry.b_values_by_interaction = b_values_by_interaction;

            if isempty(components)
                components = comp_entry;
            else
                components(end+1) = comp_entry; %#ok<AGROW>
            end
        end

        % --- Diagnostic check before normalization ---
        fprintf('[INFO] Verifying pre-normalization SVD reconstruction...\n');
        for r = 1:n_ranks
            for p = 1:n_blocks
                alpha_coeff = components(r).alpha_coeff;
                beta_coeff = components(r).b_values_by_interaction(p).beta_coeff;
                C_from_profiles = alpha_coeff * beta_coeff.';
                
                block_indices = (p-1)*block_size + 1 : p*block_size;
                C_from_svd = sigma(r) * U(:, r) * V(block_indices, r)';
                
                rel_err = norm(C_from_profiles - C_from_svd, 'fro') / max(norm(C_from_svd, 'fro'), eps);
                if rel_err > 1e-12
                    fprintf('[WARNING] Pre-norm discrepancy in Rank %d, Block %d: %.3e\n', r, p, rel_err);
                end
            end
        end

        % --- 1. Target (Receiver) Profile a_1 Normalization ---
        % Extract raw receiver profile a_1
        comp_r1 = components(1);
        a_r1_raw = comp_r1.a_values;
        scale_a = sqrt(mean(a_r1_raw.^2));
        if scale_a == 0, scale_a = eps; end
        
        % Normalize receiver profile to have mean power (RMS squared) = 1
        a_r1_norm = a_r1_raw / scale_a;
        alpha_r1_norm = comp_r1.alpha_coeff / scale_a;
        
        % Update components(1) with normalized receiver profile
        components(1).a_values = a_r1_norm;
        components(1).alpha_coeff = alpha_r1_norm;

        % Update stored sender profiles b_1^(p) to keep the product invariant
        for p = 1:n_blocks
            beta_scaled = components(1).b_values_by_interaction(p).beta_coeff * scale_a;
            components(1).b_values_by_interaction(p).beta_coeff = beta_scaled;
            components(1).b_values_by_interaction(p).b_values = real( ...
                exp(1i * phi_grid * n_vals(:).') * beta_scaled ...
            );
        end

        % --- Diagnostic check after normalization ---
        fprintf('[INFO] Verifying post-normalization SVD reconstruction...\n');
        for r = 1:n_ranks
            for p = 1:n_blocks
                alpha_coeff = components(r).alpha_coeff;
                beta_coeff = components(r).b_values_by_interaction(p).beta_coeff;
                C_from_profiles = alpha_coeff * beta_coeff.';
                
                block_indices = (p-1)*block_size + 1 : p*block_size;
                C_from_svd = sigma(r) * U(:, r) * V(block_indices, r)';
                
                rel_err = norm(C_from_profiles - C_from_svd, 'fro') / max(norm(C_from_svd, 'fro'), eps);
                if rel_err > 1e-12
                    fprintf('[WARNING] Post-norm discrepancy in Rank %d, Block %d: %.3e\n', r, p, rel_err);
                end
            end
        end

        % --- 2. Construct B matrix for r=1 (scale automatically determined by a_1 normalization) ---
        % The scale of individual b_1^(p) is scaled up by scale_a to keep the product invariant
        B_coeff = zeros(block_size, n_blocks);
        for p = 1:n_blocks
            B_coeff(:, p) = components(1).b_values_by_interaction(p).beta_coeff;
        end
        
        % --- 3. Second-Stage SVD for scale-determined sender profiles ---
        [U_B, S_B, V_B] = svd(B_coeff, 'econ');
        
        % Phase align beta_shared so its real part waveform is maximized at peak
        beta_shared = U_B(:, 1);
        b_shared_temp = exp(1i * phi_grid * n_vals(:).') * beta_shared;
        [~, idx_max_B] = max(abs(b_shared_temp));
        phase_shift_B = exp(-1i * angle(b_shared_temp(idx_max_B)));
        
        beta_shared_aligned = phase_shift_B * beta_shared;
        b_shared_vals_raw = real(exp(1i * phi_grid * n_vals(:).') * beta_shared_aligned);
        
        % Raw weights from the SVD of B_coeff (which already has scale_a absorbed)
        weights_r1_raw = S_B(1,1) * conj(V_B(:, 1)) * conj(phase_shift_B);
        
        % --- 4. Normalize Shared Sender Profile and determine coupling weights w_p ---
        % Normalize shared sender to have mean power = 1
        scale_b = sqrt(mean(b_shared_vals_raw.^2));
        if scale_b == 0, scale_b = eps; end
        
        b_shared_vals = b_shared_vals_raw / scale_b;
        beta_shared_aligned = beta_shared_aligned / scale_b;
        
        % Coupling weights w_p are determined automatically by scaling up by scale_b
        weights_r1 = weights_r1_raw * scale_b;

        % Calculate individual coupling matrix energy ratios and reconstruction metrics
        individual_metrics = {}; % Use cell array to avoid struct parsing/alignment errors
        
        for p = 1:n_blocks
            C_p = C_blocks{p};
            E_total_p = sum(abs(C_p(:)).^2);
            if E_total_p == 0, E_total_p = eps; end
            
            % --- 1. Joint SVD Projection Metrics ---
            C_rec_accum = zeros(size(C_p));
            m_r = struct();
            m_r.pair_name = interaction_meta(p).pair_name;
            m_r.target_id = interaction_meta(p).target_id;
            m_r.source_id = interaction_meta(p).source_id;
            m_r.direction = interaction_meta(p).direction;
            m_r.total_energy = E_total_p;
            
            for r = 1:n_ranks
                u_r = U(:, r);
                v_r_p = V((p-1)*block_size + 1 : p*block_size, r);
                C_comp_r = sigma(r) * (u_r * v_r_p');
                
                E_comp_r = sum(abs(C_comp_r(:)).^2);
                m_r.(sprintf('joint_r%d_energy_ratio', r)) = E_comp_r / E_total_p;
                
                C_rec_accum = C_rec_accum + C_comp_r;
                E_residual = sum(abs(C_p(:) - C_rec_accum(:)).^2);
                m_r.(sprintf('joint_r%d_cum_r2', r)) = 1 - E_residual / E_total_p;
            end
            
            % Fill NaNs if opts.ProfileRank < 3
            for r = (n_ranks+1):3
                m_r.(sprintf('joint_r%d_energy_ratio', r)) = NaN;
                m_r.(sprintf('joint_r%d_cum_r2', r)) = NaN;
            end
            
            % --- 2. Individual SVD Metrics (SVD on Cp alone) ---
            [~, S_indiv, ~] = svd(C_p, 'econ');
            sigma_indiv = diag(S_indiv);
            E_total_indiv = sum(sigma_indiv.^2);
            if E_total_indiv == 0, E_total_indiv = eps; end
            
            accum_indiv = 0;
            for r = 1:min(3, numel(sigma_indiv))
                sig_r = sigma_indiv(r);
                m_r.(sprintf('indiv_r%d_energy_ratio', r)) = (sig_r^2) / E_total_indiv;
                
                accum_indiv = accum_indiv + sig_r^2;
                m_r.(sprintf('indiv_r%d_cum_r2', r)) = accum_indiv / E_total_indiv;
            end
            
            % Fill NaNs if size is smaller than 3
            for r = (numel(sigma_indiv)+1):3
                m_r.(sprintf('indiv_r%d_energy_ratio', r)) = NaN;
                m_r.(sprintf('indiv_r%d_cum_r2', r)) = NaN;
            end
            
            % Add 2nd stage SVD coupling weight
            m_r.weight_r1 = real(weights_r1(p));
            
            individual_metrics{end+1} = m_r; %#ok<AGROW>
        end

        if ~isempty(individual_metrics)
            individual_metrics = [individual_metrics{:}];
        else
            individual_metrics = struct([]);
        end

        % Build output structure
        all_global_results = struct();
        all_global_results.sigma = sigma;
        all_global_results.energy = energy;
        all_global_results.phi_grid = phi_grid;
        all_global_results.components = components;
        all_global_results.interaction_meta = interaction_meta;
        all_global_results.individual_metrics = individual_metrics;
        all_global_results.b_shared_r1 = b_shared_vals(:);
        all_global_results.beta_shared_r1 = beta_shared_aligned(:); % Save complex coefficients
        all_global_results.weights_r1 = weights_r1(:);
        all_global_results.unique_agents = unique_agents;
        all_global_results.M = M;

        % Log SVD metrics comparison table
        fprintf('[INFO] Interaction Matrix SVD Metrics Comparison:\n');
        fprintf('%-12s %-5s | %-10s %-10s %-10s | %-10s %-10s %-10s | %-10s\n', ...
            'Pair', 'Dir', 'Indiv_R1%', 'Indiv_R2%', 'Indiv_R3%', 'Joint_R1%', 'Joint_R2%', 'Joint_R3%', 'Weight_R1');
        fprintf('%s\n', repmat('-', 1, 105));
        for p = 1:numel(individual_metrics)
            m = individual_metrics(p);
            fprintf('%-12s %-5s | %-9.2f%% %-9.2f%% %-9.2f%% | %-9.2f%% %-9.2f%% %-9.2f%% | %-10.4f\n', ...
                m.pair_name, m.direction, ...
                m.indiv_r1_energy_ratio*100, m.indiv_r2_energy_ratio*100, m.indiv_r3_energy_ratio*100, ...
                m.joint_r1_energy_ratio*100, m.joint_r2_energy_ratio*100, m.joint_r3_energy_ratio*100, ...
                m.weight_r1);
        end
        fprintf('\n');

        diary('off');
        fprintf('[INFO] Saved summary log to %s\n', log_file_path);

    % Step 5: Always plot and save figures
    self_profile_dir = fullfile(analysis_out_dir, 'agent_self_profiles');
    if ~exist(self_profile_dir, 'dir')
        mkdir(self_profile_dir);
    end

    phi_grid = all_global_results.phi_grid;
    components = all_global_results.components;
    interaction_meta = all_global_results.interaction_meta;
    n_interactions = numel(interaction_meta);
    sigma = all_global_results.sigma;
    energy = all_global_results.energy;

    % Create Global Concatenated Overlay Plot (3x2 tiled layout)
    fig_global_overlay = figure('Color', 'w', 'Position', [100, 100, 1000, 800], ...
        'Name', 'Global Joint SVD Shared & Pairwise Profiles');
    t_lay = tiledlayout(fig_global_overlay, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    title(t_lay, 'Global Concatenated Joint SVD Overlay (Shared Receiver & Pairwise Senders)', 'FontWeight', 'bold', 'FontSize', 12);

    colors = lines(n_interactions);
    legend_labels = cell(1, n_interactions);
    for p = 1:n_interactions
        legend_labels{p} = sprintf('%s (%d->%d)', interaction_meta(p).pair_name, ...
            interaction_meta(p).source_id, interaction_meta(p).target_id);
    end

    n_ranks = opts.ProfileRank;
    for r = 1:n_ranks
        if r > numel(components)
            continue;
        end
        comp = components(r);
        energy_ratio = (comp.sigma^2 / energy) * 100;

        % --- Left Plot: Single Shared Receiver Profile a_r(phi) ---
        ax_in = nexttile(t_lay, (r - 1) * 2 + 1);
        hold(ax_in, 'on');
        grid(ax_in, 'on');
        box(ax_in, 'on');

        plot(ax_in, phi_grid, comp.a_values, 'LineWidth', 2.8, 'Color', [0 0.447 0.741]);

        ylabel(ax_in, sprintf('a_%d (Shared Target)', r), 'FontWeight', 'bold');
        if r == n_ranks
            xlabel(ax_in, '\phi_{target} (Receiver Phase)');
        end
        set(ax_in, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax_in, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
        xlim(ax_in, [0, 2*pi]);
        legend(ax_in, {'Shared Receiver Profile'}, 'Location', 'best', 'FontSize', 8);
        title(ax_in, sprintf('%d-order Shared Receiver Profile a_%d (var = %.1f%%)', r, r, energy_ratio));

        % --- Right Plot: Interaction-Specific Sender Profiles b_r^(p)(phi) ---
        ax_out = nexttile(t_lay, (r - 1) * 2 + 2);
        hold(ax_out, 'on');
        grid(ax_out, 'on');
        box(ax_out, 'on');

        for p = 1:n_interactions
            b_src = comp.b_values_by_interaction(p);
            plot(ax_out, phi_grid, b_src.b_values, 'LineWidth', 1.5, 'Color', colors(p, :));
        end
        if r == 1 && isfield(all_global_results, 'b_shared_r1') && isfield(all_global_results, 'weights_r1')
            mean_weight = mean(abs(real(all_global_results.weights_r1)));
            b_shared_plot_vals = mean_weight * all_global_results.b_shared_r1;
            plot(ax_out, phi_grid, b_shared_plot_vals, 'k--', 'LineWidth', 3.0);
        end

        ylabel(ax_out, sprintf('b_%d (Pair Senders)', r), 'FontWeight', 'bold');
        if r == n_ranks
            xlabel(ax_out, '\phi_{source} (Sender Phase)');
        end
        set(ax_out, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax_out, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
        xlim(ax_out, [0, 2*pi]);
        if ~isempty(legend_labels)
            leg_lbls = legend_labels;
            if r == 1 && isfield(all_global_results, 'b_shared_r1')
                leg_lbls{end+1} = 'Shared Sender (scaled)';
            end
            legend(ax_out, leg_lbls, 'Location', 'eastoutside', 'FontSize', 7, 'Interpreter', 'none');
        end
        title(ax_out, sprintf('%d-order Sender Profiles b_%d by Pair', r, r));
    end

    global_save_path = fullfile(self_profile_dir, 'global_joint_svd_profiles_overlay.png');
    saveas(fig_global_overlay, global_save_path);
    fprintf('[INFO] Saved Global Concatenated Joint SVD profiles overlay plot to: %s\n', global_save_path);

    % Export global profiles data as CSV
    t_data = table(phi_grid, 'VariableNames', {'phi'});
    if isfield(all_global_results, 'b_shared_r1')
        t_data.b_1_shared = all_global_results.b_shared_r1(:);
    end
    for r = 1:n_ranks
        if r > numel(components), continue; end
        comp = components(r);
        t_data.(sprintf('a_%d_shared', r)) = comp.a_values(:);
        for p = 1:n_interactions
            b_src = comp.b_values_by_interaction(p);
            col_name = sprintf('b_%d_%s_%dto%d', r, b_src.pair_name, b_src.source_id, b_src.target_id);
            col_name = strrep(col_name, '-', '_');
            t_data.(col_name) = b_src.b_values(:);
        end
    end
    csv_global_path = fullfile(self_profile_dir, 'global_joint_svd_profiles_data.csv');
    writetable(t_data, csv_global_path);
    fprintf('[INFO] Saved Global Concatenated Joint SVD profiles CSV data to: %s\n', csv_global_path);

    % Export individual metrics data as CSV
    if isfield(all_global_results, 'individual_metrics')
        t_metrics = struct2table(all_global_results.individual_metrics);
        csv_metrics_path = fullfile(analysis_out_dir, 'global_joint_svd_individual_metrics.csv');
        writetable(t_metrics, csv_metrics_path);
        fprintf('[INFO] Saved individual matrix metrics CSV to: %s\n', csv_metrics_path);
    end

    % Export Simulation Parameters MAT File for external simulation use
    if isfield(all_global_results, 'individual_metrics')
        sim_params = struct();
        sim_params.M = M;
        sim_params.phi_grid = phi_grid(:);
        
        % Shared Target (Receiver) Profile (r=1)
        comp_r1 = all_global_results.components(1);
        sim_params.a_shared_vals = comp_r1.a_values(:);
        if isfield(comp_r1, 'alpha_coeff')
            sim_params.a_shared_coeff = comp_r1.alpha_coeff(:);
        end
        
        % Shared Sender Profile (r=1)
        sim_params.b_shared_vals = all_global_results.b_shared_r1(:);
        if isfield(all_global_results, 'beta_shared_r1')
            sim_params.b_shared_coeff = all_global_results.beta_shared_r1(:);
        end
        
        % Coupling weights list
        n_edges = numel(all_global_results.individual_metrics);
        coupling_weights = struct('source_id', {}, 'target_id', {}, 'weight', {});
        for p = 1:n_edges
            m = all_global_results.individual_metrics(p);
            coupling_weights(p).source_id = m.source_id;
            coupling_weights(p).target_id = m.target_id;
            coupling_weights(p).weight = m.weight_r1;
        end
        sim_params.coupling_weights = coupling_weights;
        
        % Form adjacency coupling matrix (N_agents x N_agents)
        agent_ids = all_global_results.unique_agents(:).';
        n_agents = numel(agent_ids);
        coupling_matrix = zeros(n_agents, n_agents);
        for p = 1:n_edges
            m = all_global_results.individual_metrics(p);
            t_idx = find(agent_ids == m.target_id);
            s_idx = find(agent_ids == m.source_id);
            if ~isempty(t_idx) && ~isempty(s_idx)
                coupling_matrix(t_idx, s_idx) = m.weight_r1;
            end
        end
        sim_params.agent_ids = agent_ids;
        sim_params.coupling_matrix = coupling_matrix;
        
        sim_mat_path = fullfile(analysis_out_dir, 'global_joint_svd_simulation_parameters.mat');
        save(sim_mat_path, '-struct', 'sim_params');
        fprintf('[INFO] Saved simulation parameters MAT to: %s\n', sim_mat_path);
    end

    % Create and save Influence Digraph based on Joint SVD weights
    if isfield(all_global_results, 'individual_metrics')
        % Define figure position/size to be close to square (accommodating the colorbar on the right)
        % to minimize extra whitespace margins.
        fig_digraph = figure('Color', 'w', 'Position', [100, 100, 520, 450], 'Name', 'Global Joint SVD Influence Digraph');
        ax_dig = axes('Parent', fig_digraph);
        
        n_edges = numel(all_global_results.individual_metrics);
        source_agent_id = zeros(n_edges, 1);
        target_agent_id = zeros(n_edges, 1);
        strength = zeros(n_edges, 1);
        
        for p = 1:n_edges
            m = all_global_results.individual_metrics(p);
            source_agent_id(p) = m.source_id;
            target_agent_id(p) = m.target_id;
            strength(p) = m.weight_r1;
        end
        
        node_names = arrayfun(@(id) sprintf('%d', id), all_global_results.unique_agents(:), 'UniformOutput', false);
        source_names = arrayfun(@(id) sprintf('%d', id), source_agent_id(:), 'UniformOutput', false);
        target_names = arrayfun(@(id) sprintf('%d', id), target_agent_id(:), 'UniformOutput', false);
        
        G_dig = digraph(source_names, target_names, strength(:), node_names);
        [x_data, y_data] = get_preferred_node_positions(G_dig, round_dir);
        
        p_dig = plot(ax_dig, G_dig, 'XData', x_data, 'YData', y_data, 'NodeLabel', {}, ...
            'ArrowSize', 16, 'ArrowPosition', 0.75, 'MarkerSize', 8, ...
            'NodeColor', [0.15, 0.15, 0.15], 'EdgeColor', [0.0, 0.4470, 0.7410]);
        axis(ax_dig, 'equal');
        xlim(ax_dig, [0.6, 2.4]);
        ylim(ax_dig, [0.6, 2.4]);
        title(ax_dig, 'Global Joint SVD Directed Influence Graph (Weight R1)', 'Interpreter', 'none');
        
        if numedges(G_dig) > 0
            w_vals = G_dig.Edges.Weight;
            p_dig.LineWidth = scale_edge_width(abs(w_vals), linewidth_limit); % Use absolute values for edge thickness
            p_dig.EdgeCData = w_vals;
            p_dig.EdgeColor = 'flat';
            
            % Set color limits symmetric around zero so that 0 is neutral
            caxis(ax_dig, [-clim_limit, clim_limit]);
            
            % Create custom diverging colormap: Blue (negative) -> Gray (zero) -> Red (positive)
            n_colors = 256;
            half_colors = n_colors / 2;
            c_blue = [0.0, 0.0, 0.85];       % Deep blue for negative strength
            c_center = [0.95, 0.95, 0.95];   % Light gray for zero strength
            c_red = [0.85, 0.0, 0.0];        % Deep red for positive strength
            
            r_blue = linspace(c_blue(1), c_center(1), half_colors)';
            g_blue = linspace(c_blue(2), c_center(2), half_colors)';
            b_blue = linspace(c_blue(3), c_center(3), half_colors)';
            
            r_red = linspace(c_center(1), c_red(1), half_colors)';
            g_red = linspace(c_center(2), c_red(2), half_colors)';
            b_red = linspace(c_center(3), c_red(3), half_colors)';
            
            custom_cmap = [r_blue, g_blue, b_blue; r_red, g_red, b_red];
            colormap(ax_dig, custom_cmap);
            
            cb = colorbar(ax_dig);
            cb.Label.String = 'Weight R1 (Coupling Strength)';
        end
        
        % Draw edge labels manually with offset
        offset_dist = 0.08;
        for e = 1:numedges(G_dig)
            source_node = G_dig.Edges.EndNodes{e, 1};
            target_node = G_dig.Edges.EndNodes{e, 2};
            
            s_idx = find(strcmp(G_dig.Nodes.Name, source_node), 1, 'first');
            t_idx = find(strcmp(G_dig.Nodes.Name, target_node), 1, 'first');
            
            xs = x_data(s_idx); ys = y_data(s_idx);
            xt = x_data(t_idx); yt = y_data(t_idx);
            
            dx = xt - xs; dy = yt - ys;
            len = hypot(dx, dy);
            
            if len > 0
                nx = dy / len;
                ny = -dx / len;
                xl = xs + 0.35 * dx + offset_dist * nx;
                yl = ys + 0.35 * dy + offset_dist * ny;
            else
                xl = xs;
                yl = ys + offset_dist;
            end
            
            val_str = sprintf('%.3g', G_dig.Edges.Weight(e));
            text(ax_dig, xl, yl, val_str, 'FontSize', 9, 'Color', [0.1, 0.1, 0.1], ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'BackgroundColor', 'w', 'Margin', 1);
        end
        
        % Draw node labels manually
        node_offset = 0.12;
        for n = 1:numnodes(G_dig)
            name = G_dig.Nodes.Name{n};
            xn = x_data(n);
            yn = y_data(n);
            
            dx = xn - 1.5; dy = yn - 1.5;
            len = hypot(dx, dy);
            if len > 0
                xl = xn + node_offset * (dx / len);
                yl = yn + node_offset * (dy / len);
            else
                xl = xn;
                yl = yn + node_offset;
            end
            
            text(ax_dig, xl, yl, name, 'FontSize', 12, 'FontWeight', 'bold', ...
                'Color', [0.15, 0.15, 0.15], 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');
        end
        
        graph_save_path = fullfile(self_profile_dir, 'global_joint_svd_influence_graph.png');
        saveas(fig_digraph, graph_save_path);
        fprintf('[INFO] Saved Global Joint SVD influence graph plot to: %s\n', graph_save_path);
        
        if ~opts.keep_figures
            close(fig_digraph);
        end
    end

    if ~opts.keep_figures
        close(fig_global_overlay);
    end

    fprintf('[INFO] Global Joint SVD Analysis completed. Output folder: %s\n', analysis_out_dir);
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

    is_self_term = (term_m ~= 0 & term_n == 0);
    is_other_term = (term_m == 0 & term_n ~= 0);

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


function [x_data, y_data] = get_preferred_node_positions(G, round_dir)
    node_names = G.Nodes.Name;
    if isstring(node_names)
        node_names = cellstr(node_names);
    end
    node_ids = cellfun(@str2double, node_names);

    x_data = nan(1, numnodes(G));
    y_data = nan(1, numnodes(G));

    if nargin >= 2 && contains(lower(round_dir), 'stick')
        % Stick layout:
        % 8  10
        % 7   9
        preferred_ids = [8, 10, 7, 9];
    else
        % Round layout:
        % 8  10
        % 7   9
        preferred_ids = [8, 10, 7, 9];
    end
    preferred_x = [1, 2, 1, 2];
    preferred_y = [2, 2, 1, 1];
    for k = 1:numel(preferred_ids)
        idx = find(node_ids == preferred_ids(k), 1, 'first');
        if ~isempty(idx)
            x_data(idx) = preferred_x(k);
            y_data(idx) = preferred_y(k);
        end
    end

    missing = ~isfinite(x_data) | ~isfinite(y_data);
    if any(missing)
        n_missing = nnz(missing);
        angles = linspace(0, 2*pi, n_missing + 1);
        x_data(missing) = 1.5 + 0.75 * cos(angles(1:end-1));
        y_data(missing) = 1.5 + 0.75 * sin(angles(1:end-1));
    end
end

function widths = scale_edge_width(weights, max_limit)
    weights = double(weights(:));
    if isempty(weights) || all(~isfinite(weights))
        widths = 1.5;
        return;
    end
    if nargin >= 2 && ~isempty(max_limit)
        % Normalize using the fixed max_limit (from 0 to max_limit)
        clamped_weights = min(max(weights, 0), max_limit);
        widths = 1.0 + 5.0 * (clamped_weights / max_limit);
    else
        % Fallback to dynamic scaling if max_limit is not provided
        w_min = min(weights, [], 'omitnan');
        w_max = max(weights, [], 'omitnan');
        if ~isfinite(w_min) || ~isfinite(w_max) || abs(w_max - w_min) < eps
            widths = 2.5 * ones(size(weights));
            return;
        end
        widths = 1.0 + 5.0 * (weights - w_min) / (w_max - w_min);
    end
end
