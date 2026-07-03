function all_results = rankr_approximation_test_round(round_dir, M, varargin)
% Rank-R approximation SVD analysis for s_j(phi1, phi2) on round data folder.
%
% For each pair folder such as "7-8", this function extracts points via
% plot_phase_dynamics_from_csv() and runs SVD-based low-rank analysis on the
% fitted double Fourier coefficient matrices.
%
% Usage:
%   results = rankr_approximation_test_round();
%   results = rankr_approximation_test_round('EstimateL/Round', 10);
%   results = rankr_approximation_test_round(..., 'RemoveSelfOnly', true);
%

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'Round');
    end
    if nargin < 2 || isempty(M)
        M = 5;
    end

    opts = parse_options(varargin{:});
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');

    pair_infos = list_pair_folders(round_dir);
    if isempty(pair_infos)
        error('No pair folders like "7-8" were found under %s.', round_dir);
    end

    all_results = struct([]);
    expected_result_count = 0;

    % Create SVD analysis output directory structure
    analysis_out_dir = fullfile(round_dir, 'low_rank_analysis', sprintf('M%d', M));
    if ~exist(analysis_out_dir, 'dir')
        mkdir(analysis_out_dir);
    end

    % Check SVD cache
    cache_key = compute_svd_cache_key(pair_infos, M, opts);
    cache_path = fullfile(analysis_out_dir, sprintf('svd_cache_%s.mat', cache_key));
    
    is_cached = false;
    if opts.use_cache && isfile(cache_path)
        fprintf('[INFO] Loading cached SVD analysis results from %s\n', cache_path);
        load_data = load(cache_path);
        all_results = load_data.all_results;
        is_cached = true;
    end

    if ~is_cached

    % Start diary to log console outputs to a text file
    log_file_path = fullfile(analysis_out_dir, 'svd_analysis_summary.txt');
    if exist(log_file_path, 'file'), delete(log_file_path); end
    diary(log_file_path);

    fprintf('[INFO] Starting SVD Low-Rank Analysis on Round Data\n');
    fprintf('  Round directory: %s\n', round_dir);
    fprintf('  Output directory: %s\n', analysis_out_dir);
    fprintf('  Fourier order: M=%d\n', M);
    fprintf('  Remove self-only terms: %s\n', mat2str(opts.RemoveSelfOnly));
    fprintf('  Remove constant term: %s\n', mat2str(opts.RemoveConstant));
    fprintf('  Remove other-only terms: %s\n\n', mat2str(opts.RemoveOtherOnly));

    for k = 1:numel(pair_infos)
        info = pair_infos(k);
        csv_pattern = fullfile(info.folder, '*.csv');
        fprintf('[INFO] Pair %d/%d: %d-%d\n', k, numel(pair_infos), info.agent_ids(1), info.agent_ids(2));

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
        phase_agent_ids = pair_out.phase_agent_ids;

        for agent_idx = 1:2
            target_agent_id = phase_agent_ids(agent_idx);
            
            if agent_idx == 1
                y_all = pair_out.point_cloud.s1;
                signal_index = 1;
                self_phase_name = 'phi1';
            else
                y_all = pair_out.point_cloud.s2;
                signal_index = 2;
                self_phase_name = 'phi2';
            end

            fprintf('  [Agent %d (s_%d)] - Self phase: %s\n', target_agent_id, signal_index, self_phase_name);

            % Estimate full Fourier coefficients via regularized LS
            [C_full, m_values, n_values] = estimate_fourier_coeff_matrix(phi1_all, phi2_all, y_all, M);
            
            % Remove phase marginal terms if requested
            [C_analysis, y_analysis, removed_info] = remove_phase_marginal_terms( ...
                C_full, phi1_all, phi2_all, y_all, m_values, n_values, ...
                target_agent_id, phase_agent_ids, ...
                opts.RemoveSelfOnly, opts.RemoveConstant, opts.RemoveOtherOnly);
            
            % Print removal info
            fprintf('    Removed self-only energy ratio:  %.6f\n', removed_info.removed_self_energy_ratio);
            fprintf('    Removed constant energy ratio:   %.6f\n', removed_info.removed_constant_energy_ratio);
            fprintf('    Removed other-only energy ratio: %.6f\n', removed_info.removed_other_energy_ratio);
            fprintf('    Remaining coefficient energy ratio: %.6f\n', removed_info.remaining_energy_ratio);
            fprintf('    Reconstruction mismatch RMSE: %.6e\n', removed_info.reconstruction_mismatch_rmse);

            % Compute rank-R analysis using SVD
            [U, S, V] = svd(C_analysis, 'econ');
            sigma = diag(S);
            energy = sum(sigma.^2);

            term_m = repelem(m_values(:), numel(n_values));
            term_n = repmat(n_values(:), numel(m_values), 1);

            n_rank_to_test = min(5, numel(sigma));
            expected_result_count = expected_result_count + n_rank_to_test;
            
            % Build separable components
            components_all = build_separable_components(U, S, V, m_values, n_values, min(5, numel(sigma)), opts.ImagTolAbs, opts.ImagTolRel, agent_idx);
            component_imag_checks = get_component_imag_checks(components_all);
            
            % Compute self-only profile
            self_profile = compute_self_only_profile(C_full, m_values, n_values, target_agent_id, phase_agent_ids, opts.ImagTolAbs, opts.ImagTolRel);
            
            total_coeff_energy = removed_info.total_coeff_energy;
            removed_total_coeff_energy = removed_info.removed_total_coeff_energy;
            removed_total_energy_ratio = removed_info.removed_total_energy_ratio;
            remaining_energy_ratio = removed_info.remaining_energy_ratio;
            
            fprintf('    Rank | Residual Energy | Total Energy | Residual NRMSE | Total NRMSE\n');
            fprintf('    -----+-----------------+--------------+---------------+------------\n');

            rank_results = struct([]);
            
            for R = 1:n_rank_to_test
                C_rankr = U(:, 1:R) * S(1:R, 1:R) * V(:, 1:R)';
                coef_rankr = reshape(C_rankr.', [], 1);

                energy_r = sum(sigma(1:R).^2);
                
                % Residual metrics (C_analysis based)
                residual_energy_ratio = energy_r / energy;
                residual_nrmse = compute_nrmse_fast(phi1_all, phi2_all, y_analysis, coef_rankr, term_m, term_n);
                
                % Total metrics (C_full based, with removed components added back)
                C_total_rankr = removed_info.C_removed_total + C_rankr;
                coef_total_rankr = reshape(C_total_rankr.', [], 1);
                total_energy_ratio = (removed_total_coeff_energy + energy_r) / total_coeff_energy;
                total_nrmse = compute_nrmse_fast(phi1_all, phi2_all, y_all, coef_total_rankr, term_m, term_n);

                fprintf('    %2d   |     %.6f     |    %.6f    |     %.6f    |   %.6f\n', ...
                    R, residual_energy_ratio, total_energy_ratio, residual_nrmse, total_nrmse);

                result = struct();
                result.pair_name = info.name;
                result.agent_id = target_agent_id;
                result.signal_index = signal_index;
                result.self_phase = self_phase_name;
                result.rank = R;
                result.residual_energy_ratio = residual_energy_ratio;
                result.total_energy_ratio = total_energy_ratio;
                result.residual_nrmse = residual_nrmse;
                result.total_nrmse = total_nrmse;
                result.energy_ratio = residual_energy_ratio;
                result.nrmse = residual_nrmse;
                result.removed_total_energy_ratio = removed_total_energy_ratio;
                result.remaining_energy_ratio = remaining_energy_ratio;
                result.removed_self_energy_ratio = removed_info.removed_self_energy_ratio;
                result.removed_constant_energy_ratio = removed_info.removed_constant_energy_ratio;
                result.removed_other_energy_ratio = removed_info.removed_other_energy_ratio;
                result.reconstruction_mismatch_rmse = removed_info.reconstruction_mismatch_rmse;
                result.coef = coef_rankr;
                result.C_rankr = C_rankr;
                result.C_full = C_full;
                result.C_analysis = C_analysis;
                result.removed_info = removed_info;
                result.self_profile = self_profile;
                result.components = components_all(1:min(R, numel(components_all)));
                
                result.imag_check = struct();
                result.imag_check.self_profile = struct( ...
                    'max_abs_imag', self_profile.max_abs_imag, ...
                    'max_abs_real', self_profile.max_abs_real, ...
                    'relative_imag', self_profile.relative_imag, ...
                    'imag_is_small', self_profile.imag_is_small);
                result.imag_check.components = component_imag_checks;
                result.imag_check.all_ok = self_profile.imag_is_small && all([component_imag_checks.a_imag_is_small]) && all([component_imag_checks.b_imag_is_small]);
                
                if isempty(rank_results)
                    rank_results = orderfields(result);
                else
                    rank_results(end + 1) = orderfields(result, rank_results(1)); %#ok<AGROW>
                end
                
                if isempty(all_results)
                    all_results = orderfields(result);
                else
                    all_results(end + 1) = orderfields(result, all_results(1)); %#ok<AGROW>
                end
            end
            fprintf('\n');

            if opts.CheckImaginaryParts
                fprintf('    [Imaginary-part check]\n');
                fprintf('      Self-only q_%d: max|imag| = %.3e, relative = %.3e, %s\n', ...
                    signal_index, self_profile.max_abs_imag, self_profile.relative_imag, ok_ng(self_profile.imag_is_small));
                for rr = 1:numel(components_all)
                    comp_check = component_imag_checks(rr);
                    fprintf('      r=%d: a max|imag| = %.3e, rel = %.3e, %s; b max|imag| = %.3e, rel = %.3e, %s\n', ...
                        rr, comp_check.a_max_abs_imag, comp_check.a_relative_imag, ok_ng(comp_check.a_imag_is_small), ...
                        comp_check.b_max_abs_imag, comp_check.b_relative_imag, ok_ng(comp_check.b_imag_is_small));
                end
                fprintf('\n');
            end

            % Plot results if requested
            if opts.plot_figures && numel(rank_results) >= 3
                title_suffix = sprintf(' - Pair %s (Agent %d)', info.name, target_agent_id);
                if opts.RemoveSelfOnly || opts.RemoveConstant || opts.RemoveOtherOnly
                    title_suffix = [title_suffix, ' - marginal removed']; %#ok<AGROW>
                end

                fig2d = plot_rank_approximations(phi1_all, phi2_all, y_analysis, rank_results(1:3), m_values, n_values, target_agent_id, title_suffix);
                fig3d = plot_rank_approximations_3d(phi1_all, phi2_all, y_analysis, rank_results(1:3), m_values, n_values, target_agent_id, opts.show_points, opts.colormap, title_suffix);
                fig_sep = plot_separable_profiles(components_all(1:min(opts.ProfileRank, numel(components_all))), m_values, n_values, target_agent_id, opts.ProfileRank, title_suffix);
                
                % Save figures in the pairwise_reconstructions/<pair_name> folder
                pair_plot_dir = fullfile(analysis_out_dir, 'pairwise_reconstructions', info.name);
                if ~exist(pair_plot_dir, 'dir')
                    mkdir(pair_plot_dir);
                end
                
                saveas(fig2d, fullfile(pair_plot_dir, sprintf('agent%d_rank_approx_2d.png', target_agent_id)));
                saveas(fig3d, fullfile(pair_plot_dir, sprintf('agent%d_rank_approx_3d.png', target_agent_id)));
                if ~isempty(fig_sep)
                    saveas(fig_sep, fullfile(pair_plot_dir, sprintf('agent%d_separable_profiles.png', target_agent_id)));
                end

                if opts.PlotSelfOnlyProfile
                    fig_self = plot_self_only_profile(self_profile);
                    saveas(fig_self, fullfile(pair_plot_dir, sprintf('agent%d_self_profile_single.png', target_agent_id)));
                end
            end
        end

        % Clean up figures if they shouldn't be kept open
        if opts.plot_figures && ~opts.keep_figures
            close_figures();
        end
    end

        % Save SVD cache
        if opts.use_cache
            save(cache_path, 'all_results');
            fprintf('[INFO] Saved SVD analysis cache to %s\n', cache_path);
        end
    end

    % Create overlay plots and save self-only profiles grouped by agent ID
    self_profile_dir = fullfile(analysis_out_dir, 'agent_self_profiles');
    if ~exist(self_profile_dir, 'dir')
        mkdir(self_profile_dir);
    end
    
    unique_agents = unique([all_results.agent_id]);
    for a_idx = 1:numel(unique_agents)
        target_aid = unique_agents(a_idx);
        
        % Find all results for this agent
        agent_res_mask = [all_results.agent_id] == target_aid;
        agent_results = all_results(agent_res_mask);
        
        % Filter to unique pair names, but get the result with maximum components
        unique_pair_results = {}; % Use cell array to avoid struct alignment errors
        pair_names = {agent_results.pair_name};
        unique_names = unique(pair_names, 'stable');
        for u_idx = 1:numel(unique_names)
            p_name = unique_names{u_idx};
            p_mask = strcmp(pair_names, p_name);
            p_res = agent_results(p_mask);
            [~, max_idx] = max(cellfun(@(x) numel(x), {p_res.components}));
            unique_pair_results{end+1} = p_res(max_idx); %#ok<AGROW>
        end
        
        if isempty(unique_pair_results)
            continue;
        end
        
        n_pairs_found = numel(unique_pair_results);
        phi_grid = unique_pair_results{1}.self_profile.phi_grid(:);
        
        % Collect all profiles for mean calculation
        profiles_matrix = zeros(numel(phi_grid), n_pairs_found);
        for p_idx = 1:n_pairs_found
            profiles_matrix(:, p_idx) = unique_pair_results{p_idx}.self_profile.real(:);
        end
        mean_profile_real = mean(profiles_matrix, 2);
        
        fig = figure('Color', 'w', 'Position', [100, 100, 800, 500], ...
            'Name', sprintf('Self-only Profile Overlay - Agent %d', target_aid));
        ax = axes('Parent', fig);
        hold(ax, 'on');
        
        legend_labels = {};
        colors = lines(n_pairs_found);
        
        % Plot individual pair profiles
        for p_idx = 1:n_pairs_found
            res = unique_pair_results{p_idx};
            sp = res.self_profile;
            
            plot(ax, sp.phi_grid, sp.real, 'LineWidth', 1.5, 'Color', colors(p_idx, :));
            legend_labels{end+1} = sprintf('Pair %s (s_%d)', res.pair_name, res.signal_index); %#ok<AGROW>
        end
        
        % Plot mean profile on top
        plot(ax, phi_grid, mean_profile_real, 'k--', 'LineWidth', 3);
        legend_labels{end+1} = 'Mean Profile';
        
        grid(ax, 'on');
        box(ax, 'on');
        
        sp_ref = unique_pair_results{1}.self_profile;
        xlabel(ax, sprintf('$$\\phi_{%d}$$', sp_ref.self_phase_index), 'Interpreter', 'latex');
        ylabel(ax, '$$s_{j,\\mathrm{self}}$$', 'Interpreter', 'latex');
        title(ax, sprintf('Self-only profile overlay (with Mean) for Agent %d', target_aid), 'Interpreter', 'latex');
        legend(ax, legend_labels, 'Location', 'best', 'Interpreter', 'none');
        
        set(ax, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
        xlim(ax, [0, 2*pi]);
        
        % Save plot
        save_path = fullfile(self_profile_dir, sprintf('agent%d_self_profile_overlay.png', target_aid));
        saveas(fig, save_path);
        
        % Save profile data as CSV
        t_data = table(phi_grid, mean_profile_real, 'VariableNames', {'phi', 'mean_self_profile'});
        for p_idx = 1:n_pairs_found
            col_name = sprintf('pair_%s', strrep(unique_pair_results{p_idx}.pair_name, '-', '_'));
            t_data.(col_name) = unique_pair_results{p_idx}.self_profile.real(:);
        end
        csv_profile_path = fullfile(self_profile_dir, sprintf('agent%d_self_profile_data.csv', target_aid));
        writetable(t_data, csv_profile_path);
        
        fprintf('[INFO] Saved overlay plot to: %s\n', save_path);
        fprintf('[INFO] Saved self-profile CSV data to: %s\n', csv_profile_path);
        
        % -------------------------------------------------------------
        % Create overlay plots for separable profiles (a and b) for rank 1, 2, 3
        % -------------------------------------------------------------
        
        % Collect target (incoming) profiles a_r and paired source profiles b_r
        incoming_data = struct('r1_a', {}, 'r2_a', {}, 'r3_a', {}, ...
                               'r1_b', {}, 'r2_b', {}, 'r3_b', {}, ...
                               'other_id', {}, 'pair_name', {});
        for p_idx = 1:numel(unique_pair_results)
            res = unique_pair_results{p_idx};
            pair_parts = strsplit(res.pair_name, '-');
            pair_ids = cellfun(@str2double, pair_parts);
            other_id = pair_ids(pair_ids ~= target_aid);
            
            n_comps = numel(res.components);
            incoming_data(end+1).other_id = other_id; %#ok<AGROW>
            incoming_data(end).pair_name = res.pair_name;
            
            % a_values (Target / Incoming) & paired b_values (Source / Sender)
            if n_comps >= 1
                incoming_data(end).r1_a = real(res.components(1).a_values(:));
                incoming_data(end).r1_b = real(res.components(1).b_values(:));
            else
                incoming_data(end).r1_a = zeros(size(phi_grid));
                incoming_data(end).r1_b = zeros(size(phi_grid));
            end
            if n_comps >= 2
                incoming_data(end).r2_a = real(res.components(2).a_values(:));
                incoming_data(end).r2_b = real(res.components(2).b_values(:));
            else
                incoming_data(end).r2_a = zeros(size(phi_grid));
                incoming_data(end).r2_b = zeros(size(phi_grid));
            end
            if n_comps >= 3
                incoming_data(end).r3_a = real(res.components(3).a_values(:));
                incoming_data(end).r3_b = real(res.components(3).b_values(:));
            else
                incoming_data(end).r3_a = zeros(size(phi_grid));
                incoming_data(end).r3_b = zeros(size(phi_grid));
            end
        end
        
        % Plotting Separable Profiles Overlay (3x2 tiled layout)
        fig_sep_overlay = figure('Color', 'w', 'Position', [100, 100, 1000, 800], ...
            'Name', sprintf('Separable Profiles Overlay - Agent %d', target_aid));
        t_lay = tiledlayout(fig_sep_overlay, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        title(t_lay, sprintf('Separable Profiles Overlay for Agent %d (Paired Interactions)', target_aid), 'FontWeight', 'bold', 'FontSize', 12);
        
        % Left col (col 1): Incoming target profiles a_r (phi_target)
        % Right col (col 2): Paired incoming source profiles b_r (phi_source)
        
        for r = 1:3
            % Left Plot: a_r
            ax_in = nexttile(t_lay, (r - 1) * 2 + 1);
            hold(ax_in, 'on');
            grid(ax_in, 'on');
            box(ax_in, 'on');
            
            n_in = numel(incoming_data);
            colors_in = lines(max(1, n_in));
            legend_in = {};
            for k = 1:n_in
                field_name = sprintf('r%d_a', r);
                plot(ax_in, phi_grid, incoming_data(k).(field_name), 'LineWidth', 1.5, 'Color', colors_in(k, :));
                legend_in{end+1} = sprintf('from Agent %d (Pair %s)', incoming_data(k).other_id, incoming_data(k).pair_name); %#ok<AGROW>
            end
            
            ylabel(ax_in, sprintf('a_%d (Target)', r), 'FontWeight', 'bold');
            if r == 3
                xlabel(ax_in, sprintf('\\phi_{%d} (Target Phase)', target_aid));
            end
            set(ax_in, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
            set(ax_in, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
            xlim(ax_in, [0, 2*pi]);
            if ~isempty(legend_in)
                legend(ax_in, legend_in, 'Location', 'best', 'FontSize', 8, 'Interpreter', 'none');
            end
            title(ax_in, sprintf('%d-order Target Profile a_%d', r, r));
            
            % Right Plot: Paired b_r
            ax_out = nexttile(t_lay, (r - 1) * 2 + 2);
            hold(ax_out, 'on');
            grid(ax_out, 'on');
            box(ax_out, 'on');
            
            for k = 1:n_in
                field_name = sprintf('r%d_b', r);
                plot(ax_out, phi_grid, incoming_data(k).(field_name), 'LineWidth', 1.5, 'Color', colors_in(k, :));
            end
            
            ylabel(ax_out, sprintf('b_%d (Paired Source)', r), 'FontWeight', 'bold');
            if r == 3
                xlabel(ax_out, '\\phi_{source} (Source Phase)');
            end
            set(ax_out, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
            set(ax_out, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
            xlim(ax_out, [0, 2*pi]);
            if ~isempty(legend_in)
                legend(ax_out, legend_in, 'Location', 'best', 'FontSize', 8, 'Interpreter', 'none');
            end
            title(ax_out, sprintf('%d-order Paired Source Profile b_%d', r, r));
        end
        
        sep_overlay_save_path = fullfile(self_profile_dir, sprintf('agent%d_separable_profiles_overlay.png', target_aid));
        saveas(fig_sep_overlay, sep_overlay_save_path);
        fprintf('[INFO] Saved separable profiles overlay plot to: %s\n', sep_overlay_save_path);
        
        if ~opts.keep_figures
            close(fig);
            close(fig_sep_overlay);
        end
    end

    % Stop diary logging
    if ~is_cached
        diary('off');
        fprintf('[INFO] Saved text summary to %s\n', log_file_path);
    end

    % Save CSV results table
    if ~isempty(all_results)
        fields_to_remove = {'coef', 'C_rankr', 'C_full', 'C_analysis', 'removed_info', 'self_profile', 'components', 'imag_check'};
        fields_to_remove = fields_to_remove(isfield(all_results, fields_to_remove));
        table_struct = rmfield(all_results, fields_to_remove);
        results_table = struct2table(table_struct);
        csv_path = fullfile(analysis_out_dir, 'svd_analysis_results.csv');
        writetable(results_table, csv_path);
        fprintf('[INFO] Saved CSV results to %s\n', csv_path);
    end

    fprintf('[INFO] SVD Analysis completed. Total saved results: %d\n', numel(all_results));
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

    % Rank analysis parameters
    addParameter(p, 'ProfileRank', 3, @isnumeric);
    addParameter(p, 'RemoveSelfOnly', true, @islogical);
    addParameter(p, 'RemoveConstant', true, @islogical);
    addParameter(p, 'RemoveOtherOnly', false, @islogical);
    addParameter(p, 'PlotSelfOnlyProfile', true, @islogical);
    addParameter(p, 'CheckImaginaryParts', true, @islogical);
    addParameter(p, 'ImagTolAbs', 1e-10, @isnumeric);
    addParameter(p, 'ImagTolRel', 1e-8, @isnumeric);
    addParameter(p, 'show_points', false, @islogical);
    addParameter(p, 'colormap', 'jet', @ischar);

    % Execution options
    addParameter(p, 'plot_figures', true, @islogical);
    addParameter(p, 'keep_figures', false, @islogical);

    parse(p, varargin{:});
    opts = p.Results;
end

function pair_infos = list_pair_folders(round_dir)
    if isstring(round_dir)
        round_dir = char(round_dir);
    end
    if ~isfolder(round_dir)
        candidate = fullfile(pwd, round_dir);
        if isfolder(candidate)
            round_dir = candidate;
        else
            error('round_dir was not found: %s', round_dir);
        end
    end

    dirs = dir(round_dir);
    dirs = dirs([dirs.isdir]);
    pair_infos = struct('name', {}, 'folder', {}, 'agent_ids', {});
    for k = 1:numel(dirs)
        name = dirs(k).name;
        tokens = regexp(name, '^(\d+)-(\d+)$', 'tokens', 'once');
        if isempty(tokens)
            continue;
        end
        folder = fullfile(dirs(k).folder, name);
        if isempty(dir(fullfile(folder, '*.csv')))
            continue;
        end

        pair_infos(end + 1, 1) = struct( ...
            'name', name, ...
            'folder', folder, ...
            'agent_ids', [str2double(tokens{1}), str2double(tokens{2})]); %#ok<AGROW>
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

    for k0 = 1:chunk_size:n_samples
        k1 = min(k0 + chunk_size - 1, n_samples);
        idx = k0:k1;

        A = exp(1i * (phi1(idx) * term_m.' + phi2(idx) * term_n.'));
        G = G + A' * A;
        h = h + A' * y(idx(:));
    end

    lambda = 1e-12 * trace(G) / n_terms;
    Greg = G + lambda * eye(n_terms);

    coef = Greg \ h;
    C = reshape(coef, numel(n_values), numel(m_values)).';
end

function nrmse = compute_nrmse_fast(phi1, phi2, y, coef, term_m, term_n)
    y = y(:);
    n_samples = numel(y);

    A = exp(1i * (phi1(:) * term_m.' + phi2(:) * term_n.'));
    y_pred = real(A * coef);
    residual = y - y_pred;
    sse = sum(residual.^2);
    y_std = std(y);

    if y_std > 0
        nrmse = sqrt(sse / n_samples) / y_std;
    else
        nrmse = sqrt(sse / n_samples);
    end
end

function [C_analysis, y_analysis, removed_info] = remove_phase_marginal_terms( ...
    C_full, phi1, phi2, y, m_values, n_values, ...
    target_agent_id, phase_agent_ids, ...
    remove_self_only, remove_constant, remove_other_only)
    
    C_analysis = C_full;
    C_removed_self = zeros(size(C_full));
    C_removed_constant = zeros(size(C_full));
    C_removed_other = zeros(size(C_full));
    
    idx_m0 = find(m_values == 0);
    idx_n0 = find(n_values == 0);
    
    if target_agent_id == phase_agent_ids(1)
        signal_index = 1;
        self_phase_index = 1;
        self_phase_name = 'phi1';
    else
        signal_index = 2;
        self_phase_index = 2;
        self_phase_name = 'phi2';
    end
    
    if signal_index == 1
        if remove_self_only
            C_removed_self(m_values ~= 0, idx_n0) = C_full(m_values ~= 0, idx_n0);
            C_analysis(m_values ~= 0, idx_n0) = 0;
        end
        if remove_other_only
            C_removed_other(idx_m0, n_values ~= 0) = C_full(idx_m0, n_values ~= 0);
            C_analysis(idx_m0, n_values ~= 0) = 0;
        end
    else
        if remove_self_only
            C_removed_self(idx_m0, n_values ~= 0) = C_full(idx_m0, n_values ~= 0);
            C_analysis(idx_m0, n_values ~= 0) = 0;
        end
        if remove_other_only
            C_removed_other(m_values ~= 0, idx_n0) = C_full(m_values ~= 0, idx_n0);
            C_analysis(m_values ~= 0, idx_n0) = 0;
        end
    end
    
    if remove_constant
        C_removed_constant(idx_m0, idx_n0) = C_full(idx_m0, idx_n0);
        C_analysis(idx_m0, idx_n0) = 0;
    end
    
    C_removed_total = C_removed_self + C_removed_constant + C_removed_other;
    
    y_removed_total = evaluate_fourier_from_C(phi1, phi2, C_removed_total, m_values, n_values);
    y_analysis = y(:) - y_removed_total(:);
    
    y_analysis_from_C = evaluate_fourier_from_C(phi1, phi2, C_analysis, m_values, n_values);
    reconstruction_mismatch_rmse = sqrt(mean((y_analysis(:) - y_analysis_from_C(:)).^2));
    
    y_removed_self = evaluate_fourier_from_C(phi1, phi2, C_removed_self, m_values, n_values);
    y_removed_constant = evaluate_fourier_from_C(phi1, phi2, C_removed_constant, m_values, n_values);
    y_removed_other = evaluate_fourier_from_C(phi1, phi2, C_removed_other, m_values, n_values);
    
    total_coeff_energy = sum(abs(C_full(:)).^2);
    removed_self_coeff_energy = sum(abs(C_removed_self(:)).^2);
    removed_constant_energy = sum(abs(C_removed_constant(:)).^2);
    removed_other_coeff_energy = sum(abs(C_removed_other(:)).^2);
    removed_total_coeff_energy = sum(abs(C_removed_total(:)).^2);
    remaining_coeff_energy = sum(abs(C_analysis(:)).^2);
    
    removed_info = struct();
    removed_info.remove_self_only = remove_self_only;
    removed_info.remove_constant = remove_constant;
    removed_info.remove_other_only = remove_other_only;
    removed_info.self_phase = self_phase_name;
    removed_info.self_phase_index = self_phase_index;
    removed_info.target_agent_id = target_agent_id;
    removed_info.phase_agent_ids = phase_agent_ids;
    removed_info.signal_index = signal_index;
    removed_info.C_removed_self = C_removed_self;
    removed_info.C_removed_constant = C_removed_constant;
    removed_info.C_removed_other = C_removed_other;
    removed_info.C_removed_total = C_removed_total;
    removed_info.y_removed_self = y_removed_self;
    removed_info.y_removed_constant = y_removed_constant;
    removed_info.y_removed_other = y_removed_other;
    removed_info.y_removed_total = y_removed_total;
    removed_info.y_analysis = y_analysis;
    removed_info.removed_self_coeff_energy = removed_self_coeff_energy;
    removed_info.removed_constant_energy = removed_constant_energy;
    removed_info.removed_other_coeff_energy = removed_other_coeff_energy;
    removed_info.removed_total_coeff_energy = removed_total_coeff_energy;
    removed_info.remaining_coeff_energy = remaining_coeff_energy;
    removed_info.total_coeff_energy = total_coeff_energy;
    removed_info.removed_self_energy_ratio = removed_self_coeff_energy / total_coeff_energy;
    removed_info.removed_constant_energy_ratio = removed_constant_energy / total_coeff_energy;
    removed_info.removed_other_energy_ratio = removed_other_coeff_energy / total_coeff_energy;
    removed_info.removed_total_energy_ratio = removed_total_coeff_energy / total_coeff_energy;
    removed_info.remaining_energy_ratio = remaining_coeff_energy / total_coeff_energy;
    removed_info.reconstruction_mismatch_rmse = reconstruction_mismatch_rmse;
end

function y_hat = evaluate_fourier_from_C(phi1, phi2, C, m_values, n_values)
    coef = reshape(C.', [], 1);
    term_m = repelem(m_values(:), numel(n_values));
    term_n = repmat(n_values(:), numel(m_values), 1);
    term_n = term_n(:);
    
    phi1 = phi1(:);
    phi2 = phi2(:);
    n_samples = numel(phi1);
    chunk_size = 20000;
    y_hat = zeros(n_samples, 1);
    
    for k0 = 1:chunk_size:n_samples
        k_end = min(k0 + chunk_size - 1, n_samples);
        k_idx = k0:k_end;
        A_chunk = exp(1i * (phi1(k_idx) * term_m.' + phi2(k_idx) * term_n.'));
        y_hat(k_idx) = real(A_chunk * coef);
    end
end

function components = build_separable_components(U, S, V, m_values, n_values, profile_rank, imag_tol_abs, imag_tol_rel, agent_idx)
    components = struct('alpha', {}, 'beta', {}, 'sigma', {});
    n_ranks = min(profile_rank, size(U, 2));
    phi_grid = linspace(0, 2*pi, 512).';
    
    for r = 1:n_ranks
        sigma_r = S(r, r);
        
        if nargin >= 9 && agent_idx == 2
            % For s2: target is phi2 (V-side), source is phi1 (U-side)
            % So target profile (a) should be associated with V, and source (b) with U.
            alpha_r = sqrt(sigma_r) * conj(V(:, r));
            beta_r = sqrt(sigma_r) * U(:, r);
            
            a_r_temp = exp(1i * phi_grid * n_values(:).') * alpha_r;
            [~, idx_max] = max(abs(a_r_temp));
            phase_shift = exp(-1i * angle(a_r_temp(idx_max)));
            alpha_r = phase_shift * alpha_r;
            beta_r = conj(phase_shift) * beta_r;

            a_values = exp(1i * phi_grid * n_values(:).') * alpha_r;
            b_values = exp(1i * phi_grid * m_values(:).') * beta_r;
        else
            % For s1: target is phi1 (U-side), source is phi2 (V-side)
            alpha_r = sqrt(sigma_r) * U(:, r);
            beta_r = sqrt(sigma_r) * conj(V(:, r));
            
            a_r_temp = exp(1i * phi_grid * m_values(:).') * alpha_r;
            [~, idx_max] = max(abs(a_r_temp));
            phase_shift = exp(-1i * angle(a_r_temp(idx_max)));
            alpha_r = phase_shift * alpha_r;
            beta_r = conj(phase_shift) * beta_r;

            a_values = exp(1i * phi_grid * m_values(:).') * alpha_r;
            b_values = exp(1i * phi_grid * n_values(:).') * beta_r;
        end
        
        a_stats = compute_imag_part_metrics(a_values, imag_tol_abs, imag_tol_rel);
        b_stats = compute_imag_part_metrics(b_values, imag_tol_abs, imag_tol_rel);
        
        components(r).alpha = alpha_r;
        components(r).beta = beta_r;
        components(r).sigma = sigma_r;
        components(r).phi_grid = phi_grid;
        components(r).a_values = a_values;
        components(r).b_values = b_values;
        components(r).a_max_abs_imag = a_stats.max_abs_imag;
        components(r).a_max_abs_real = a_stats.max_abs_real;
        components(r).a_relative_imag = a_stats.relative_imag;
        components(r).a_imag_is_small = a_stats.imag_is_small;
        components(r).b_max_abs_imag = b_stats.max_abs_imag;
        components(r).b_max_abs_real = b_stats.max_abs_real;
        components(r).b_relative_imag = b_stats.relative_imag;
        components(r).b_imag_is_small = b_stats.imag_is_small;
    end
end

function fig = plot_separable_profiles(components, m_values, n_values, agent_id, profile_rank, title_suffix)
    if nargin < 6
        title_suffix = '';
    end
    n_components = numel(components);
    n_display = min(n_components, profile_rank);
    if n_display == 0, fig = []; return; end
    
    fig = figure('Color', 'w', 'Position', [100, 100, 1000, 300*n_display], ...
        'Name', sprintf('Separable Profiles - Agent %d%s', agent_id, title_suffix));
    tiledlayout(n_display, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    phi_grid = linspace(0, 2*pi, 512);
    
    for r = 1:n_display
        comp = components(r);
        sigma_r = comp.sigma;
        
        a_r = exp(1i * phi_grid(:) * m_values(:).') * comp.alpha;
        b_r = exp(1i * phi_grid(:) * n_values(:).') * comp.beta;
        
        ax_a = nexttile;
        plot(ax_a, phi_grid, real(a_r), 'LineWidth', 1.5);
        xlabel(ax_a, '$$\phi_1$$', 'Interpreter', 'latex');
        ylabel(ax_a, '$$a_{j,r}(\phi_1)$$', 'Interpreter', 'latex');
        title(ax_a, sprintf('$a_{%d,%d}(\\phi_1)$, $\\sigma = %.4e$', agent_id, r, sigma_r), 'Interpreter', 'latex');
        set(ax_a, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax_a, 'XTickLabel', {'0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'}, 'TickLabelInterpreter', 'latex');
        grid(ax_a, 'on');
        
        ax_b = nexttile;
        plot(ax_b, phi_grid, real(b_r), 'LineWidth', 1.5);
        xlabel(ax_b, '$$\phi_2$$', 'Interpreter', 'latex');
        ylabel(ax_b, '$$b_{j,r}(\phi_2)$$', 'Interpreter', 'latex');
        title(ax_b, sprintf('$b_{%d,%d}(\\phi_2)$, $\\sigma = %.4e$', agent_id, r, sigma_r), 'Interpreter', 'latex');
        set(ax_b, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax_b, 'XTickLabel', {'0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'}, 'TickLabelInterpreter', 'latex');
        grid(ax_b, 'on');
    end
end

function fig = plot_rank_approximations(phi1_all, phi2_all, y_all, rank_results, m_values, n_values, agent_id, title_suffix)
    if nargin < 8
        title_suffix = '';
    end
    n_grid = 64;
    phi1_grid = linspace(0, 2*pi, n_grid);
    phi2_grid = linspace(0, 2*pi, n_grid);
    
    fig = figure('Color', 'w', 'Position', [100, 100, 1200, 400], ...
        'Name', sprintf('Rank Approximations - Agent %d%s', agent_id, title_suffix));
    
    for rank_idx = 1:3
        if rank_idx > numel(rank_results), break; end
        
        result = rank_results(rank_idx);
        rank = result.rank;
        coef = result.coef;
        
        term_m = repelem(m_values(:), numel(n_values));
        term_n = repmat(n_values(:), numel(m_values), 1);
        
        n_phi1 = numel(phi1_grid);
        n_phi2 = numel(phi2_grid);
        Z = zeros(n_phi2, n_phi1);
        
        for i = 1:n_phi1
            for j = 1:n_phi2
                A = exp(1i * (phi1_grid(i) * term_m.' + phi2_grid(j) * term_n.'));
                Z(j, i) = real(A * coef);
            end
        end
        
        ax = subplot(1, 3, rank_idx);
        imagesc(phi1_grid, phi2_grid, Z);
        set(ax, 'YDir', 'normal');
        colorbar(ax);
        
        n_colors = 256;
        cmap_blue = [linspace(0, 1, n_colors/2).' linspace(0, 0, n_colors/2).' linspace(1, 0.5, n_colors/2).'];
        cmap_red = [linspace(0.5, 1, n_colors/2).' linspace(0, 0, n_colors/2).' linspace(0, 0, n_colors/2).'];
        cmap = [flipud(cmap_blue); cmap_red];
        colormap(ax, cmap);
        
        z_max = max(abs(Z(:)));
        if z_max > 0, caxis(ax, [-z_max, z_max]); end
        
        xlabel(ax, '$$\phi_1$$', 'Interpreter', 'latex');
        ylabel(ax, '$$\phi_2$$', 'Interpreter', 'latex');
        title(ax, sprintf('Rank %d (NRMSE=%.4f)', rank, result.nrmse), 'Interpreter', 'latex');
        
        set(ax, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax, 'YTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
        set(ax, 'YTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
        axis(ax, 'square');
    end
end

function fig = plot_rank_approximations_3d(phi1_all, phi2_all, y_all, rank_results, m_values, n_values, agent_id, show_points, colormap_name, title_suffix)
    if nargin < 10
        title_suffix = '';
    end
    n_grid = 64;
    phi1_grid = linspace(0, 2*pi, n_grid);
    phi2_grid = linspace(0, 2*pi, n_grid);
    [PHI1_grid, PHI2_grid] = meshgrid(phi1_grid, phi2_grid);
    
    fig = figure('Color', 'w', 'Position', [100, 550, 1200, 400], ...
        'Name', sprintf('Rank Approximations 3D - Agent %d%s', agent_id, title_suffix));
    
    for rank_idx = 1:3
        if rank_idx > numel(rank_results), break; end
        
        result = rank_results(rank_idx);
        rank = result.rank;
        coef = result.coef;
        
        term_m = repelem(m_values(:), numel(n_values));
        term_n = repmat(n_values(:), numel(m_values), 1);
        
        n_phi1 = numel(phi1_grid);
        n_phi2 = numel(phi2_grid);
        Z = zeros(n_phi2, n_phi1);
        
        for i = 1:n_phi1
            for j = 1:n_phi2
                A = exp(1i * (phi1_grid(i) * term_m.' + phi2_grid(j) * term_n.'));
                Z(j, i) = real(A * coef);
            end
        end
        
        ax = subplot(1, 3, rank_idx);
        surf(ax, PHI1_grid, PHI2_grid, Z, 'EdgeColor', 'none');
        hold(ax, 'on');
        
        if show_points
            scatter3(ax, phi1_all, phi2_all, y_all, 15, y_all, 'filled', 'MarkerEdgeColor', 'k', ...
                'MarkerEdgeAlpha', 0.3, 'MarkerFaceAlpha', 0.5);
        end
        
        colormap(ax, colormap_name);
        colorbar(ax);
        
        xlabel(ax, '$$\phi_1$$', 'Interpreter', 'latex');
        ylabel(ax, '$$\phi_2$$', 'Interpreter', 'latex');
        zlabel(ax, '$$s_j(\phi_1,\phi_2)$$', 'Interpreter', 'latex');
        title(ax, sprintf('Rank %d (NRMSE=%.4f)', rank, result.nrmse), 'Interpreter', 'latex');
        
        set(ax, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax, 'YTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
        set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
        set(ax, 'YTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
        
        view(ax, 45, 30);
        grid(ax, 'on');
        hold(ax, 'off');
    end
end

function self_profile = compute_self_only_profile(C_full, m_values, n_values, target_agent_id, phase_agent_ids, imag_tol_abs, imag_tol_rel)
    phi_grid = linspace(0, 2*pi, 512).';
    
    if target_agent_id == phase_agent_ids(1)
        signal_index = 1;
        self_phase_index = 1;
        self_phase_name = 'phi1';
        
        idx_n0 = find(n_values == 0);
        idx_m_nonzero = find(m_values ~= 0);
        
        if ~isempty(idx_n0) && ~isempty(idx_m_nonzero)
            coeff_vec = C_full(idx_m_nonzero, idx_n0(:));
            coeff_vec = coeff_vec(:);
            m_basis = m_values(idx_m_nonzero);
            E = exp(1i * phi_grid * m_basis(:).');
        else
            coeff_vec = 0;
            E = zeros(numel(phi_grid), 1);
        end
    else
        signal_index = 2;
        self_phase_index = 2;
        self_phase_name = 'phi2';
        
        idx_m0 = find(m_values == 0);
        idx_n_nonzero = find(n_values ~= 0);
        
        if ~isempty(idx_m0) && ~isempty(idx_n_nonzero)
            coeff_vec = C_full(idx_m0(:), idx_n_nonzero);
            coeff_vec = coeff_vec(:);
            n_basis = n_values(idx_n_nonzero);
            E = exp(1i * phi_grid * n_basis(:).');
        else
            coeff_vec = 0;
            E = zeros(numel(phi_grid), 1);
        end
    end
    
    if numel(E) == 1
        profile_values = zeros(size(phi_grid));
    else
        profile_values = E * coeff_vec;
    end
    
    self_profile = struct();
    self_profile.phi_grid = phi_grid;
    self_profile.values = profile_values;
    self_profile.real = real(profile_values);
    self_profile.imag = imag(profile_values);
    self_profile.abs = abs(profile_values);
    self_profile.self_phase = self_phase_name;
    self_profile.self_phase_index = self_phase_index;
    self_profile.target_agent_id = target_agent_id;
    self_profile.signal_index = signal_index;
    imag_stats = compute_imag_part_metrics(profile_values, imag_tol_abs, imag_tol_rel);
    self_profile.max_abs_imag = imag_stats.max_abs_imag;
    self_profile.max_abs_real = imag_stats.max_abs_real;
    self_profile.relative_imag = imag_stats.relative_imag;
    self_profile.imag_is_small = imag_stats.imag_is_small;
end

function imag_stats = compute_imag_part_metrics(values, imag_tol_abs, imag_tol_rel)
    values = values(:);
    max_abs_imag = max(abs(imag(values)));
    max_abs_real = max(abs(real(values)));
    relative_imag = max_abs_imag / max(max_abs_real, eps);
    imag_stats = struct();
    imag_stats.max_abs_imag = max_abs_imag;
    imag_stats.max_abs_real = max_abs_real;
    imag_stats.relative_imag = relative_imag;
    imag_stats.imag_is_small = (max_abs_imag < imag_tol_abs) || (relative_imag < imag_tol_rel);
end

function comp_checks = get_component_imag_checks(components)
    comp_checks = struct('a_max_abs_imag', {}, 'a_max_abs_real', {}, 'a_relative_imag', {}, 'a_imag_is_small', {}, ...
        'b_max_abs_imag', {}, 'b_max_abs_real', {}, 'b_relative_imag', {}, 'b_imag_is_small', {});
    if isempty(components), return; end
    comp_checks(1, numel(components)) = struct('a_max_abs_imag', [], 'a_max_abs_real', [], 'a_relative_imag', [], 'a_imag_is_small', [], ...
        'b_max_abs_imag', [], 'b_max_abs_real', [], 'b_relative_imag', [], 'b_imag_is_small', []);
    for k = 1:numel(components)
        comp_checks(k).a_max_abs_imag = components(k).a_max_abs_imag;
        comp_checks(k).a_max_abs_real = components(k).a_max_abs_real;
        comp_checks(k).a_relative_imag = components(k).a_relative_imag;
        comp_checks(k).a_imag_is_small = components(k).a_imag_is_small;
        comp_checks(k).b_max_abs_imag = components(k).b_max_abs_imag;
        comp_checks(k).b_max_abs_real = components(k).b_max_abs_real;
        comp_checks(k).b_relative_imag = components(k).b_relative_imag;
        comp_checks(k).b_imag_is_small = components(k).b_imag_is_small;
    end
end

function s = ok_ng(tf)
    if tf, s = 'OK'; else, s = 'NG'; end
end

function fig = plot_self_only_profile(self_profile)
    phi_grid = self_profile.phi_grid;
    signal_index = self_profile.signal_index;
    target_agent_id = self_profile.target_agent_id;
    self_phase = self_profile.self_phase;
    self_phase_index = self_profile.self_phase_index;
    
    fig = figure('Color', 'w', 'Position', [100, 50, 800, 400], ...
        'Name', sprintf('Self-only Profile - s_%d (Agent %d)', signal_index, target_agent_id));
    
    ax = axes('Parent', fig);
    plot(ax, phi_grid, self_profile.real, 'LineWidth', 1.5);
    
    xlabel(ax, sprintf('$$%s$$', self_phase), 'Interpreter', 'latex');
    ylabel(ax, '$$s_{j,\mathrm{self}}$$', 'Interpreter', 'latex');
    title(ax, sprintf('Self-only profile $q_{%d}(\\phi_{%d})$, max $|\\mathrm{imag}|$ = %.2e', ...
        target_agent_id, self_phase_index, self_profile.max_abs_imag), ...
        'Interpreter', 'latex');
    
    set(ax, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
    set(ax, 'XTickLabel', {'0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'}, 'TickLabelInterpreter', 'latex');
    grid(ax, 'on');
end

function close_figures()
    fig_handles = findobj('Type', 'figure');
    for i = 1:numel(fig_handles)
        if contains(get(fig_handles(i), 'Name'), 'Rank Approximations') || ...
           contains(get(fig_handles(i), 'Name'), 'Separable Profiles') || ...
           contains(get(fig_handles(i), 'Name'), 'Self-only Profile')
            close(fig_handles(i));
        end
    end
end

function cache_key = compute_svd_cache_key(pair_infos, M, opts)
    parts = {};
    parts{end + 1} = 'version=v2_direction_unified';
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
    
    % Include CSV files metadata to detect changes
    for i = 1:numel(pair_infos)
        csv_files = dir(fullfile(pair_infos(i).folder, '*.csv'));
        for j = 1:numel(csv_files)
            parts{end + 1} = sprintf('%s|%d|%.15g', csv_files(j).name, csv_files(j).bytes, csv_files(j).datenum); %#ok<AGROW>
        end
    end
    
    cache_key = md5_hex(strjoin(parts, '\n'));
end

function hex_text = md5_hex(text_value)
    % Compute MD5 hash using Java Cryptography Architecture
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
