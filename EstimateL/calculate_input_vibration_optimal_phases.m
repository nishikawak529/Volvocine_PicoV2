function results = calculate_input_vibration_optimal_phases(source, varargin)
%CALCULATE_INPUT_VIBRATION_OPTIMAL_PHASES Calculates phase relationships that
% minimize and maximize input vibrations based on exported SVD modes and network coupling.
%
% Theoretical Foundation:
%   From global_joint_cp_rank1_profile_free_network_svd, the interaction model is:
%     s_{i<-j}(phi_i, phi_j) approx W_{ij} * a(phi_i) * b(phi_j)
%     b(phi_j) = sqrt(2) * cos(phi_j - delta)
%   Under phase-locked oscillations phi_j(t) = Omega*t + theta_j:
%     b(phi_j(t)) = Re{ sqrt(2) * exp(-1i*delta) * exp(1i*theta_j) * exp(1i*Omega*t) }
%   Defining complex phasor z_j = exp(1i * theta_j), the vibrations are:
%
%   1. SVD Mode-wise Collective Sender Signal:
%      X_l(t) = sum_j v_{jl} * b(phi_j(t))
%      Amplitude A_{X,l}(theta) = sqrt(2) * |sum_j v_{jl} exp(1i*theta_j)|
%      - Maximized when arg(v_{jl} * exp(1i*theta_j)) are aligned (sign-coherent in-phase)
%        Theoretical Max Amplitude = sqrt(2) * ||v_l||_1 = sqrt(2) * sum_j |v_{jl}|
%      - Minimized by destructive interference (sum_j v_{jl} exp(1i*theta_j) -> 0).
%
%   2. Total Network Input Vibration Energy:
%      Each receiver i receives input I_i(t) = sum_{j ~= i} W_{ij} * b(phi_j(t))
%      Total time-averaged input power J_{total}(theta) = sum_i <I_i(t)^2>
%        J_{total}(theta) = z^H * (W^T * W) * z = sum_{j,k} Q_{jk} * cos(theta_j - theta_k)
%      - Maximized: phases constructively interfere across the network.
%      - Minimized: destructive network interference (quenching input oscillations).
%
%   3. Agent-wise Received Input Vibration:
%      Agent i receives amplitude A_{I,i}(theta) = sqrt(2) * |sum_{j ~= i} W_{ij} exp(1i*theta_j)|
%      - Maximized: senders j in-phase with sgn(W_{ij}). Max = sqrt(2) * sum_{j ~= i} |W_{ij}|.
%      - Minimized: senders j arranged to cancel sum_{j ~= i} W_{ij} exp(1i*theta_j) -> 0.
%
%   4. Total Receiver Input Amplitude Sum:
%      x = exp(1i * theta); G = W * x;
%      agent_amplitudes = sqrt(2) * abs(G);
%      J_amp = sum(agent_amplitudes);
%      - Evaluates the sum of vibration amplitudes across all receiver modules under
%        a single global phase configuration theta.
%      - Optimizes over N-1 variables with ReferenceAgent fixed to 0.
%      - Solved via smooth continuation and multi-start local search ("Best-found" states).
%      - Distinct from quadratic energy sum(abs(G).^2) or coherent sum abs(sum(G)).
%
% Usage:
%   results = calculate_input_vibration_optimal_phases();             % Analyzes DEFAULT_DATASET ('SStick')
%   results = calculate_input_vibration_optimal_phases('Round');      % Direct dataset name
%   results = calculate_input_vibration_optimal_phases('Round6');     % 6-agent dataset
%   results = calculate_input_vibration_optimal_phases('SStick');
%   results = calculate_input_vibration_optimal_phases('Stick');
%   results = calculate_input_vibration_optimal_phases(source_dir);
%   results = calculate_input_vibration_optimal_phases(..., 'NumStarts', 100, 'SaveOutputs', true);
%
% Outputs:
%   results: Struct containing optimal phases, amplitudes, energy metrics, and diagnostics.

    % =========================================================================
    % USER CONFIGURATION: TARGET DATASET / DIRECTORY (For F5 / Run without args)
    % =========================================================================
    % Change this single variable to switch datasets easily in code:
    %   'SStickFlat': SStickFlat experiment (Agents 8, 9, 11, 12)
    %   'SStick'    : SStick experiment (Agents 7, 8, 9, 10)
    %   'Round'     : Round experiment (Agents 7, 8, 9, 10)
    %   'Round6'    : Round6 experiment (Agents 7, 8, 9, 10, 11, 12)
    %   'Stick'     : Stick experiment (Agents 7, 8, 9, 10)
    % Or specify any relative/absolute path or results struct.
    % =========================================================================
    DEFAULT_DATASET = 'SStickFlat';

    if nargin < 1 || isempty(source)
        source = DEFAULT_DATASET;
    end

    opts = parse_options(varargin{:});

    % Initialize global random seed upfront before any random operations
    if isfield(opts, 'RandomSeed') && ~isempty(opts.RandomSeed)
        rng(opts.RandomSeed);
    end

    % 1. Load exported mode data (from folder, CSVs, or struct)
    data = load_exported_mode_data(source);
    N = data.N;
    agent_ids = data.agent_ids;
    W = data.W;
    U = data.U;
    V = data.V;
    sigma = data.singular_values;
    delta = data.delta;
    output_dir = opts.OutputDir;
    if isempty(output_dir)
        output_dir = fullfile(data.source_dir, 'optimal_phase_analysis');
    end
    if opts.SaveOutputs && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    fprintf('=========================================================================\n');
    fprintf('   INPUT VIBRATION OPTIMAL PHASE ANALYSIS (MINIMIZATION & MAXIMIZATION)\n');
    fprintf('=========================================================================\n');
    fprintf('  Source: %s\n', data.source_dir);
    fprintf('  Agents (N=%d): %s\n', N, mat2str(agent_ids));
    fprintf('  Output Dir: %s\n\n', output_dir);

    % Gauge fixing reference agent
    ref_idx = 1;
    if ~isempty(opts.ReferenceAgent)
        match_idx = find(agent_ids == opts.ReferenceAgent, 1);
        if ~isempty(match_idx)
            ref_idx = match_idx;
        end
    end
    ref_agent_id = agent_ids(ref_idx);

    % 2. Optimization for SVD Mode Collective Signals X_l
    fprintf('[INFO] Optimizing phase relationships for SVD Modes (l = 1..%d)...\n', N);
    mode_cell = cell(N, 1);
    for l = 1:N
        mode_cell{l} = optimize_svd_mode_signal(V(:, l), sigma(l), U(:, l), ref_idx, opts);
        fprintf('  Mode %2d (sigma = %.4f): Max Amp = %.4f | Min Amp = %.4e | Ratio (Min/Max) = %.2e\n', ...
            l, sigma(l), mode_cell{l}.max_amplitude, mode_cell{l}.min_amplitude, ...
            mode_cell{l}.amplitude_ratio);
    end
    mode_results = [mode_cell{:}];

    % 3. Optimization for Total Network Input Vibration Energy ||W * z||^2
    fprintf('\n[INFO] Optimizing phase relationships for Total Network Input Vibration...\n');
    network_results = optimize_total_network_vibration(W, ref_idx, opts);
    fprintf('  Total Network Vibration Energy: Max = %.5f | Min = %.5e | Suppression = %.2f%%\n', ...
        network_results.max_energy, network_results.min_energy, ...
        100 * (1 - network_results.min_energy / max(1e-12, network_results.max_energy)));

    % 4. Optimization for Agent-wise Received Input Vibration
    fprintf('\n[INFO] Optimizing phase relationships for Agent-wise Received Input Vibration...\n');
    agent_cell = cell(N, 1);
    for i = 1:N
        agent_cell{i} = optimize_agent_wise_vibration(W(i, :), i, ref_idx, opts);
        fprintf('  Receiver Agent %2d: Max Amp = %.4f | Min Amp = %.4e\n', ...
            agent_ids(i), agent_cell{i}.max_amplitude, agent_cell{i}.min_amplitude);
    end
    agent_results = [agent_cell{:}];

    % 5. Optimization for Total Input Amplitude Sum J_amp = sum_i sqrt(2)*|G_i|
    fprintf('\n[INFO] Optimizing phase relationships for Total Input Amplitude Sum (Best-found)...\n');
    total_amp_results = optimize_total_input_amplitude_sum(W, ref_idx, mode_results, network_results, opts);
    fprintf('  Total Input Amplitude Sum: Best-found Max = %.5f | Best-found Min = %.5f | Suppression = %.2f%%\n', ...
        total_amp_results.max_amplitude_sum, total_amp_results.min_amplitude_sum, ...
        total_amp_results.suppression_percent);

    % 6. Compare with observed phase time series if available
    observed_results = evaluate_observed_phases(data, W, V, sigma, delta);

    % 7. Assemble complete results struct
    results = struct();
    results.source_dir = data.source_dir;
    results.N = N;
    results.agent_ids = agent_ids;
    results.ref_agent_id = ref_agent_id;
    results.W = W;
    results.U = U;
    results.V = V;
    results.singular_values = sigma;
    results.delta = delta;
    results.modes = mode_results;
    results.total_network = network_results;
    results.agent_wise = agent_results;
    results.total_input_amplitude_sum = total_amp_results;
    results.observed = observed_results;
    results.output_dir = output_dir;

    % 8. Run self-consistency validation
    if opts.RunSyntheticValidation
        run_validation_checks(results, opts);
    end

    % 9. Save Outputs (Figures and CSV tables)
    if opts.SaveOutputs
        fprintf('\n[INFO] Generating and saving figures...\n');
        save_mode_phases_figure(results, output_dir);
        save_total_network_figure(results, output_dir);
        save_agent_wise_figure(results, output_dir);
        save_total_amplitude_sum_figure(results, output_dir);
        if isfield(results, 'observed') && isfield(results.observed, 'available') && results.observed.available
            save_observed_amplitude_sum_timeseries_figure(results, output_dir);
        end

        fprintf('[INFO] Exporting summary CSV files...\n');
        export_results_csv(results, output_dir);

        if opts.SaveMat
            mat_path = fullfile(output_dir, 'optimal_phase_results.mat');
            save(mat_path, '-struct', 'results', '-v7.3');
            fprintf('  Saved MAT file: %s\n', mat_path);
        end
    end

    % 9. Print concise summary table
    print_summary_report(results);
end

% =========================================================================
% 1. OPTIMIZATION ROUTINES
% =========================================================================

function res = optimize_svd_mode_signal(v_l, sigma_l, u_l, ref_idx, opts)
% Optimize phase vector theta to maximize and minimize |sum_j v_{jl} exp(1i*theta_j)|
    N = numel(v_l);
    v_abs = abs(v_l);
    v_sum = sum(v_abs);
    v_max = max(v_abs);
    v_rest = v_sum - v_max;

    % (A) Maximization (Analytic Exact Solution)
    % When arg(v_{jl} * exp(1i*theta_j)) are identical, absolute sum is achieved.
    theta_max_raw = zeros(N, 1);
    for j = 1:N
        if v_l(j) >= 0
            theta_max_raw(j) = 0;
        else
            theta_max_raw(j) = pi;
        end
    end
    % Gauge fix: theta(ref_idx) = 0
    theta_max = wrap_to_pi(theta_max_raw - theta_max_raw(ref_idx));

    Z_max = sum(v_l .* exp(1i * theta_max));
    amp_max = sqrt(2) * abs(Z_max);
    theoretical_max = sqrt(2) * v_sum;

    % (B) Minimization
    % Check polygon inequality: Can a polygon with side lengths |v_{jl}| close?
    can_close_polygon = (v_max <= v_rest + 1e-12);
    if can_close_polygon
        theoretical_min = 0.0;
    else
        theoretical_min = sqrt(2) * (v_max - v_rest);
    end

    % Numerical search for global minimum using analytical gradient
    cost_fun = @(th) mode_cost(th, v_l);
    grad_fun = @(th) mode_grad(th, v_l);

    best_val = Inf;
    best_th = theta_max;

    % Seed initial guesses:
    % 1. Splay state: 2*pi*(0:N-1)/N
    % 2. Multi-start random sampling
    seeds = cell(opts.NumStarts + 2, 1);
    seeds{1} = (0:N-1).' * (2 * pi / N);
    seeds{2} = (0:N-1).' * (pi / N);
    for s = 1:opts.NumStarts
        seeds{s + 2} = rand(N, 1) * 2 * pi - pi;
    end

    for s = 1:numel(seeds)
        th_opt = run_lbfgs_optimizer(seeds{s}, cost_fun, grad_fun, 300, 1e-12);
        val = cost_fun(th_opt);
        if val < best_val
            best_val = val;
            best_th = th_opt;
        end
        if best_val < 1e-13 && can_close_polygon
            break; % Already reached machine zero
        end
    end

    theta_min = wrap_to_pi(best_th - best_th(ref_idx));
    Z_min = sum(v_l .* exp(1i * theta_min));
    amp_min = sqrt(2) * abs(Z_min);

    res = struct();
    res.v_l = v_l;
    res.sigma_l = sigma_l;
    res.u_l = u_l;
    res.theta_max = theta_max;
    res.theta_min = theta_min;
    res.max_amplitude = amp_max;
    res.min_amplitude = amp_min;
    res.theoretical_max = theoretical_max;
    res.theoretical_min = theoretical_min;
    res.Z_max = Z_max;
    res.Z_min = Z_min;
    res.amplitude_ratio = amp_min / max(1e-12, amp_max);
    res.extinction_ratio = 1.0 - res.amplitude_ratio;
    res.can_close_polygon = can_close_polygon;
end

function [f, g] = mode_cost_and_grad(theta, v_l)
    z = exp(1i * theta);
    Z = sum(v_l .* z);
    f = abs(Z)^2; % 0.5 * amp^2
    if nargout > 1
        % d(Z * conj(Z))/d(theta_j) = 2 * Im{ z_j^* * v_{jl} * Z }
        g = 2 * v_l .* (-sin(theta) * real(Z) + cos(theta) * imag(Z));
    end
end

function f = mode_cost(theta, v_l)
    f = mode_cost_and_grad(theta, v_l);
end

function g = mode_grad(theta, v_l)
    [~, g] = mode_cost_and_grad(theta, v_l);
end

function res = optimize_total_network_vibration(W, ref_idx, opts)
% Optimize theta to maximize and minimize J_{total}(theta) = ||W * z||_2^2 = z^H * (W^T * W) * z
    N = size(W, 1);
    Q = W.' * W; % Symmetric positive semidefinite matrix

    cost_fun = @(th) total_network_cost(th, Q);
    grad_fun = @(th) total_network_grad(th, Q);

    % (A) Maximization: minimize -J(theta)
    neg_cost_fun = @(th) -total_network_cost(th, Q);
    neg_grad_fun = @(th) -total_network_grad(th, Q);

    % Seed candidates for maximization
    [V_q, D_q] = eig(Q);
    [~, max_eig_idx] = max(diag(D_q));
    v_dom = V_q(:, max_eig_idx);
    th_eig_max = angle(v_dom);

    best_max_val = -Inf;
    best_max_th = th_eig_max;

    max_seeds = cell(opts.NumStarts + 2, 1);
    max_seeds{1} = th_eig_max;
    max_seeds{2} = (th_eig_max > 0) * pi;
    for s = 1:opts.NumStarts
        max_seeds{s + 2} = rand(N, 1) * 2 * pi - pi;
    end

    for s = 1:numel(max_seeds)
        th_opt = run_lbfgs_optimizer(max_seeds{s}, neg_cost_fun, neg_grad_fun, 300, 1e-12);
        val = cost_fun(th_opt);
        if val > best_max_val
            best_max_val = val;
            best_max_th = th_opt;
        end
    end

    theta_max = wrap_to_pi(best_max_th - best_max_th(ref_idx));
    z_max = exp(1i * theta_max);
    I_max_complex = W * z_max;
    agent_amp_max = sqrt(2) * abs(I_max_complex);
    max_energy = best_max_val;

    % (B) Minimization: minimize +J(theta)
    [~, min_eig_idx] = min(diag(D_q));
    th_eig_min = angle(V_q(:, min_eig_idx));

    best_min_val = Inf;
    best_min_th = th_eig_min;

    min_seeds = cell(opts.NumStarts + 3, 1);
    min_seeds{1} = th_eig_min;
    min_seeds{2} = (0:N-1).' * (2 * pi / N);
    min_seeds{3} = (0:N-1).' * (pi / N);
    for s = 1:opts.NumStarts
        min_seeds{s + 3} = rand(N, 1) * 2 * pi - pi;
    end

    for s = 1:numel(min_seeds)
        th_opt = run_lbfgs_optimizer(min_seeds{s}, cost_fun, grad_fun, 300, 1e-12);
        val = cost_fun(th_opt);
        if val < best_min_val
            best_min_val = val;
            best_min_th = th_opt;
        end
        if best_min_val < 1e-13
            break;
        end
    end

    theta_min = wrap_to_pi(best_min_th - best_min_th(ref_idx));
    z_min = exp(1i * theta_min);
    I_min_complex = W * z_min;
    agent_amp_min = sqrt(2) * abs(I_min_complex);
    min_energy = best_min_val;

    res = struct();
    res.Q = Q;
    res.theta_max = theta_max;
    res.theta_min = theta_min;
    res.max_energy = max_energy;
    res.min_energy = min_energy;
    res.energy_ratio = min_energy / max(1e-12, max_energy);
    res.suppression_percent = 100 * (1 - res.energy_ratio);
    res.agent_amplitudes_at_max = agent_amp_max;
    res.agent_amplitudes_at_min = agent_amp_min;
    res.mean_amp_max = mean(agent_amp_max);
    res.mean_amp_min = mean(agent_amp_min);
end

function [f, g] = total_network_cost_and_grad(theta, Q)
    % J(theta) = sum_{j,k} Q_{jk} * cos(theta_j - theta_k)
    % grad_j J = -2 * sum_k Q_{jk} * sin(theta_j - theta_k)
    dTheta = theta - theta.'; % N x N, (j,k) is theta_j - theta_k
    cos_mat = cos(dTheta);
    f = sum(sum(Q .* cos_mat));
    if nargout > 1
        sin_mat = sin(dTheta);
        g = -2 * sum(Q .* sin_mat, 2);
    end
end

function f = total_network_cost(theta, Q)
    f = total_network_cost_and_grad(theta, Q);
end

function g = total_network_grad(theta, Q)
    [~, g] = total_network_cost_and_grad(theta, Q);
end

function res = optimize_agent_wise_vibration(w_row, target_idx, ref_idx, opts)
% Optimize phase of senders j ~= target_idx to maximize/minimize input to agent target_idx
    N = numel(w_row);
    w_vec = w_row(:);
    w_vec(target_idx) = 0; % Ensure self is zero

    w_abs = abs(w_vec);
    w_sum = sum(w_abs);
    w_max = max(w_abs);
    w_rest = w_sum - w_max;

    % Maximization: senders in phase with sign(W_{ij})
    theta_max_raw = zeros(N, 1);
    for j = 1:N
        if w_vec(j) >= 0
            theta_max_raw(j) = 0;
        else
            theta_max_raw(j) = pi;
        end
    end
    theta_max = wrap_to_pi(theta_max_raw - theta_max_raw(ref_idx));
    Y_max = sum(w_vec .* exp(1i * theta_max));
    amp_max = sqrt(2) * abs(Y_max);
    theoretical_max = sqrt(2) * w_sum;

    % Minimization:
    can_close = (w_max <= w_rest + 1e-12);
    if can_close
        theoretical_min = 0.0;
    else
        theoretical_min = sqrt(2) * (w_max - w_rest);
    end

    cost_fun = @(th) mode_cost(th, w_vec);
    grad_fun = @(th) mode_grad(th, w_vec);

    best_val = Inf;
    best_th = theta_max;
    for s = 1:opts.NumStarts
        th_init = rand(N, 1) * 2 * pi - pi;
        th_opt = run_lbfgs_optimizer(th_init, cost_fun, grad_fun, 250, 1e-12);
        val = cost_fun(th_opt);
        if val < best_val
            best_val = val;
            best_th = th_opt;
        end
        if best_val < 1e-13 && can_close
            break;
        end
    end

    theta_min = wrap_to_pi(best_th - best_th(ref_idx));
    Y_min = sum(w_vec .* exp(1i * theta_min));
    amp_min = sqrt(2) * abs(Y_min);

    res = struct();
    res.target_idx = target_idx;
    res.w_incoming = w_vec;
    res.theta_max = theta_max;
    res.theta_min = theta_min;
    res.max_amplitude = amp_max;
    res.min_amplitude = amp_min;
    res.theoretical_max = theoretical_max;
    res.theoretical_min = theoretical_min;
    res.amplitude_ratio = amp_min / max(1e-12, amp_max);
    res.can_close_polygon = can_close;
end

function res = optimize_total_input_amplitude_sum(W, ref_idx, mode_results, network_results, opts)
% OPTIMIZE_TOTAL_INPUT_AMPLITUDE_SUM Calculates the phase relationships that
% minimize and maximize the sum of input vibration amplitudes across all receiver modules:
%
%   x = exp(1i * theta);
%   G = W * x;
%   agent_amplitudes = sqrt(2) * abs(G);
%   J_amp = sum(agent_amplitudes);
%
% Optimizes over the remaining N-1 variables with theta(ref_idx) fixed to 0.
% Uses smooth continuation J_smooth = sqrt(2) * sum(sqrt(abs(G).^2 + eps^2)) with
% analytical gradients. Directly evaluates all initial seeds to ensure no degradation.

    N = size(W, 1);
    W_scale = max(abs(W(:)));

    % Zero matrix edge case handling
    if W_scale < 1e-14
        theta_zero = zeros(N, 1);
        res = struct();
        res.theta_min = theta_zero;
        res.theta_max = theta_zero;
        res.min_amplitude_sum = 0.0;
        res.max_amplitude_sum = 0.0;
        res.agent_amplitudes_at_min = zeros(N, 1);
        res.agent_amplitudes_at_max = zeros(N, 1);
        res.J2_at_min = 0.0;
        res.J2_at_max = 0.0;
        res.amplitude_ratio = 1.0;
        res.suppression_percent = 0.0;
        res.best_seed_min = 'zero_matrix';
        res.best_seed_max = 'zero_matrix';
        return;
    end

    opt_indices = setdiff(1:N, ref_idx);
    N_opt = numel(opt_indices);

    % Helper for exact J_amp calculation
    exact_cost = @(th) sum(sqrt(2) * abs(W * exp(1i * th)));

    % --- Gather Initial Seed Candidates ---
    seed_list = {};
    seed_names = {};

    % 1. Network quadratic (J2) optimal states
    if isfield(network_results, 'theta_min') && ~isempty(network_results.theta_min)
        seed_list{end+1} = wrap_to_pi(network_results.theta_min - network_results.theta_min(ref_idx));
        seed_names{end+1} = 'network_J2_min';
    end
    if isfield(network_results, 'theta_max') && ~isempty(network_results.theta_max)
        seed_list{end+1} = wrap_to_pi(network_results.theta_max - network_results.theta_max(ref_idx));
        seed_names{end+1} = 'network_J2_max';
    end

    % 2. SVD Mode-wise candidates
    if ~isempty(mode_results)
        for l = 1:min(N, numel(mode_results))
            if isfield(mode_results(l), 'theta_max')
                seed_list{end+1} = wrap_to_pi(mode_results(l).theta_max - mode_results(l).theta_max(ref_idx));
                seed_names{end+1} = sprintf('mode_%d_max', l);
            end
            if isfield(mode_results(l), 'theta_min')
                seed_list{end+1} = wrap_to_pi(mode_results(l).theta_min - mode_results(l).theta_min(ref_idx));
                seed_names{end+1} = sprintf('mode_%d_min', l);
            end
        end
    end

    % 3. In-phase configuration (all zeros)
    seed_list{end+1} = zeros(N, 1);
    seed_names{end+1} = 'in_phase_zeros';

    % 4. Splay states (uniform phase spacing)
    seed_list{end+1} = wrap_to_pi((0:N-1).' * (2 * pi / N));
    seed_names{end+1} = 'splay_forward';
    seed_list{end+1} = wrap_to_pi((0:N-1).' * (-2 * pi / N));
    seed_names{end+1} = 'splay_backward';

    % 5. Bimodal / alternating states
    alt_state = zeros(N, 1);
    alt_state(2:2:end) = pi;
    seed_list{end+1} = wrap_to_pi(alt_state);
    seed_names{end+1} = 'alternating_pi';

    % 6. Multi-start random seeds
    for s = 1:opts.NumStarts
        th_rand = zeros(N, 1);
        th_rand(opt_indices) = rand(N_opt, 1) * 2 * pi - pi;
        seed_list{end+1} = th_rand;
        seed_names{end+1} = sprintf('random_%d', s);
    end

    num_seeds = numel(seed_list);

    % --- Evaluate Initial Seeds (Ensure good initial candidates are never discarded) ---
    best_min_val = Inf;
    best_min_theta = zeros(N, 1);
    best_min_seed = '';

    best_max_val = -Inf;
    best_max_theta = zeros(N, 1);
    best_max_seed = '';

    for k = 1:num_seeds
        th_k = seed_list{k};
        th_k = wrap_to_pi(th_k - th_k(ref_idx));
        val_k = exact_cost(th_k);
        if val_k < best_min_val
            best_min_val = val_k;
            best_min_theta = th_k;
            best_min_seed = seed_names{k};
        end
        if val_k > best_max_val
            best_max_val = val_k;
            best_max_theta = th_k;
            best_max_seed = seed_names{k};
        end
    end

    % --- Smooth Continuation Optimization ---
    % Scale epsilon continuation schedule by max coupling weight
    eps_levels = [1e-2, 1e-4, 1e-6] * W_scale;

    % (A) Minimization
    for k = 1:num_seeds
        th_cur = seed_list{k};
        th_cur = wrap_to_pi(th_cur - th_cur(ref_idx));
        psi_cur = th_cur(opt_indices);

        for ep = eps_levels
            cost_fn = @(p) smooth_amp_cost_and_grad(p, W, ref_idx, opt_indices, ep, +1);
            grad_fn = @(p) smooth_amp_grad_only(p, W, ref_idx, opt_indices, ep, +1);
            psi_cur = run_lbfgs_optimizer(psi_cur, cost_fn, grad_fn, 150, 1e-10);

            % Evaluate at the end of each continuation stage with exact J_amp
            th_stage = zeros(N, 1);
            th_stage(opt_indices) = psi_cur;
            th_stage = wrap_to_pi(th_stage - th_stage(ref_idx));
            val_stage = exact_cost(th_stage);
            if val_stage < best_min_val
                best_min_val = val_stage;
                best_min_theta = th_stage;
                best_min_seed = sprintf('%s+lbfgs_eps_%.1e', seed_names{k}, ep);
            end
        end
    end

    % (B) Maximization
    for k = 1:num_seeds
        th_cur = seed_list{k};
        th_cur = wrap_to_pi(th_cur - th_cur(ref_idx));
        psi_cur = th_cur(opt_indices);

        for ep = eps_levels
            cost_fn = @(p) smooth_amp_cost_and_grad(p, W, ref_idx, opt_indices, ep, -1);
            grad_fn = @(p) smooth_amp_grad_only(p, W, ref_idx, opt_indices, ep, -1);
            psi_cur = run_lbfgs_optimizer(psi_cur, cost_fn, grad_fn, 150, 1e-10);

            % Evaluate at the end of each continuation stage with exact J_amp
            th_stage = zeros(N, 1);
            th_stage(opt_indices) = psi_cur;
            th_stage = wrap_to_pi(th_stage - th_stage(ref_idx));
            val_stage = exact_cost(th_stage);
            if val_stage > best_max_val
                best_max_val = val_stage;
                best_max_theta = th_stage;
                best_max_seed = sprintf('%s+lbfgs_eps_%.1e', seed_names{k}, ep);
            end
        end
    end

    % Final metrics assembly
    theta_min = wrap_to_pi(best_min_theta - best_min_theta(ref_idx));
    theta_max = wrap_to_pi(best_max_theta - best_max_theta(ref_idx));

    G_min = W * exp(1i * theta_min);
    G_max = W * exp(1i * theta_max);

    agent_amp_min = sqrt(2) * abs(G_min);
    agent_amp_max = sqrt(2) * abs(G_max);

    min_amp_sum = sum(agent_amp_min);
    max_amp_sum = sum(agent_amp_max);

    J2_min = sum(abs(G_min).^2);
    J2_max = sum(abs(G_max).^2);

    res = struct();
    res.theta_min = theta_min;
    res.theta_max = theta_max;
    res.min_amplitude_sum = min_amp_sum;
    res.max_amplitude_sum = max_amp_sum;
    res.agent_amplitudes_at_min = agent_amp_min;
    res.agent_amplitudes_at_max = agent_amp_max;
    res.J2_at_min = J2_min;
    res.J2_at_max = J2_max;
    res.amplitude_ratio = min_amp_sum / max(1e-12, max_amp_sum);
    res.suppression_percent = 100 * (1 - res.amplitude_ratio);
    res.best_seed_min = best_min_seed;
    res.best_seed_max = best_max_seed;
end

function [f, g] = smooth_amp_cost_and_grad(psi, W, ref_idx, opt_indices, epsilon, sign_factor)
% Smooth amplitude sum cost and analytical gradient with respect to N-1 variables psi
    N = size(W, 1);
    theta = zeros(N, 1);
    theta(opt_indices) = psi;

    x = exp(1i * theta);
    G = W * x; % N x 1
    G_mag_sq = real(G .* conj(G)); % |G_k|^2

    smoothed_mag = sqrt(G_mag_sq + epsilon^2);
    f = sign_factor * sum(sqrt(2) * smoothed_mag);

    if nargout > 1
        % Analytical gradient
        % d J_smooth / d theta_j = -sqrt(2) * sum_k (W_kj / smoothed_mag_k) * Im(x_j * G_k^*)
        w_vec = sqrt(2) ./ smoothed_mag;
        u_vec = w_vec .* G;
        g_full = -sign_factor * imag(x .* (W.' * conj(u_vec)));
        g = g_full(opt_indices);
    end
end

function g = smooth_amp_grad_only(psi, W, ref_idx, opt_indices, epsilon, sign_factor)
    [~, g] = smooth_amp_cost_and_grad(psi, W, ref_idx, opt_indices, epsilon, sign_factor);
end

% =========================================================================
% 2. NUMERICAL OPTIMIZER (Self-contained, zero-dependency L-BFGS)
% =========================================================================

function [x, f, exitflag] = run_lbfgs_optimizer(x0, cost_fn, grad_fn, max_iter, tol)
% Compact self-contained gradient descent / quasi-Newton optimizer with
% robust Armijo line-search and NaN/Inf rejection.
    x = x0;
    n = numel(x0);
    m_mem = 5;
    s_hist = zeros(n, m_mem);
    y_hist = zeros(n, m_mem);
    rho_hist = zeros(m_mem, 1);
    hist_count = 0;
    exitflag = 2; % default: reached max_iter

    f = cost_fn(x);
    if isnan(f) || isinf(f)
        exitflag = -1;
        return;
    end
    g = grad_fn(x);
    if any(isnan(g)) || any(isinf(g))
        exitflag = -2;
        return;
    end

    for iter = 1:max_iter
        if norm(g) < tol
            exitflag = 1; % converged on gradient tolerance
            break;
        end

        % Two-loop recursion to compute search direction p = -H * g
        p = -g;
        if hist_count > 0
            k_len = min(hist_count, m_mem);
            alphas = zeros(k_len, 1);
            q = g;
            for i = k_len:-1:1
                idx = mod(hist_count - (k_len - i) - 1, m_mem) + 1;
                alphas(i) = rho_hist(idx) * (s_hist(:, idx).' * q);
                q = q - alphas(i) * y_hist(:, idx);
            end
            % Initial Hessian estimate gamma
            idx_last = mod(hist_count - 1, m_mem) + 1;
            denom = y_hist(:, idx_last).' * y_hist(:, idx_last);
            if denom > 1e-14
                gamma_k = (s_hist(:, idx_last).' * y_hist(:, idx_last)) / denom;
                r = gamma_k * q;
            else
                r = q;
            end
            for i = 1:k_len
                idx = mod(hist_count - (k_len - i) - 1, m_mem) + 1;
                beta = rho_hist(idx) * (y_hist(:, idx).' * r);
                r = r + s_hist(:, idx) * (alphas(i) - beta);
            end
            p = -r;
            % Ensure descent direction
            if p.' * g >= 0
                p = -g;
            end
        end

        % Backtracking Armijo Line Search
        alpha_step = 1.0;
        c1 = 1e-4;
        f_old = f;
        g_old = g;
        x_old = x;
        dphi_0 = g_old.' * p;
        if dphi_0 >= 0
            % Not a descent direction, reset to steepest descent
            p = -g_old;
            dphi_0 = -norm(g_old)^2;
        end

        step_accepted = false;
        ls_iter = 0;
        while ls_iter < 25
            x_trial = x_old + alpha_step * p;
            f_trial = cost_fn(x_trial);
            if ~isnan(f_trial) && ~isinf(f_trial) && (f_trial <= f_old + c1 * alpha_step * dphi_0)
                step_accepted = true;
                break;
            end
            alpha_step = alpha_step * 0.5;
            ls_iter = ls_iter + 1;
        end

        if ~step_accepted
            % Line search failed: retain accepted point, do NOT register failed step in L-BFGS history, and terminate
            exitflag = 0;
            break;
        end

        g_trial = grad_fn(x_trial);
        if any(isnan(g_trial)) || any(isinf(g_trial))
            % Gradient evaluation failed: terminate
            exitflag = -2;
            break;
        end

        % Accept step
        x = x_trial;
        f = f_trial;
        g = g_trial;

        s = x - x_old;
        y = g - g_old;
        sy = s.' * y;
        if sy > 1e-12
            hist_count = hist_count + 1;
            idx_cur = mod(hist_count - 1, m_mem) + 1;
            s_hist(:, idx_cur) = s;
            y_hist(:, idx_cur) = y;
            rho_hist(idx_cur) = 1.0 / sy;
        end

        if abs(f_old - f) < 1e-14 && norm(s) < 1e-12
            exitflag = 1;
            break;
        end
    end
end

% =========================================================================
% 3. DATA LOADING AND RESOLUTION
% =========================================================================

function data = load_exported_mode_data(source)
    data = struct();
    if isstruct(source) && isfield(source, 'W') && isfield(source, 'svd')
        % Source is already a results struct
        data.source_dir = pwd;
        if isfield(source, 'output_dir'), data.source_dir = source.output_dir; end
        raw_ids = source.agent_ids(:).';
        [agent_ids, sort_order] = sort(raw_ids);
        data.N = numel(agent_ids);
        data.agent_ids = agent_ids;
        data.W = source.W(sort_order, sort_order);
        data.U = source.svd.U(sort_order, :);
        data.V = source.svd.V(sort_order, :);
        data.singular_values = source.svd.singular_values(:);
        if isfield(source, 'fit_R1') && isfield(source.fit_R1, 'delta')
            data.delta = source.fit_R1.delta(1);
        else
            data.delta = 0.0;
        end
        validate_mode_data_integrity(data);
        return;
    end

    % Source is a folder path or relative name
    dir_path = char(source);
    target_dir = resolve_mode_directory(dir_path);
    if isempty(target_dir) || ~exist(target_dir, 'dir')
        error('calculate_input_vibration_optimal_phases:DirNotFound', ...
            ['Could not resolve exported SVD mode directory for dataset: "%s".\n' ...
             'Please make sure global_joint_cp_rank1_profile_free_network_svd(''%s'') ' ...
             'has been executed to export network_coupling_matrix_W.csv.'], ...
            dir_path, dir_path);
    end

    % 1. Read network_coupling_matrix_W.csv
    w_path = fullfile(target_dir, 'network_coupling_matrix_W.csv');
    if ~isfile(w_path)
        error('calculate_input_vibration_optimal_phases:MissingW', ...
            'Required file network_coupling_matrix_W.csv not found in: %s', target_dir);
    end
    t_w = readtable(w_path);
    col_names = t_w.Properties.VariableNames;

    % Parse sender columns
    sender_cols = startsWith(col_names, 'sender_agent_');
    if ~any(sender_cols)
        error('calculate_input_vibration_optimal_phases:InvalidWFormat', ...
            'network_coupling_matrix_W.csv contains no columns matching "sender_agent_<ID>".');
    end
    sender_col_names = col_names(sender_cols);
    N_senders = numel(sender_col_names);
    sender_ids = zeros(1, N_senders);
    for j = 1:N_senders
        tok = regexp(sender_col_names{j}, 'sender_agent_(\d+)', 'tokens');
        if isempty(tok)
            error('calculate_input_vibration_optimal_phases:InvalidSenderCol', ...
                'Cannot parse sender agent ID from column name: %s', sender_col_names{j});
        end
        sender_ids(j) = str2double(tok{1}{1});
    end

    % Check duplicate sender IDs
    if numel(unique(sender_ids)) ~= N_senders
        error('calculate_input_vibration_optimal_phases:DuplicateSenderIDs', ...
            'Duplicate sender agent IDs detected in network_coupling_matrix_W.csv: %s', mat2str(sender_ids));
    end

    % Parse receiver rows from first column (or receiver_agent column)
    if ismember('receiver_agent', col_names)
        rec_col_name = 'receiver_agent';
    else
        rec_col_name = col_names{1};
    end
    rec_raw = t_w.(rec_col_name);
    N_receivers = height(t_w);
    receiver_ids = zeros(1, N_receivers);
    for i = 1:N_receivers
        val_i = rec_raw(i);
        if iscell(val_i), val_str = char(val_i{1});
        elseif isstring(val_i), val_str = char(val_i);
        elseif isnumeric(val_i), val_str = num2str(val_i);
        else, val_str = char(string(val_i));
        end
        tok = regexp(val_str, '(\d+)', 'tokens');
        if isempty(tok)
            error('calculate_input_vibration_optimal_phases:InvalidReceiverRow', ...
                'Cannot parse receiver agent ID from row %d ("%s") in %s', i, val_str, w_path);
        end
        receiver_ids(i) = str2double(tok{end}{1});
    end

    % Check duplicate receiver IDs
    if numel(unique(receiver_ids)) ~= N_receivers
        error('calculate_input_vibration_optimal_phases:DuplicateReceiverIDs', ...
            'Duplicate receiver agent IDs detected in network_coupling_matrix_W.csv: %s', mat2str(receiver_ids));
    end

    % Receiver and sender IDs must match as sets
    if N_receivers ~= N_senders || ~isempty(setxor(receiver_ids, sender_ids))
        error('calculate_input_vibration_optimal_phases:ReceiverSenderMismatch', ...
            ['Mismatch between receiver IDs (%s) and sender IDs (%s) in ' ...
             'network_coupling_matrix_W.csv.'], mat2str(receiver_ids), mat2str(sender_ids));
    end

    % Canonical sorted Agent IDs
    agent_ids = sort(sender_ids(:).');
    N = numel(agent_ids);

    % Align rows and columns of W to canonical agent_ids order
    [~, row_order_w] = ismember(agent_ids, receiver_ids);
    [~, col_order_w] = ismember(agent_ids, sender_ids);
    raw_W = table2array(t_w(:, sender_cols));
    W_mat = raw_W(row_order_w, col_order_w);

    % 2. Read network_svd_modes_summary.csv
    modes_path = fullfile(target_dir, 'network_svd_modes_summary.csv');
    if isfile(modes_path)
        t_modes = readtable(modes_path);
        if ~ismember('mode_l', t_modes.Properties.VariableNames) || ...
           ~ismember('singular_value_sigma', t_modes.Properties.VariableNames)
            error('calculate_input_vibration_optimal_phases:InvalidModesSummary', ...
                'network_svd_modes_summary.csv missing required columns: mode_l, singular_value_sigma');
        end
        mode_l_vals = t_modes.mode_l(:).';
        if numel(unique(mode_l_vals)) ~= numel(mode_l_vals)
            error('calculate_input_vibration_optimal_phases:DuplicateModes', ...
                'Duplicate mode numbers detected in network_svd_modes_summary.csv');
        end
        if numel(mode_l_vals) < N || any(~ismember(1:N, mode_l_vals))
            error('calculate_input_vibration_optimal_phases:MissingModes', ...
                'network_svd_modes_summary.csv must contain modes 1..%d without gaps', N);
        end
        [~, mode_order_sigma] = ismember(1:N, mode_l_vals);
        sigma = double(t_modes.singular_value_sigma(mode_order_sigma(:)));
    else
        [~, S_mat, ~] = svd(W_mat);
        sigma = diag(S_mat);
    end

    % 3. Read agent_svd_contributions.csv
    agent_path = fullfile(target_dir, 'agent_svd_contributions.csv');
    if isfile(agent_path)
        t_agent = readtable(agent_path);
        if ~ismember('agent_id', t_agent.Properties.VariableNames)
            error('calculate_input_vibration_optimal_phases:MissingAgentIdCol', ...
                'agent_svd_contributions.csv must contain "agent_id" column.');
        end
        csv_agent_ids = t_agent.agent_id(:).';
        if numel(unique(csv_agent_ids)) ~= numel(csv_agent_ids)
            error('calculate_input_vibration_optimal_phases:DuplicateAgentContribIDs', ...
                'Duplicate agent IDs detected in agent_svd_contributions.csv: %s', mat2str(csv_agent_ids));
        end
        [found_ids, row_order_contrib] = ismember(agent_ids, csv_agent_ids);
        if ~all(found_ids)
            missing_ids = agent_ids(~found_ids);
            error('calculate_input_vibration_optimal_phases:MissingContribAgentIDs', ...
                'agent_svd_contributions.csv is missing agent IDs: %s', mat2str(missing_ids));
        end

        U = zeros(N, N);
        V = zeros(N, N);
        for l = 1:N
            u_name = sprintf('receiver_u_mode%d', l);
            v_name = sprintf('sender_v_mode%d', l);
            if ~ismember(u_name, t_agent.Properties.VariableNames)
                error('calculate_input_vibration_optimal_phases:MissingUCol', ...
                    'agent_svd_contributions.csv is missing column "%s" for mode %d. Silently zero-padding is forbidden.', ...
                    u_name, l);
            end
            if ~ismember(v_name, t_agent.Properties.VariableNames)
                error('calculate_input_vibration_optimal_phases:MissingVCol', ...
                    'agent_svd_contributions.csv is missing column "%s" for mode %d. Silently zero-padding is forbidden.', ...
                    v_name, l);
            end
            u_raw = double(t_agent.(u_name));
            v_raw = double(t_agent.(v_name));
            % Align rows to agent_ids canonical ordering
            U(:, l) = u_raw(row_order_contrib);
            V(:, l) = v_raw(row_order_contrib);
        end
    else
        [U, ~, V] = svd(W_mat);
    end

    % 4. Read sender_phase_shift_delta.csv
    delta_path = fullfile(target_dir, 'sender_phase_shift_delta.csv');
    if isfile(delta_path)
        t_delta = readtable(delta_path);
        if ismember('delta_rad', t_delta.Properties.VariableNames)
            delta = double(t_delta.delta_rad(1));
        elseif ismember('delta', t_delta.Properties.VariableNames)
            delta = double(t_delta.delta(1));
        else
            delta = 0.0;
        end
    else
        delta = 0.0;
    end

    data.source_dir = target_dir;
    data.N = N;
    data.agent_ids = agent_ids;
    data.W = W_mat;
    data.U = U;
    data.V = V;
    data.singular_values = sigma;
    data.delta = delta;

    validate_mode_data_integrity(data);
end

function validate_mode_data_integrity(data)
    N = data.N;
    if numel(data.agent_ids) ~= N
        error('calculate_input_vibration_optimal_phases:IntegrityError', ...
            'Number of agent IDs (%d) does not match N=%d', numel(data.agent_ids), N);
    end
    if numel(unique(data.agent_ids)) ~= N
        error('calculate_input_vibration_optimal_phases:DuplicateAgentIDs', ...
            'Agent IDs contain duplicates: %s', mat2str(data.agent_ids));
    end
    if any(size(data.W) ~= [N, N])
        error('calculate_input_vibration_optimal_phases:IntegrityError', ...
            'W matrix size [%d x %d] does not match N=%d', size(data.W, 1), size(data.W, 2), N);
    end
    if any(size(data.U) ~= [N, N]) || any(size(data.V) ~= [N, N])
        error('calculate_input_vibration_optimal_phases:IntegrityError', ...
            'U [%d x %d] or V [%d x %d] size does not match [%d x %d]', ...
            size(data.U, 1), size(data.U, 2), size(data.V, 1), size(data.V, 2), N, N);
    end
    if numel(data.singular_values) ~= N
        error('calculate_input_vibration_optimal_phases:IntegrityError', ...
            'Singular values count (%d) does not match N=%d', numel(data.singular_values), N);
    end
end

function target_dir = resolve_mode_directory(p)
    if ischar(p) || isstring(p)
        p_str = char(p);
    else
        target_dir = '';
        return;
    end

    base_root = fileparts(fileparts(mfilename('fullpath'))); % e.g., C:\Users\nishikaw\Codes\Volvocine_PicoV2
    estimate_dir = fileparts(mfilename('fullpath'));         % e.g., C:\Users\nishikaw\Codes\Volvocine_PicoV2\EstimateL

    candidates = {
        p_str, ...
        fullfile(base_root, p_str), ...
        fullfile(estimate_dir, p_str), ...
        fullfile(p_str, 'global_joint_cp_rank1_profile_free_network_svd'), ...
        fullfile(base_root, p_str, 'global_joint_cp_rank1_profile_free_network_svd'), ...
        fullfile(estimate_dir, p_str, 'global_joint_cp_rank1_profile_free_network_svd'), ...
        fullfile(p_str, 'low_rank_analysis', 'M10', 'global_joint_cp_rank1_profile_free_network_svd'), ...
        fullfile(base_root, p_str, 'low_rank_analysis', 'M10', 'global_joint_cp_rank1_profile_free_network_svd'), ...
        fullfile(estimate_dir, p_str, 'low_rank_analysis', 'M10', 'global_joint_cp_rank1_profile_free_network_svd'), ...
        fullfile(base_root, 'EstimateL', p_str, 'low_rank_analysis', 'M10', 'global_joint_cp_rank1_profile_free_network_svd')
    };

    target_dir = '';
    for c = 1:numel(candidates)
        cand = candidates{c};
        if isfile(fullfile(cand, 'network_coupling_matrix_W.csv'))
            target_dir = char(cand);
            return;
        end
    end

    % Dynamic search across any M* directory under low_rank_analysis
    base_dirs = {
        p_str, ...
        fullfile(base_root, p_str), ...
        fullfile(estimate_dir, p_str), ...
        fullfile(base_root, 'EstimateL', p_str)
    };
    for b = 1:numel(base_dirs)
        low_rank_dir = fullfile(base_dirs{b}, 'low_rank_analysis');
        if exist(low_rank_dir, 'dir')
            m_dirs = dir(fullfile(low_rank_dir, 'M*'));
            for k = 1:numel(m_dirs)
                if m_dirs(k).isdir
                    cand = fullfile(low_rank_dir, m_dirs(k).name, 'global_joint_cp_rank1_profile_free_network_svd');
                    if isfile(fullfile(cand, 'network_coupling_matrix_W.csv'))
                        target_dir = char(cand);
                        return;
                    end
                end
            end
        end
    end
end

function obs = evaluate_observed_phases(data, W, V, sigma, delta)
    obs = struct('available', false);
    cur_dir = data.source_dir;
    found_cache = false;
    cache_path = '';

    for depth = 1:5
        cand_cache = fullfile(cur_dir, 'phase_analysis_cache.mat');
        if isfile(cand_cache)
            found_cache = true;
            cache_path = cand_cache;
            break;
        end
        parent = fileparts(cur_dir);
        if strcmp(parent, cur_dir), break; end
        cur_dir = parent;
    end

    if ~found_cache
        obs.skip_reason = 'phase_analysis_cache.mat not found in parent folders.';
        fprintf('[INFO] Observed phase comparison skipped: %s\n', obs.skip_reason);
        return;
    end

    try
        cached = load(cache_path);
    catch ME
        obs.skip_reason = sprintf('Failed to load %s: %s', cache_path, ME.message);
        fprintf('[INFO] Observed phase comparison skipped: %s\n', obs.skip_reason);
        return;
    end

    if ~isfield(cached, 'time_sec') || ~isfield(cached, 'phase_matrix')
        obs.skip_reason = 'Cached MAT missing required time_sec or phase_matrix fields.';
        fprintf('[INFO] Observed phase comparison skipped: %s\n', obs.skip_reason);
        return;
    end

    if ~isfield(cached, 'agent_ids')
        obs.skip_reason = 'Cached phase data missing "agent_ids" metadata; cannot verify correspondence to model sender agents.';
        fprintf('[INFO] Observed phase comparison skipped: %s\n', obs.skip_reason);
        return;
    end

    cache_agent_ids = cached.agent_ids(:).';
    [found_all, loc_in_cache] = ismember(data.agent_ids, cache_agent_ids);
    if ~all(found_all)
        missing_ids = data.agent_ids(~found_all);
        obs.skip_reason = sprintf('Cached phase data missing required sender agent IDs: %s', mat2str(missing_ids));
        fprintf('[INFO] Observed phase comparison skipped: %s\n', obs.skip_reason);
        return;
    end

    t = cached.time_sec(:);
    phi_mat = cached.phase_matrix; % T x N_cache
    if size(phi_mat, 2) < max(loc_in_cache)
        obs.skip_reason = sprintf('phase_matrix column count (%d) is smaller than required agent index (%d).', ...
            size(phi_mat, 2), max(loc_in_cache));
        fprintf('[INFO] Observed phase comparison skipped: %s\n', obs.skip_reason);
        return;
    end

    T_len = numel(t);
    % Subsample for efficiency
    step = max(1, floor(T_len / 1000));
    idx_sub = 1:step:T_len;
    t_sub = t(idx_sub);
    phi_sub = phi_mat(idx_sub, :);

    % Align columns of phi to match model sender Agent ID ordering: data.agent_ids
    phi_aligned = phi_sub(:, loc_in_cache);

    % Instantaneous complex inputs G_t = exp(1i * phi_aligned) * W.' (T_sub x N)
    G_t = exp(1i * phi_aligned) * W.';
    agent_amps_t = sqrt(2) * abs(G_t);       % T_sub x N
    J_amp_t = sum(agent_amps_t, 2);          % T_sub x 1
    J_amp_mean = mean(J_amp_t);

    % Instantaneous quadratic vibration energy: J2(t) = sum_i |G_ti|^2
    J2_t = sum(abs(G_t).^2, 2);              % T_sub x 1
    J2_mean = mean(J2_t);

    % SVD mode signals X_l(t)
    b_sub = sqrt(2) * cos(phi_aligned - delta); % T_sub x N
    X_l_obs = b_sub * V;                        % T_sub x N
    X_amp_mean = sqrt(2) * sqrt(mean(X_l_obs.^2, 1)).';

    % Legacy real-signal time-average <I_i(t)^2>
    I_obs = b_sub * W.';
    I_amp_mean = sqrt(2) * sqrt(mean(I_obs.^2, 1)).';
    total_energy_mean = sum(mean(I_obs.^2, 1));

    obs.available = true;
    obs.time_sec = t_sub;
    obs.phi_sample = phi_aligned;
    obs.G_t = G_t;
    obs.J_amp_t = J_amp_t;
    obs.J_amp_mean = J_amp_mean;
    obs.J2_t = J2_t;
    obs.J2_mean = J2_mean;
    obs.agent_amps_t = agent_amps_t;
    obs.agent_amps_mean = mean(agent_amps_t, 1).';
    obs.X_l_mean_amplitude = X_amp_mean;
    obs.I_i_mean_amplitude = I_amp_mean;
    obs.total_energy_mean = total_energy_mean;
end

% =========================================================================
% 4. SELF-CONSISTENCY VALIDATION SUITE
% =========================================================================

function run_validation_checks(results, opts)
    fprintf('[VALIDATION] Running mathematical consistency checks...\n');
    N = results.N;

    % 1. Check triangle inequality and maximum amplitude exactness
    for l = 1:N
        m = results.modes(l);
        v_l = results.V(:, l);
        assert(abs(m.max_amplitude - m.theoretical_max) < 1e-10, ...
            sprintf('Mode %d max amplitude (%.6f) does not match theoretical max (%.6f)', ...
            l, m.max_amplitude, m.theoretical_max));
        assert(m.min_amplitude <= m.max_amplitude + 1e-12, ...
            sprintf('Mode %d min amplitude exceeds max amplitude!', l));
        if m.can_close_polygon
            assert(m.min_amplitude < 1e-6, ...
                sprintf('Mode %d polygon can close but min amplitude is non-zero (%.2e)!', ...
                l, m.min_amplitude));
        end
    end
    fprintf('  * Mode-wise triangle inequality & polygon closure: PASSED\n');

    % 2. Check total network energy stationary points (grad norm near zero)
    th_max = results.total_network.theta_max;
    th_min = results.total_network.theta_min;
    Q = results.total_network.Q;
    g_max = total_network_grad(th_max, Q);
    g_min = total_network_grad(th_min, Q);
    % Projected gradient norm (removing gauge direction 1_N)
    proj_g_max = g_max - mean(g_max);
    proj_g_min = g_min - mean(g_min);
    assert(norm(proj_g_max) < 1e-4, 'Total network max phase is not a stationary point!');
    assert(norm(proj_g_min) < 1e-4, 'Total network min phase is not a stationary point!');
    assert(results.total_network.max_energy >= results.total_network.min_energy - 1e-12, ...
        'Total network max energy < min energy!');
    fprintf('  * Total network energy stationary points & bounds: PASSED\n');

    % 3. Check gauge invariance for quadratic energy and amplitude sum
    shift_angle = 1.2345;
    cost_shifted = total_network_cost(th_max + shift_angle, Q);
    assert(abs(cost_shifted - results.total_network.max_energy) < 1e-11, ...
        'Gauge invariance check failed for total network energy!');
    
    th_amp_min = results.total_input_amplitude_sum.theta_min;
    th_amp_max = results.total_input_amplitude_sum.theta_max;
    amp_sum_exact = @(th) sum(sqrt(2) * abs(results.W * exp(1i * th)));
    assert(abs(amp_sum_exact(th_amp_min + shift_angle) - results.total_input_amplitude_sum.min_amplitude_sum) < 1e-11, ...
        'Gauge invariance check failed for total input amplitude sum (min)!');
    assert(abs(amp_sum_exact(th_amp_max + shift_angle) - results.total_input_amplitude_sum.max_amplitude_sum) < 1e-11, ...
        'Gauge invariance check failed for total input amplitude sum (max)!');
    fprintf('  * Gauge invariance under global phase shifts:     PASSED\n');

    % 4. Synthetic Test Matrix: W_test
    % Verify J2 is constant (= 4) while J_amp varies between 2*sqrt(2) and 4.
    W_test = [
        0  0  0  0;
        0  0  0  0;
        1  1  0  0;
        1 -1  0  0
    ];
    test_opts = opts;
    test_opts.NumStarts = 40;
    test_opts.RandomSeed = 42;

    % Preserve RNG state before running synthetic test with fixed seed
    orig_rng = rng();
    rng(test_opts.RandomSeed);
    test_res = optimize_total_input_amplitude_sum(W_test, 1, [], [], test_opts);
    rng(orig_rng); % Restore original RNG state

    % (a) J2 must be identically 4 for both min and max states
    assert(abs(test_res.J2_at_min - 4.0) < 1e-5, ...
        sprintf('W_test J2_at_min mismatch: expected 4.0, got %.6f', test_res.J2_at_min));
    assert(abs(test_res.J2_at_max - 4.0) < 1e-5, ...
        sprintf('W_test J2_at_max mismatch: expected 4.0, got %.6f', test_res.J2_at_max));

    % (b) J_amp minimum must be 2*sqrt(2) approx 2.828427
    expected_min = 2 * sqrt(2);
    assert(abs(test_res.min_amplitude_sum - expected_min) < 1e-3, ...
        sprintf('W_test min_amplitude_sum mismatch: expected %.6f, got %.6f', ...
        expected_min, test_res.min_amplitude_sum));

    % (c) J_amp maximum must be 4.0
    expected_max = 4.0;
    assert(abs(test_res.max_amplitude_sum - expected_max) < 1e-3, ...
        sprintf('W_test max_amplitude_sum mismatch: expected %.6f, got %.6f', ...
        expected_max, test_res.max_amplitude_sum));

    fprintf('  * Synthetic W_test validation (J2=const=4, min=2*sqrt(2), max=4): PASSED\n');

    % 5. Analytical gradient vs central finite differences for smooth amplitude sum
    test_psi = linspace(0.35, 1.85, N - 1).';
    test_W = results.W;
    test_ref = 1;
    test_opt = 2:N;
    test_eps = 1e-3;
    [~, g_analytic] = smooth_amp_cost_and_grad(test_psi, test_W, test_ref, test_opt, test_eps, +1);
    g_numerical = zeros(size(test_psi));
    h_step = 1e-6;
    for d = 1:numel(test_psi)
        p_plus = test_psi; p_plus(d) = p_plus(d) + h_step;
        p_minus = test_psi; p_minus(d) = p_minus(d) - h_step;
        f_plus = smooth_amp_cost_and_grad(p_plus, test_W, test_ref, test_opt, test_eps, +1);
        f_minus = smooth_amp_cost_and_grad(p_minus, test_W, test_ref, test_opt, test_eps, +1);
        g_numerical(d) = (f_plus - f_minus) / (2 * h_step);
    end
    grad_rel_err = norm(g_analytic - g_numerical) / max(1e-10, norm(g_analytic));
    assert(grad_rel_err < 1e-4, ...
        sprintf('Smooth amplitude analytical vs numerical gradient error too large: %.2e', grad_rel_err));
    fprintf('  * Analytical gradient vs finite difference check: PASSED (rel err = %.2e)\n', grad_rel_err);

    % 6. Agent ID row/column alignment verification
    assert(numel(results.agent_ids) == N, 'Agent IDs count mismatch');
    assert(numel(unique(results.agent_ids)) == N, 'Agent IDs contain duplicates');
    fprintf('  * Agent ID row/column bijective alignment:     PASSED\n');

    % 7. W vs SVD reconstruction exactness check (reported separately from ID alignment)
    if isfield(results, 'U') && isfield(results, 'V') && isfield(results, 'singular_values')
        W_svd_rec = results.U * diag(results.singular_values) * results.V.';
        svd_rec_err = norm(results.W - W_svd_rec, 'fro') / max(1e-12, norm(results.W, 'fro'));
        assert(svd_rec_err < 1e-10, sprintf('W vs SVD reconstruction mismatch: %.2e', svd_rec_err));
        fprintf('  * W vs SVD reconstruction exactness:             PASSED (err = %.2e)\n', svd_rec_err);
    end

    % 8. ID Permutation Invariance and Duplicate Detection
    perm = [2:N, 1];
    perm_ids = results.agent_ids(perm);
    perm_W = results.W(perm, perm);
    [~, inv_row] = ismember(results.agent_ids, perm_ids);
    [~, inv_col] = ismember(results.agent_ids, perm_ids);
    W_realigned = perm_W(inv_row, inv_col);
    assert(norm(W_realigned - results.W, 'fro') < 1e-14, 'Permuted ID alignment failed!');

    dup_data = struct('N', N, 'agent_ids', [results.agent_ids(1), results.agent_ids(1:N-1)], ...
        'W', results.W, 'U', results.U, 'V', results.V, 'singular_values', results.singular_values);
    caught_dup = false;
    try
        validate_mode_data_integrity(dup_data);
    catch
        caught_dup = true;
    end
    assert(caught_dup, 'Failed to detect duplicate agent IDs in integrity check!');
    fprintf('  * ID permutation invariance & duplicate check:  PASSED\n');

    % 9. Line search failure rejection (no invalid updates accepted into history or state)
    cost_bad = @(x) NaN;
    grad_dummy = @(x) ones(size(x));
    x0_test = [0.1; 0.2];
    [x_ret, ~, flag_ret] = run_lbfgs_optimizer(x0_test, cost_bad, grad_dummy, 10, 1e-6);
    assert(isequal(x_ret, x0_test) && flag_ret < 0, 'L-BFGS accepted NaN step!');

    cost_uphill = @(x) sum(x.^2);
    grad_uphill = @(x) -2 * x; % Negative of true gradient points uphill
    [x_uphill_ret, ~, flag_uphill] = run_lbfgs_optimizer([1.0; 1.0], cost_uphill, grad_uphill, 10, 1e-6);
    assert(isequal(x_uphill_ret, [1.0; 1.0]) && flag_uphill <= 0, 'L-BFGS accepted uphill step on line search failure!');
    fprintf('  * Line search failure rejection check:          PASSED\n');

    % 10. RandomSeed reproducibility check
    rng_test_opts = opts;
    rng_test_opts.NumStarts = 10;
    rng_test_opts.RandomSeed = 12345;
    orig_rng_rep = rng();
    rng(12345);
    res_rep1 = optimize_total_input_amplitude_sum(W_test, 1, [], [], rng_test_opts);
    rng(12345);
    res_rep2 = optimize_total_input_amplitude_sum(W_test, 1, [], [], rng_test_opts);
    rng(orig_rng_rep);
    assert(abs(res_rep1.min_amplitude_sum - res_rep2.min_amplitude_sum) < 1e-14, 'RandomSeed reproducibility failed for min!');
    assert(abs(res_rep1.max_amplitude_sum - res_rep2.max_amplitude_sum) < 1e-14, 'RandomSeed reproducibility failed for max!');
    assert(norm(res_rep1.theta_min - res_rep2.theta_min) < 1e-14, 'RandomSeed reproducibility failed for theta_min!');
    fprintf('  * RandomSeed reproducibility check:              PASSED\n');

    fprintf('  Validation suite successfully passed (100%%).\n\n');
end

% =========================================================================
% 5. FIGURES GENERATION
% =========================================================================

function save_mode_phases_figure(results, output_dir)
    N = results.N;
    agent_ids = results.agent_ids;
    modes = results.modes;

    fig = figure('Color', 'w', 'Position', [100, 100, 1200, 260 * N], 'Visible', 'off');
    t_lay = tiledlayout(fig, N, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_lay, 'SVD Modes Collective Signal X_l: Optimal Phases (Max vs Min) and Amplitudes', ...
        'FontWeight', 'bold', 'FontSize', 12);

    colors = lines(N);
    agent_labels = arrayfun(@(id) sprintf('Agent %d', id), agent_ids, 'UniformOutput', false);

    for l = 1:N
        m = modes(l);

        % Col 1: Phasor plot for MAXIMIZATION
        ax_max = nexttile(t_lay);
        plot_phasor_circle(ax_max, m.theta_max, abs(m.v_l), agent_labels, colors, ...
            sprintf('Mode %d MAX: |X_%d| = %.3f', l, l, m.max_amplitude));

        % Col 2: Phasor plot for MINIMIZATION
        ax_min = nexttile(t_lay);
        plot_phasor_circle(ax_min, m.theta_min, abs(m.v_l), agent_labels, colors, ...
            sprintf('Mode %d MIN: |X_%d| = %.2e', l, l, m.min_amplitude));

        % Col 3: Comparison Bar Chart
        ax_bar = nexttile(t_lay); hold(ax_bar, 'on'); grid(ax_bar, 'on'); box(ax_bar, 'on');
        bar(ax_bar, 1, m.max_amplitude, 0.45, 'FaceColor', [0.85, 0.325, 0.098]);
        bar(ax_bar, 2, m.min_amplitude, 0.45, 'FaceColor', [0.0, 0.447, 0.741]);
        xticks(ax_bar, [1, 2]);
        xticklabels(ax_bar, {'Max Amplitude', 'Min Amplitude'});
        ylabel(ax_bar, '|X| Amplitude');
        title(ax_bar, sprintf('Suppression: %.1f%% (\\sigma_%d = %.3f)', ...
            100 * m.extinction_ratio, l, m.sigma_l), 'FontSize', 9);
        ylim(ax_bar, [0, max(1e-4, m.max_amplitude * 1.15)]);
    end

    saveas(fig, fullfile(output_dir, 'optimal_phases_svd_modes.png'));
    close(fig);
end

function save_total_network_figure(results, output_dir)
    N = results.N;
    agent_ids = results.agent_ids;
    net = results.total_network;
    W = results.W;
    delta = results.delta;

    fig = figure('Color', 'w', 'Position', [100, 100, 1150, 750], 'Visible', 'off');
    t_lay = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_lay, 'Total Network Input Vibration Optimization: Extreme Phase States & Waveforms', ...
        'FontWeight', 'bold', 'FontSize', 12);

    colors = lines(N);
    agent_labels = arrayfun(@(id) sprintf('Agent %d', id), agent_ids, 'UniformOutput', false);

    % 1. Phasor Circle: MAX
    ax1 = nexttile(t_lay, 1);
    plot_phasor_circle(ax1, net.theta_max, ones(N, 1), agent_labels, colors, ...
        sprintf('Network MAX Phase (Total E = %.4f)', net.max_energy));

    % 2. Phasor Circle: MIN
    ax2 = nexttile(t_lay, 2);
    plot_phasor_circle(ax2, net.theta_min, ones(N, 1), agent_labels, colors, ...
        sprintf('Network MIN Phase (Total E = %.2e)', net.min_energy));

    % 3. Agent Amplitudes Comparison Bar Chart
    ax3 = nexttile(t_lay, 3); hold(ax3, 'on'); grid(ax3, 'on'); box(ax3, 'on');
    x_idx = 1:N;
    b_data = [net.agent_amplitudes_at_max(:), net.agent_amplitudes_at_min(:)];
    has_obs = results.observed.available;
    if has_obs
        b_data = [b_data, results.observed.I_i_mean_amplitude(:)];
    end
    b_h = bar(ax3, x_idx, b_data, 'grouped');
    b_h(1).FaceColor = [0.85, 0.325, 0.098];
    b_h(2).FaceColor = [0.0, 0.447, 0.741];
    if has_obs
        b_h(3).FaceColor = [0.466, 0.674, 0.188];
        legend(ax3, {'Max Phase', 'Min Phase', 'Observed RMS'}, 'Location', 'northeast', 'FontSize', 7.5);
    else
        legend(ax3, {'Max Phase', 'Min Phase'}, 'Location', 'northeast', 'FontSize', 7.5);
    end
    xticks(ax3, x_idx); xticklabels(ax3, agent_ids);
    xlabel(ax3, 'Receiver Agent'); ylabel(ax3, 'Input Amplitude |I_i|');
    title(ax3, sprintf('Receiver Amplitudes (Suppression: %.1f%%)', net.suppression_percent), 'FontSize', 9.5);

    % Simulate 1 period of input waveforms I_i(t)
    omega = 2 * pi; % 1 Hz normalized cycle
    t_cycle = linspace(0, 1, 200).';

    % 4. Waveforms at MAX Phase
    ax4 = nexttile(t_lay, 4); hold(ax4, 'on'); grid(ax4, 'on'); box(ax4, 'on');
    b_max_t = sqrt(2) * cos(omega * t_cycle + net.theta_max.' - delta);
    I_max_t = b_max_t * W.';
    for i = 1:N
        plot(ax4, t_cycle, I_max_t(:, i), 'LineWidth', 1.6, 'Color', colors(i, :), ...
            'DisplayName', sprintf('Agent %d', agent_ids(i)));
    end
    xlabel(ax4, 'Normalized Cycle (t/T)'); ylabel(ax4, 'Input Signal I_i(t)');
    title(ax4, 'Maximized Input Waveforms (Coherent Interference)', 'FontSize', 9.5);
    legend(ax4, 'Location', 'eastoutside', 'FontSize', 7);

    % 5. Waveforms at MIN Phase
    ax5 = nexttile(t_lay, 5); hold(ax5, 'on'); grid(ax5, 'on'); box(ax5, 'on');
    b_min_t = sqrt(2) * cos(omega * t_cycle + net.theta_min.' - delta);
    I_min_t = b_min_t * W.';
    for i = 1:N
        plot(ax5, t_cycle, I_min_t(:, i), 'LineWidth', 1.6, 'Color', colors(i, :), ...
            'DisplayName', sprintf('Agent %d', agent_ids(i)));
    end
    xlabel(ax5, 'Normalized Cycle (t/T)'); ylabel(ax5, 'Input Signal I_i(t)');
    title(ax5, 'Minimized Input Waveforms (Destructive Interference)', 'FontSize', 9.5);
    % Match y-limits with max plot for direct visual comparison
    ylim(ax5, ylim(ax4));

    % 6. Network Quadratic Form Matrix Q = W^T * W Heatmap
    ax6 = nexttile(t_lay, 6);
    imagesc(ax6, net.Q); colormap(ax6, jet(256)); colorbar(ax6);
    xticks(ax6, 1:N); yticks(ax6, 1:N);
    xticklabels(ax6, agent_ids); yticklabels(ax6, agent_ids);
    xlabel(ax6, 'Agent j'); ylabel(ax6, 'Agent k');
    title(ax6, 'Network Energy Matrix Q = W^T W', 'FontSize', 9.5);

    saveas(fig, fullfile(output_dir, 'optimal_phases_total_network.png'));
    close(fig);
end

function save_agent_wise_figure(results, output_dir)
    N = results.N;
    agent_ids = results.agent_ids;
    ag = results.agent_wise;

    fig = figure('Color', 'w', 'Position', [100, 100, 1150, 260 * ceil(N/2)], 'Visible', 'off');
    n_rows = ceil(N / 2);
    t_lay = tiledlayout(fig, n_rows, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_lay, 'Agent-wise Targeted Input Vibration: Maximum vs Minimum Attainable Amplitudes', ...
        'FontWeight', 'bold', 'FontSize', 12);

    colors = lines(N);

    for i = 1:N
        ax = nexttile(t_lay); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
        rec = ag(i);
        bar(ax, 1, rec.max_amplitude, 0.45, 'FaceColor', [0.85, 0.325, 0.098]);
        bar(ax, 2, rec.min_amplitude, 0.45, 'FaceColor', [0.0, 0.447, 0.741]);
        xticks(ax, [1, 2]); xticklabels(ax, {'Max Input', 'Min Input'});
        ylabel(ax, 'Amplitude |I_i|');

        % Annotate sender phases
        th_max_deg = rad2deg(rec.theta_max);
        th_min_deg = rad2deg(rec.theta_min);
        str_max = sprintf('Max phases: %s', mat2str(round(th_max_deg.')));
        str_min = sprintf('Min phases: %s', mat2str(round(th_min_deg.')));
        title(ax, sprintf('Receiver %d (Max: %.3f, Min: %.2e)\n%s\n%s', ...
            agent_ids(i), rec.max_amplitude, rec.min_amplitude, str_max, str_min), 'FontSize', 8);
    end

    saveas(fig, fullfile(output_dir, 'optimal_phases_agent_wise.png'));
    close(fig);
end

function save_total_amplitude_sum_figure(results, output_dir)
% Visualizes Best-found minimum and maximum total input amplitude sum J_amp,
% per-agent amplitude distribution, and 4-way comparison against quadratic J2 states.
    N = results.N;
    agent_ids = results.agent_ids;
    amp_res = results.total_input_amplitude_sum;
    net_res = results.total_network;
    W = results.W;

    fig = figure('Color', 'w', 'Position', [100, 100, 1250, 780], 'Visible', 'off');
    t_lay = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_lay, 'Total Receiver Input Amplitude Sum J_{amp} = \Sigma_i \surd2 |(W e^{i\theta})_i|: Best-Found Extreme Phase States', ...
        'FontWeight', 'bold', 'FontSize', 12);

    colors = lines(N);
    agent_labels = arrayfun(@(id) sprintf('Agent %d', id), agent_ids, 'UniformOutput', false);

    % 1. Phasor plot: Best-found Minimum J_amp
    % 1. Phasor plot: Best-found Minimum J_amp
    ax1 = nexttile(t_lay, 1);
    plot_phasor_circle(ax1, amp_res.theta_min, ones(N, 1), ...
        agent_labels, colors, sprintf('Best-Found Min J_{amp} = %.4f (J_2 = %.4f)', ...
        amp_res.min_amplitude_sum, amp_res.J2_at_min));

    % 2. Phasor plot: Best-found Maximum J_amp
    ax2 = nexttile(t_lay, 2);
    plot_phasor_circle(ax2, amp_res.theta_max, ones(N, 1), ...
        agent_labels, colors, sprintf('Best-Found Max J_{amp} = %.4f (J_2 = %.4f)', ...
        amp_res.max_amplitude_sum, amp_res.J2_at_max));

    % 3. Per-receiver Amplitude Comparison Bar Chart
    ax3 = nexttile(t_lay, 3); hold(ax3, 'on'); grid(ax3, 'on'); box(ax3, 'on');
    x_idx = 1:N;
    b_data = [amp_res.agent_amplitudes_at_min(:), amp_res.agent_amplitudes_at_max(:)];
    has_obs = isfield(results, 'observed') && isfield(results.observed, 'available') && results.observed.available;
    if has_obs && isfield(results.observed, 'agent_amps_mean')
        b_data = [b_data, results.observed.agent_amps_mean(:)];
    end
    b_h = bar(ax3, x_idx, b_data, 'grouped');
    b_h(1).FaceColor = [0.0, 0.447, 0.741];  % Blue for Min
    b_h(2).FaceColor = [0.85, 0.325, 0.098]; % Red-orange for Max
    if has_obs && isfield(results.observed, 'agent_amps_mean')
        b_h(3).FaceColor = [0.466, 0.674, 0.188];
        legend(ax3, {'Best-found Min J_{amp}', 'Best-found Max J_{amp}', 'Observed Mean'}, ...
            'Location', 'northwest', 'FontSize', 8);
    else
        legend(ax3, {'Best-found Min J_{amp}', 'Best-found Max J_{amp}'}, ...
            'Location', 'northwest', 'FontSize', 8);
    end
    xticks(ax3, x_idx); xticklabels(ax3, agent_ids);
    xlabel(ax3, 'Receiver Agent'); ylabel(ax3, 'Input Amplitude |I_i|');
    title(ax3, sprintf('Receiver Amplitudes (Min Sum = %.4f, Max Sum = %.4f, Suppr = %.1f%%)', ...
        amp_res.min_amplitude_sum, amp_res.max_amplitude_sum, amp_res.suppression_percent), 'FontSize', 9.5);

    % 4. 4-Way Candidate Comparison: J_amp vs J2 across 4 extreme phase states
    cand_labels = {'Min J_{amp}', 'Max J_{amp}', 'Min J_2', 'Max J_2'};

    % Compute J_amp and J2 for all 4 states
    J_amp_vals = zeros(4, 1);
    J2_vals = zeros(4, 1);

    % State 1: Min J_amp
    J_amp_vals(1) = amp_res.min_amplitude_sum;
    J2_vals(1)    = amp_res.J2_at_min;

    % State 2: Max J_amp
    J_amp_vals(2) = amp_res.max_amplitude_sum;
    J2_vals(2)    = amp_res.J2_at_max;

    % State 3: Min J2
    G_net_min = W * exp(1i * net_res.theta_min);
    J_amp_vals(3) = sum(sqrt(2) * abs(G_net_min));
    J2_vals(3)    = net_res.min_energy;

    % State 4: Max J2
    G_net_max = W * exp(1i * net_res.theta_max);
    J_amp_vals(4) = sum(sqrt(2) * abs(G_net_max));
    J2_vals(4)    = net_res.max_energy;

    ax4 = nexttile(t_lay, 4); hold(ax4, 'on'); grid(ax4, 'on'); box(ax4, 'on');
    yyaxis(ax4, 'left');
    b_amp = bar(ax4, (1:4) - 0.15, J_amp_vals, 0.3, 'FaceColor', [0.2, 0.6, 0.8]);
    ylabel(ax4, 'Amplitude Sum J_{amp} = \Sigma_i |I_i|');
    if has_obs && isfield(results.observed, 'J_amp_mean')
        yline(ax4, results.observed.J_amp_mean, 'b--', 'Observed J_{amp}', 'LineWidth', 1.2);
    end

    yyaxis(ax4, 'right');
    b_j2 = bar(ax4, (1:4) + 0.15, J2_vals, 0.3, 'FaceColor', [0.85, 0.5, 0.2]);
    ylabel(ax4, 'Quadratic Vibration Energy J_2 = ||W z||^2');
    if has_obs && isfield(results.observed, 'J2_mean')
        yline(ax4, results.observed.J2_mean, 'r--', 'Observed J_2', 'LineWidth', 1.2);
    end

    xticks(ax4, 1:4); xticklabels(ax4, cand_labels);
    title(ax4, '4-Way Comparison: J_{amp} and J_2 across Extreme Phase Candidates', 'FontSize', 9.5);
    xlabel(ax4, 'Extreme Phase Candidate State');

    % Explanatory footnote in bottom annotation
    annotation(fig, 'textbox', [0.1, 0.005, 0.8, 0.03], 'String', ...
        'Note: Best-found states from multi-start local search. Global optimality and dynamical stability not assumed.', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontAngle', 'italic');

    saveas(fig, fullfile(output_dir, 'optimal_phases_total_input_amplitude_sum.png'));
    close(fig);
end

function save_observed_amplitude_sum_timeseries_figure(results, output_dir)
% Saves time series of observed instantaneous input amplitude sum J_amp(t)
% compared against best-found minimum, best-found maximum, and time-average.
    if ~isfield(results, 'observed') || ~results.observed.available
        return;
    end

    t = results.observed.time_sec;
    J_amp_t = results.observed.J_amp_t;
    J_amp_mean = results.observed.J_amp_mean;

    amp_res = results.total_input_amplitude_sum;
    min_sum = amp_res.min_amplitude_sum;
    max_sum = amp_res.max_amplitude_sum;

    fig = figure('Color', 'w', 'Position', [150, 150, 950, 480], 'Visible', 'off');
    ax = axes(fig); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

    h_obs = plot(ax, t, J_amp_t, 'Color', [0.15, 0.45, 0.75], 'LineWidth', 1.2, ...
        'DisplayName', 'Observed J_{amp}(t)');
    h_mean = yline(ax, J_amp_mean, '--', ...
        sprintf('Observed Mean (%.4f)', J_amp_mean), ...
        'Color', [0.0, 0.5, 0.2], 'LineWidth', 1.6, ...
        'DisplayName', sprintf('Observed Mean = %.4f', J_amp_mean));
    h_min = yline(ax, min_sum, '-.', ...
        sprintf('Best-Found Min J_{amp} (%.4f)', min_sum), ...
        'Color', [0.2, 0.6, 0.8], 'LineWidth', 1.6, ...
        'DisplayName', sprintf('Best-Found Min J_{amp} = %.4f', min_sum));
    h_max = yline(ax, max_sum, '-.', ...
        sprintf('Best-Found Max J_{amp} (%.4f)', max_sum), ...
        'Color', [0.85, 0.3, 0.1], 'LineWidth', 1.6, ...
        'DisplayName', sprintf('Best-Found Max J_{amp} = %.4f', max_sum));

    xlabel(ax, 'Time [s]', 'FontSize', 10);
    ylabel(ax, 'Total Input Amplitude Sum J_{amp} = \Sigma_i \surd2 |(W e^{i\theta})_i|', 'FontSize', 10);
    title(ax, 'Observed Input Vibration Amplitude Sum J_{amp}(t) vs Best-Found Extremes', 'FontSize', 11, 'FontWeight', 'bold');
    legend(ax, [h_obs, h_mean, h_min, h_max], 'Location', 'best', 'FontSize', 9);

    annotation(fig, 'textbox', [0.08, 0.01, 0.84, 0.04], 'String', ...
        '[Note: All metrics use consistent \surd2 amplitude convention. Best-found states are static extreme candidates; dynamic convergence or stability under control laws is not implied or proven.]', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.35, 0.35, 0.35]);

    fig_path = fullfile(output_dir, 'optimal_phases_observed_amplitude_sum_timeseries.png');
    exportgraphics(fig, fig_path, 'Resolution', 300);
    close(fig);
    fprintf('  Saved Figure: %s\n', fig_path);
end

function plot_phasor_circle(ax, theta_vec, weights, labels, colors, title_str)
    hold(ax, 'on'); axis(ax, 'equal'); box(ax, 'on'); grid(ax, 'on');
    N = numel(theta_vec);

    % Draw unit circle
    th_c = linspace(0, 2*pi, 200);
    plot(ax, cos(th_c), sin(th_c), 'k--', 'LineWidth', 0.8, 'Color', [0.6, 0.6, 0.6]);
    plot(ax, [-1.2, 1.2], [0, 0], 'k:', 'Color', [0.7, 0.7, 0.7]);
    plot(ax, [0, 0], [-1.2, 1.2], 'k:', 'Color', [0.7, 0.7, 0.7]);

    max_w = max(weights);
    if max_w == 0, max_w = 1.0; end
    scaled_w = 0.3 + 0.7 * (weights / max_w);

    for j = 1:N
        r = scaled_w(j);
        th = theta_vec(j);
        xj = r * cos(th);
        yj = r * sin(th);

        % Arrow from origin
        quiver(ax, 0, 0, xj, yj, 0, 'LineWidth', 1.8, 'Color', colors(j, :), ...
            'MaxHeadSize', 0.4);
        % Marker and text at tip
        plot(ax, xj, yj, 'o', 'MarkerSize', 6, 'MarkerFaceColor', colors(j, :), ...
            'MarkerEdgeColor', 'k');

        offset = 1.15;
        text(ax, offset * cos(th), offset * sin(th), labels{j}, ...
            'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
            'Color', colors(j, :) * 0.8);
    end

    xlim(ax, [-1.4, 1.4]); ylim(ax, [-1.4, 1.4]);
    title(ax, title_str, 'FontSize', 8.5);
    set(ax, 'XTick', [-1, 0, 1], 'YTick', [-1, 0, 1]);
end

% =========================================================================
% 6. CSV EXPORTS
% =========================================================================

function export_results_csv(results, output_dir)
    N = results.N;
    agent_ids = results.agent_ids;

    % 1. Modes Summary CSV
    mode_num = (1:N).';
    sigma_vec = results.singular_values(:);
    max_amp_vec = arrayfun(@(m) m.max_amplitude, results.modes);
    min_amp_vec = arrayfun(@(m) m.min_amplitude, results.modes);
    ratio_vec = arrayfun(@(m) m.amplitude_ratio, results.modes);
    extinction_pct = 100 * arrayfun(@(m) m.extinction_ratio, results.modes);

    t_modes = table(mode_num, sigma_vec, max_amp_vec(:), min_amp_vec(:), ratio_vec(:), extinction_pct(:), ...
        'VariableNames', {'mode_l', 'singular_value_sigma', 'max_amplitude', 'min_amplitude', ...
        'amplitude_ratio', 'suppression_percent'});

    for j = 1:N
        col_max_deg = sprintf('max_phase_deg_agent_%d', agent_ids(j));
        col_min_deg = sprintf('min_phase_deg_agent_%d', agent_ids(j));
        vals_max = arrayfun(@(m) rad2deg(m.theta_max(j)), results.modes);
        vals_min = arrayfun(@(m) rad2deg(m.theta_min(j)), results.modes);
        t_modes.(col_max_deg) = vals_max(:);
        t_modes.(col_min_deg) = vals_min(:);
    end
    writetable(t_modes, fullfile(output_dir, 'optimal_phases_modes_summary.csv'));

    % 2. Total Network Optimal Phases CSV
    net = results.total_network;
    state_type = {'MAX_VIBRATION'; 'MIN_VIBRATION'};
    total_energy = [net.max_energy; net.min_energy];
    mean_amplitude = [net.mean_amp_max; net.mean_amp_min];
    t_net = table(state_type, total_energy, mean_amplitude, ...
        'VariableNames', {'state_type', 'total_energy', 'mean_amplitude'});
    for j = 1:N
        col_rad = sprintf('phase_rad_agent_%d', agent_ids(j));
        col_deg = sprintf('phase_deg_agent_%d', agent_ids(j));
        t_net.(col_rad) = [net.theta_max(j); net.theta_min(j)];
        t_net.(col_deg) = [rad2deg(net.theta_max(j)); rad2deg(net.theta_min(j))];
    end
    writetable(t_net, fullfile(output_dir, 'optimal_phases_total_network.csv'));

    % 3. Agent-wise Targeted Input Optimal Phases CSV
    target_agent_id = agent_ids(:);
    max_input_amp = arrayfun(@(a) a.max_amplitude, results.agent_wise);
    min_input_amp = arrayfun(@(a) a.min_amplitude, results.agent_wise);
    t_agent = table(target_agent_id, max_input_amp(:), min_input_amp(:), ...
        'VariableNames', {'target_agent_id', 'max_input_amp', 'min_input_amp'});
    for j = 1:N
        col_max_deg = sprintf('sender_%d_phase_for_max_deg', agent_ids(j));
        col_min_deg = sprintf('sender_%d_phase_for_min_deg', agent_ids(j));
        vals_max_ag = arrayfun(@(a) rad2deg(a.theta_max(j)), results.agent_wise);
        vals_min_ag = arrayfun(@(a) rad2deg(a.theta_min(j)), results.agent_wise);
        t_agent.(col_max_deg) = vals_max_ag(:);
        t_agent.(col_min_deg) = vals_min_ag(:);
    end
    writetable(t_agent, fullfile(output_dir, 'optimal_phases_agent_wise.csv'));

    % 4. Total Input Amplitude Sum Optimal Phases CSV (Single consolidated summary CSV)
    amp_res = results.total_input_amplitude_sum;
    state_type_amp = {'BEST_FOUND_MIN_AMP_SUM'; 'BEST_FOUND_MAX_AMP_SUM'};
    amp_sum_vals = [amp_res.min_amplitude_sum; amp_res.max_amplitude_sum];
    j2_vals = [amp_res.J2_at_min; amp_res.J2_at_max];
    mean_amp_vals = [mean(amp_res.agent_amplitudes_at_min); mean(amp_res.agent_amplitudes_at_max)];
    t_amp = table(state_type_amp, amp_sum_vals, j2_vals, mean_amp_vals, ...
        'VariableNames', {'state_type', 'total_amplitude_sum_J_amp', 'energy_J2', 'mean_agent_amplitude'});
    for j = 1:N
        col_rad = sprintf('phase_rad_agent_%d', agent_ids(j));
        col_deg = sprintf('phase_deg_agent_%d', agent_ids(j));
        col_amp = sprintf('input_amp_agent_%d', agent_ids(j));
        t_amp.(col_rad) = [amp_res.theta_min(j); amp_res.theta_max(j)];
        t_amp.(col_deg) = [rad2deg(amp_res.theta_min(j)); rad2deg(amp_res.theta_max(j))];
        t_amp.(col_amp) = [amp_res.agent_amplitudes_at_min(j); amp_res.agent_amplitudes_at_max(j)];
    end
    writetable(t_amp, fullfile(output_dir, 'optimal_phases_total_input_amplitude_sum.csv'));
end

% =========================================================================
% 7. CONSOLE REPORTING & HELPERS
% =========================================================================

function print_summary_report(results)
    N = results.N;
    agent_ids = results.agent_ids;
    ref_id = results.ref_agent_id;

    fprintf('\n=========================================================================\n');
    fprintf('  FINAL SUMMARY: INPUT VIBRATION OPTIMAL PHASE RELATIONSHIPS\n');
    fprintf('=========================================================================\n');
    fprintf('Reference Gauge Agent: ID %d (Phase fixed to 0 deg)\n\n', ref_id);

    fprintf('--- 1. SVD MODES COLLECTIVE SENDER SIGNALS X_l ---\n');
    for l = 1:N
        m = results.modes(l);
        th_max_deg = round(rad2deg(m.theta_max));
        th_min_deg = round(rad2deg(m.theta_min));
        fprintf('  Mode %d (sigma = %.4f):\n', l, m.sigma_l);
        fprintf('    * MAX Amp = %7.4f | Phases: %s deg\n', m.max_amplitude, mat2str(th_max_deg.'));
        fprintf('    * MIN Amp = %7.2e | Phases: %s deg\n', m.min_amplitude, mat2str(th_min_deg.'));
        fprintf('    * Suppression Efficiency: %.2f%%\n', 100 * m.extinction_ratio);
    end

    fprintf('\n--- 2. TOTAL NETWORK INPUT VIBRATION ENERGY ||W*z||^2 ---\n');
    net = results.total_network;
    th_net_max_deg = round(rad2deg(net.theta_max));
    th_net_min_deg = round(rad2deg(net.theta_min));
    fprintf('  MAX VIBRATION STATE: Energy = %.5f (Mean Amp = %.4f)\n', net.max_energy, net.mean_amp_max);
    fprintf('    * Phase Relationship : %s deg\n', mat2str(th_net_max_deg.'));
    fprintf('  MIN VIBRATION STATE: Energy = %.5e (Mean Amp = %.4e)\n', net.min_energy, net.mean_amp_min);
    fprintf('    * Phase Relationship : %s deg\n', mat2str(th_net_min_deg.'));
    fprintf('  Total Network Vibration Suppression: %.2f%%\n', net.suppression_percent);

    fprintf('\n--- 3. TOTAL INPUT AMPLITUDE SUM J_amp = sum_i sqrt(2)*|G_i| ---\n');
    amp_res = results.total_input_amplitude_sum;
    th_amp_max_deg = round(rad2deg(amp_res.theta_max));
    th_amp_min_deg = round(rad2deg(amp_res.theta_min));
    fprintf('  BEST-FOUND MAX AMPLITUDE SUM: J_amp = %.5f (J2 = %.5f, Mean Amp = %.4f)\n', ...
        amp_res.max_amplitude_sum, amp_res.J2_at_max, mean(amp_res.agent_amplitudes_at_max));
    fprintf('    * Phase Relationship : %s deg\n', mat2str(th_amp_max_deg.'));
    fprintf('  BEST-FOUND MIN AMPLITUDE SUM: J_amp = %.5f (J2 = %.5f, Mean Amp = %.4f)\n', ...
        amp_res.min_amplitude_sum, amp_res.J2_at_min, mean(amp_res.agent_amplitudes_at_min));
    fprintf('    * Phase Relationship : %s deg\n', mat2str(th_amp_min_deg.'));
    fprintf('  Amplitude Sum Suppression: %.2f%%\n', amp_res.suppression_percent);
    fprintf('  [Note: Best-found states from multi-start local search; global optimality and dynamic stability are not assumed.]\n');

    if results.observed.available
        fprintf('\n--- 4. COMPARISON WITH OBSERVED EXPERIMENTAL PHASES ---\n');
        if isfield(results.observed, 'J_amp_mean')
            ratio_amp = 100 * (results.observed.J_amp_mean / max(1e-12, amp_res.max_amplitude_sum));
            fprintf('  Observed Mean Input Amplitude Sum J_amp : %.5f\n', results.observed.J_amp_mean);
            fprintf('  Relative Amplitude Level                : %.2f%% of Best-found Maximum\n', ratio_amp);
        end
        if isfield(results.observed, 'J2_mean')
            ratio_j2 = 100 * (results.observed.J2_mean / max(1e-12, net.max_energy));
            fprintf('  Observed Mean Vibration Energy J2       : %.5f\n', results.observed.J2_mean);
            fprintf('  Relative J2 Energy Level                : %.2f%% of Best-found Maximum\n', ratio_j2);
        end
        if isfield(results.observed, 'total_energy_mean')
            fprintf('  Observed Direct Signal Mean Energy      : %.5f (from time-averaged <I_i(t)^2>)\n', ...
                results.observed.total_energy_mean);
        end
    elseif isfield(results.observed, 'skip_reason')
        fprintf('\n--- 4. COMPARISON WITH OBSERVED EXPERIMENTAL PHASES (SKIPPED) ---\n');
        fprintf('  Reason: %s\n', results.observed.skip_reason);
    end
    fprintf('=========================================================================\n\n');
end

function th_wrapped = wrap_to_pi(th)
    th_wrapped = mod(th + pi, 2 * pi) - pi;
end

function opts = parse_options(varargin)
    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'NumStarts', 50, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(parser, 'SaveOutputs', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'SaveMat', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'OutputDir', '', @(x) ischar(x) || isstring(x));
    addParameter(parser, 'ReferenceAgent', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(parser, 'RunSyntheticValidation', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'RandomSeed', 0, @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    parse(parser, varargin{:});
    opts = parser.Results;
end
