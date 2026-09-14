function [results, res_max, res_min] = simulate_cw_optimal_phase_relationships(source, varargin)
% SIMULATE_CW_OPTIMAL_PHASE_RELATIONSHIPS
% Calculates and simulates the phase relationships that maximize and minimize
% the network phase-coherence objective function:
%
%   C_W(theta) = 0.5 * Re( x' * W * x )
%              = 0.5 * sum_{j,k} W_{sym, jk} * cos(theta_j - theta_k)
%
% where:
%   x_j = exp(1i * theta_j) is the complex phasor of agent j,
%   W in R^{N x N} is the signed directed coupling weight matrix (W_{jj} = 0),
%   W_sym = 0.5 * (W + W') is the symmetric interaction component.
%
% This function performs:
%   1. Multi-start global optimization (using analytical gradients & Hessians)
%      to find the exact global maximum and minimum phase relationships.
%   2. Gradient flow dynamic simulation (d(theta)/dt = +/- grad C_W) to simulate
%      how arbitrary phase states dynamically relax into the maximum and minimum
%      coherence configurations over time.
%   3. Comprehensive visualization:
%      - Dynamic simulation trajectories of relative phases and C_W(t).
%      - Phasor diagrams on the complex unit circle with network coupling edges.
%      - Pairwise energy contribution matrices and spectral eigenvalue bounds.
%
% Examples:
%   % Default run using SStickFlat network_coupling_matrix_W.csv
%   res = simulate_cw_optimal_phase_relationships();
%
%   % Run on SStick dataset
%   res = simulate_cw_optimal_phase_relationships('SStick');
%
%   % Run with custom start count and save outputs
%   res = simulate_cw_optimal_phase_relationships([], 'NumStarts', 500, 'SaveOutputs', true);

    % =========================================================================
    % USER CONFIGURATION PRESETS (For F5 / Run without arguments)
    % =========================================================================
    % Default weight matrix CSV location
    DEFAULT_W_CSV = fullfile('EstimateL', 'SStickFlat', 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');

    % Optimization settings
    DEFAULT_NUM_STARTS    = 300;   % Number of random multi-start initializations
    DEFAULT_RANDOM_SEED   = 0;     % Random seed for reproducibility (empty [] for shuffle)

    % Gradient flow simulation settings
    DEFAULT_SIM_TIME_SEC  = 30.0;  % Relaxation simulation duration (s)
    DEFAULT_SIM_DT        = 0.01;  % Simulation time step dt (s)
    DEFAULT_GRAD_GAIN     = 10.0;  % Gradient ascent/descent rate (gain)

    % Reference agent for relative phase calculation (empty [] uses 1st agent)
    DEFAULT_REF_AGENT_ID  = [];

    % Visualization & Export settings
    DEFAULT_PLOT_TRAJECTORIES = true;  % Plot relaxation dynamics of theta(t) and C_W(t)
    DEFAULT_PLOT_PHASORS      = true;  % Plot optimal phases on complex unit circle
    DEFAULT_PLOT_ENERGY_MATRIX= true;  % Plot pairwise energy contribution matrices
    DEFAULT_AGENT_OFFSET      = 0;     % Display ID offset (e.g. -7 to show 8->1, 9->2)
    DEFAULT_SAVE_OUTPUTS      = false; % Save CSV and PNG outputs
    DEFAULT_OUTPUT_DIR        = '';    % Custom output folder (empty [] uses default)
    % =========================================================================

    % Flexible argument dispatch:
    % Allow calls like:
    %   simulate_cw_optimal_phase_relationships()
    %   simulate_cw_optimal_phase_relationships('SStickFlat')
    %   simulate_cw_optimal_phase_relationships('SaveOutputs', false)
    %   simulate_cw_optimal_phase_relationships('save_plots', false)
    %   simulate_cw_optimal_phase_relationships('path/to/W.csv', 'PlotPhasors', true)
    raw_args = varargin;
    if nargin < 1 || isempty(source)
        source = DEFAULT_W_CSV;
    elseif ischar(source) || isstring(source)
        source_str = char(source);
        % If source does not exist as file/folder and does not end with .csv, treat as option name
        if ~isfile(source_str) && ~isfolder(source_str) && ~endsWith(lower(source_str), '.csv')
            % Check if it's a known dataset name (e.g. SStick, SStickFlat)
            s_cand1 = fullfile('EstimateL', source_str, 'low_rank_analysis', 'M10', ...
                'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
            s_cand2 = fullfile(source_str, 'low_rank_analysis', 'M10', ...
                'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
            if isfile(s_cand1) || isfile(s_cand2)
                % Valid dataset name
            else
                % Option name passed as 1st argument
                raw_args = [{source}, raw_args];
                source = DEFAULT_W_CSV;
            end
        end
    end

    opts = parse_options(DEFAULT_NUM_STARTS, DEFAULT_RANDOM_SEED, DEFAULT_SIM_TIME_SEC, ...
        DEFAULT_SIM_DT, DEFAULT_GRAD_GAIN, DEFAULT_REF_AGENT_ID, DEFAULT_PLOT_TRAJECTORIES, ...
        DEFAULT_PLOT_PHASORS, DEFAULT_PLOT_ENERGY_MATRIX, DEFAULT_AGENT_OFFSET, ...
        DEFAULT_SAVE_OUTPUTS, DEFAULT_OUTPUT_DIR, raw_args{:});

    if ~isempty(opts.RandomSeed)
        rng(opts.RandomSeed, 'twister');
    end

    % 1. Load coupling matrix W and agent IDs
    [W, agent_ids, w_source_path] = load_coupling_matrix(source);
    N = numel(agent_ids);
    if N < 2
        error('Weight matrix W must have at least 2 agents (got %d).', N);
    end

    % Symmetric component of W
    W_sym = 0.5 * (W + W.');

    % Spectral relaxation bounds: lambda_min * N/2 <= C_W <= lambda_max * N/2
    [V_eig, D_eig] = eig(W_sym);
    eigvals = diag(D_eig);
    [eigvals_sorted, sort_idx] = sort(eigvals, 'ascend');
    V_eig_sorted = V_eig(:, sort_idx);

    lambda_min = eigvals_sorted(1);
    lambda_max = eigvals_sorted(end);
    cw_spectral_upper_bound = 0.5 * N * lambda_max;
    cw_spectral_lower_bound = 0.5 * N * lambda_min;

    % Setup reference agent
    ref_agent_id = opts.ReferenceAgentId;
    if isempty(ref_agent_id)
        ref_agent_id = agent_ids(1);
    end
    ref_idx = find(agent_ids == ref_agent_id, 1, 'first');
    if isempty(ref_idx)
        error('Reference agent %d is not present in agent_ids: %s', ref_agent_id, mat2str(agent_ids));
    end

    fprintf('=========================================================================\n');
    fprintf('   NETWORK PHASE-COHERENCE OBJECTIVE C_W(theta) OPTIMIZATION & SIMULATION\n');
    fprintf('=========================================================================\n');
    fprintf('  Source W: %s\n', w_source_path);
    fprintf('  Agents (N=%d): %s | Reference Agent: ID %d\n', N, mat2str(agent_ids), ref_agent_id);
    fprintf('  W_sym Eigenvalues: min = %.4f, max = %.4f\n', lambda_min, lambda_max);
    fprintf('  Spectral Unconstrained Bounds: [%.4f, %.4f]\n', cw_spectral_lower_bound, cw_spectral_upper_bound);
    fprintf('  Multi-starts: %d | Sim duration: %.1fs | Sim dt: %.3fs\n\n', ...
        opts.NumStarts, opts.SimTimeSec, opts.SimDt);

    % 2. Multi-start Global Optimization for Maximum and Minimum
    fprintf('[INFO] Searching for Global Maximum and Minimum of C_W(theta)...\n');
    opt_results = run_global_optimization(W, W_sym, opts.NumStarts, V_eig_sorted, ref_idx);

    % 3. Gradient Flow Dynamic Simulation from Random Initial State
    fprintf('[INFO] Simulating gradient flow dynamics (maximization & minimization)...\n');
    sim_results = run_gradient_flow_simulations(W_sym, opt_results.theta_init_sim, ...
        opts.SimTimeSec, opts.SimDt, opts.GradGain, ref_idx);

    % 4. Assemble results struct
    results = struct();
    results.source_path = w_source_path;
    results.agent_ids = agent_ids;
    results.N = N;
    results.reference_agent_id = ref_agent_id;
    results.reference_idx = ref_idx;
    results.W = W;
    results.W_sym = W_sym;
    results.eigenvalues = eigvals_sorted;
    results.spectral_bounds = [cw_spectral_lower_bound, cw_spectral_upper_bound];
    results.maximum = opt_results.maximum;
    results.minimum = opt_results.minimum;
    results.extrema = opt_results.extrema;
    results.all_starts = opt_results.all_starts;
    results.simulation = sim_results;
    results.options = opts;

    % Support secondary output arguments
    res_max = results.maximum;
    res_min = results.minimum;

    % 5. Print comprehensive numerical summary
    print_results_summary(results, opts.AgentOffset);

    % 6. Plotting
    figures = struct();
    if opts.PlotTrajectories
        figures.trajectories = plot_gradient_flow_trajectories(sim_results, agent_ids, ...
            ref_agent_id, opts.AgentOffset, opt_results.maximum.cw_value, opt_results.minimum.cw_value);
    else
        figures.trajectories = [];
    end

    if opts.PlotPhasors
        figures.phasors = plot_optimal_phasor_diagrams(results, opts.AgentOffset);
    else
        figures.phasors = [];
    end

    if opts.PlotEnergyMatrix
        figures.energy_matrix = plot_pairwise_energy_contributions(results, opts.AgentOffset);
    else
        figures.energy_matrix = [];
    end
    results.figures = figures;

    % 7. Save outputs if requested
    if opts.SaveOutputs
        results.export = save_analysis_outputs(results, opts);
    end

    fprintf('[INFO] Analysis completed successfully.\n');
end

% =========================================================================
% MATHEMATICAL FORMULATION: OBJECTIVE, GRADIENT, HESSIAN
% =========================================================================

function [cw, grad, H] = evaluate_cw(theta, W_sym)
% EVALUATE_CW Computes C_W(theta), its analytical gradient, and Hessian.
%
%   C_W(theta) = 0.5 * sum_{j,k} W_{sym, jk} * cos(theta_j - theta_k)
%   grad_j = sum_k W_{sym, jk} * sin(theta_k - theta_j)
%   H_{jk} = W_{sym, jk} * cos(theta_j - theta_k)  (j ~= k)
%   H_{jj} = -sum_{k ~= j} W_{sym, jk} * cos(theta_j - theta_k)

    theta = theta(:);
    N = numel(theta);
    x = exp(1i * theta);

    % Objective value
    % 0.5 * Re(x' * W_sym * x)
    Wx = W_sym * x;
    cw = 0.5 * real(x' * Wx);

    if nargout >= 2
        % Analytical gradient
        % grad_j = Im( conj(x_j) * (W_sym * x)_j )
        grad = imag(conj(x) .* Wx);
    end

    if nargout >= 3
        % Analytical Hessian
        delta_mat = theta - theta.';
        H = W_sym .* cos(delta_mat);
        H(1:N+1:end) = -sum(H, 2);
    end
end

% =========================================================================
% GLOBAL OPTIMIZATION (MULTI-START SEARCH)
% =========================================================================

function opt_res = run_global_optimization(W, W_sym, num_starts, V_eig_sorted, ref_idx)
    N = size(W, 1);

    % Solver options for MATLAB optimization
    use_optim_toolbox = (exist('fminunc', 'file') == 2);

    best_max_cw = -Inf;
    best_max_theta = [];
    best_min_cw = Inf;
    best_min_theta = [];

    start_cw_max = zeros(num_starts, 1);
    start_cw_min = zeros(num_starts, 1);

    % Generate diverse initial starts:
    % 1. Spectral eigenvector angles (max and min eigenvectors)
    % 2. Discrete in-phase / anti-phase combinations
    % 3. Quadrature / traveling wave configurations
    % 4. Random uniform phases
    initial_candidates = cell(num_starts, 1);
    initial_candidates{1} = angle(V_eig_sorted(:, end));    % Spectral max candidate
    initial_candidates{2} = angle(V_eig_sorted(:, 1));      % Spectral min candidate
    initial_candidates{3} = zeros(N, 1);                    % All in-phase
    initial_candidates{4} = pi * (mod(1:N, 2).');           % Alternating in-phase/anti-phase
    initial_candidates{5} = (2 * pi / N) * (0:N-1).';       % Splay / traveling wave state
    initial_candidates{6} = (pi / 2) * (mod(0:N-1, 4).');   % Quadrature state

    for k = 7:num_starts
        initial_candidates{k} = 2 * pi * rand(N, 1) - pi;
    end

    % Random state chosen for the gradient flow dynamic demonstration
    theta_init_sim = initial_candidates{min(7, num_starts)};

    candidates_extrema = cell(2 * num_starts, 1);
    cand_count = 0;

    for s = 1:num_starts
        th0 = initial_candidates{s};

        if use_optim_toolbox
            % Quasi-Newton maximization (minimize -C_W)
            opts_opt = optimoptions('fminunc', 'Display', 'off', ...
                'Algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true, ...
                'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12, 'MaxIterations', 500);

            % Maximization
            obj_fun_max = @(th) objective_with_gradient(th, W_sym, -1);
            [th_max_opt, fval_max] = fminunc(obj_fun_max, th0, opts_opt);
            cw_max_cand = -fval_max;

            % Minimization
            obj_fun_min = @(th) objective_with_gradient(th, W_sym, +1);
            [th_min_opt, fval_min] = fminunc(obj_fun_min, th0, opts_opt);
            cw_min_cand = fval_min;
        else
            % Standalone robust gradient ascent / descent with backtracking line search
            [th_max_opt, cw_max_cand] = optimize_gradient_flow_local(th0, W_sym, +1);
            [th_min_opt, cw_min_cand] = optimize_gradient_flow_local(th0, W_sym, -1);
        end

        start_cw_max(s) = cw_max_cand;
        start_cw_min(s) = cw_min_cand;

        cand_count = cand_count + 1;
        candidates_extrema{cand_count} = th_max_opt;
        cand_count = cand_count + 1;
        candidates_extrema{cand_count} = th_min_opt;

        if cw_max_cand > best_max_cw
            best_max_cw = cw_max_cand;
            best_max_theta = th_max_opt;
        end

        if cw_min_cand < best_min_cw
            best_min_cw = cw_min_cand;
            best_min_theta = th_min_opt;
        end
    end

    % Cluster and classify all unique stationary extrema & equilibria
    extrema = cluster_and_classify_extrema(candidates_extrema(1:cand_count), W_sym, ref_idx);

    % Normalize phase representation relative to reference agent
    max_theta_norm = normalize_phase_vector(best_max_theta, ref_idx);
    min_theta_norm = normalize_phase_vector(best_min_theta, ref_idx);

    opt_res = struct();
    opt_res.theta_init_sim = theta_init_sim;
    opt_res.extrema = extrema;

    % Maximum result
    opt_res.maximum = assemble_state_struct(max_theta_norm, best_max_cw, W_sym, 'Global Maximum (Max Coherence)', ref_idx);

    % Minimum result
    opt_res.minimum = assemble_state_struct(min_theta_norm, best_min_cw, W_sym, 'Global Minimum (Min Coherence)', ref_idx);

    % All starts summary
    opt_res.all_starts = struct( ...
        'max_values', start_cw_max, ...
        'min_values', start_cw_min, ...
        'num_starts', num_starts ...
    );
end

function [f, g] = objective_with_gradient(th, W_sym, sign_factor)
    [cw, grad] = evaluate_cw(th, W_sym);
    f = sign_factor * cw;
    g = sign_factor * grad;
end

function [th_opt, cw_opt] = optimize_gradient_flow_local(th0, W_sym, direction_sign)
    % direction_sign: +1 for ascent (max), -1 for descent (min)
    th = th0;
    lr = 0.2;
    for iter = 1:600
        [cw, grad] = evaluate_cw(th, W_sym);
        step = direction_sign * grad;
        if norm(step) < 1e-9
            break;
        end
        % Armijo backtracking
        for bt = 1:20
            th_new = th + lr * step;
            cw_new = evaluate_cw(th_new, W_sym);
            if direction_sign * (cw_new - cw) >= 0.1 * lr * (grad' * step)
                th = th_new;
                break;
            end
            lr = lr * 0.5;
        end
        lr = min(lr * 1.1, 1.0);
    end
    th_opt = th;
    cw_opt = evaluate_cw(th_opt, W_sym);
end

function state = assemble_state_struct(theta_norm, cw_val, W_sym, state_name, ~)
    x = exp(1i * theta_norm);

    % Pairwise phase differences: delta_theta(j, k) = theta_j - theta_k
    delta_theta_mat = wrap_to_pi(theta_norm - theta_norm.');

    % Pairwise interaction energy matrix: E_{jk} = W_{sym, jk} * cos(theta_j - theta_k)
    energy_matrix = W_sym .* cos(delta_theta_mat);

    % Net interaction force / torque on each agent
    grad = imag(conj(x) .* (W_sym * x));

    state = struct();
    state.name = state_name;
    state.cw_value = cw_val;
    state.theta_rad = theta_norm;
    state.theta_deg = rad2deg(theta_norm);
    state.phasor_x = x;
    state.relative_phase_to_ref_deg = rad2deg(theta_norm);
    state.delta_theta_mat_deg = rad2deg(delta_theta_mat);
    state.energy_matrix = energy_matrix;
    state.gradient_norm = norm(grad);
end

function extrema = cluster_and_classify_extrema(candidates, W_sym, ref_idx)
    N = size(W_sym, 1);
    tol_phase = 0.08; % rad (~4.6 deg)
    tol_cw = 1e-4;

    raw_extrema = struct('theta_norm', {}, 'theta_deg', {}, 'theta_rad', {}, ...
        'cw', {}, 'grad_norm', {}, 'type', {}, 'type_code', {}, ...
        'hessian_eigenvalues', {}, 'order_param', {}, 'count', {}, ...
        'is_global_max', {}, 'is_global_min', {});

    % Subspace orthogonal to ones(N, 1) for gauge-invariant stability analysis
    Q = null(ones(1, N)); % N x (N-1)

    for c = 1:numel(candidates)
        th = candidates{c};
        if isempty(th), continue; end
        th_norm = normalize_phase_vector(th, ref_idx);
        [cw, grad, H] = evaluate_cw(th_norm, W_sym);
        grad_norm = norm(grad);
        if grad_norm > 5e-3
            % Candidate not sufficiently converged to a critical point
            continue;
        end

        % Check if this critical point matches an already recorded one
        match_idx = 0;
        for e = 1:numel(raw_extrema)
            diff_th = wrap_to_pi(th_norm - raw_extrema(e).theta_norm);
            d_phase = norm(diff_th);
            d_cw = abs(cw - raw_extrema(e).cw);
            if d_phase < tol_phase || (d_phase < 0.20 && d_cw < tol_cw)
                match_idx = e;
                break;
            end
        end

        if match_idx > 0
            raw_extrema(match_idx).count = raw_extrema(match_idx).count + 1;
        else
            % Project Hessian onto (N-1) relative-phase subspace (gauge invariant)
            H_proj = Q' * H * Q;
            lambda_proj = sort(real(eig(H_proj)), 'ascend');

            % For C_W maximization, Hessian is negative-definite at maximum.
            % In the gradient flow phase dynamics dot(theta) = + grad C_W:
            % Jacobian is H, so negative eigenvalues -> stable attractor!
            tol_zero = 1e-4;
            n_pos = sum(lambda_proj > tol_zero);
            n_neg = sum(lambda_proj < -tol_zero);

            if n_pos == 0
                type_str = 'Stable Maximum (Attractor)';
                type_code = 'STABLE_MAX';
            elseif n_neg == 0
                type_str = 'Unstable Minimum (Repeller)';
                type_code = 'UNSTABLE_MIN';
            else
                type_str = sprintf('Saddle Point (Index %d)', n_pos);
                type_code = sprintf('SADDLE_%d', n_pos);
            end

            x = exp(1i * th_norm);
            order_p = abs(mean(x));

            entry = struct();
            entry.theta_norm = th_norm;
            entry.theta_deg = rad2deg(th_norm);
            entry.theta_rad = th_norm;
            entry.cw = cw;
            entry.grad_norm = grad_norm;
            entry.type = type_str;
            entry.type_code = type_code;
            entry.hessian_eigenvalues = lambda_proj;
            entry.order_param = order_p;
            entry.count = 1;
            entry.is_global_max = false;
            entry.is_global_min = false;

            raw_extrema(end+1) = entry; %#ok<AGROW>
        end
    end

    if ~isempty(raw_extrema)
        % Sort by C_W descending
        [~, sort_idx] = sort([raw_extrema.cw], 'descend');
        extrema = raw_extrema(sort_idx);
        extrema(1).is_global_max = true;
        extrema(end).is_global_min = true;
    else
        extrema = raw_extrema;
    end
end

% =========================================================================
% GRADIENT FLOW TIME-SERIES DYNAMIC SIMULATION
% =========================================================================

function sim_res = run_gradient_flow_simulations(W_sym, theta0, duration_sec, dt, grad_gain, ref_idx)
% Simulates the continuous relaxation equations:
%   Maximization: d(theta)/dt = + grad_gain * grad C_W(theta)
%   Minimization: d(theta)/dt = - grad_gain * grad C_W(theta)

    time = (0:dt:duration_sec).';
    n_steps = numel(time);
    N = numel(theta0);

    % 1. Maximization trajectory
    theta_max_traj = zeros(n_steps, N);
    cw_max_traj = zeros(n_steps, 1);
    theta_curr = theta0(:);

    for t = 1:n_steps
        theta_max_traj(t, :) = theta_curr.';
        [cw, grad] = evaluate_cw(theta_curr, W_sym);
        cw_max_traj(t) = cw;

        if t < n_steps
            % Euler step with gradient ascent
            theta_curr = theta_curr + dt * (grad_gain * grad);
        end
    end

    % 2. Minimization trajectory
    theta_min_traj = zeros(n_steps, N);
    cw_min_traj = zeros(n_steps, 1);
    theta_curr = theta0(:);

    for t = 1:n_steps
        theta_min_traj(t, :) = theta_curr.';
        [cw, grad] = evaluate_cw(theta_curr, W_sym);
        cw_min_traj(t) = cw;

        if t < n_steps
            % Euler step with gradient descent
            theta_curr = theta_curr - dt * (grad_gain * grad);
        end
    end

    % Compute relative phases with respect to reference agent
    rel_theta_max = wrap_to_pi(theta_max_traj - theta_max_traj(:, ref_idx));
    rel_theta_min = wrap_to_pi(theta_min_traj - theta_min_traj(:, ref_idx));

    sim_res = struct();
    sim_res.time = time;
    sim_res.dt = dt;
    sim_res.grad_gain = grad_gain;
    sim_res.theta_init = theta0;
    sim_res.max_traj = struct('theta', theta_max_traj, 'relative_phase', rel_theta_max, 'cw', cw_max_traj);
    sim_res.min_traj = struct('theta', theta_min_traj, 'relative_phase', rel_theta_min, 'cw', cw_min_traj);
end

% =========================================================================
% DATA LOADING
% =========================================================================

function [W, agent_ids, source_path] = load_coupling_matrix(source)
    if isempty(source)
        source = fullfile('EstimateL', 'SStickFlat', 'low_rank_analysis', 'M10', ...
            'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
    end

    % Case 1: Direct numeric matrix passed
    if isnumeric(source) && ismatrix(source) && size(source, 1) == size(source, 2)
        W = source;
        N = size(W, 1);
        agent_ids = 1:N;
        source_path = 'Direct numeric matrix';
        return;
    end

    % Case 2: Struct passed (e.g. from global_joint_cp_rank1_profile_free_network_svd)
    if isstruct(source) && isfield(source, 'W')
        W = source.W;
        if isfield(source, 'agent_ids')
            agent_ids = source.agent_ids(:).';
        else
            agent_ids = 1:size(W, 1);
        end
        source_path = 'Struct field W';
        return;
    end

    % Case 3: Character / string path
    if ischar(source) || isstring(source)
        source_str = char(source);
    else
        error('Invalid source format for weight matrix W.');
    end

    % Check if source is a dataset name like 'SStickFlat', 'Round', etc.
    candidate_csv = fullfile(source_str, 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');
    estimate_candidate = fullfile('EstimateL', source_str, 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd', 'network_coupling_matrix_W.csv');

    if exist(source_str, 'file') && ~isfolder(source_str)
        target_file = source_str;
    elseif exist(candidate_csv, 'file')
        target_file = candidate_csv;
    elseif exist(estimate_candidate, 'file')
        target_file = estimate_candidate;
    else
        % Try relative to script folder
        base_dir = fileparts(fileparts(mfilename('fullpath')));
        local_cand = fullfile(base_dir, source_str);
        if exist(local_cand, 'file') && ~isfolder(local_cand)
            target_file = local_cand;
        else
            error('Coupling matrix W CSV not found at: %s', source_str);
        end
    end

    source_path = target_file;
    T = readtable(target_file);

    % Find sender columns: 'sender_agent_8', 'sender_agent_9', ...
    var_names = T.Properties.VariableNames;
    sender_cols = var_names(startsWith(var_names, 'sender_agent_'));

    if ~isempty(sender_cols)
        agent_ids = zeros(1, numel(sender_cols));
        for k = 1:numel(sender_cols)
            agent_ids(k) = str2double(regexprep(sender_cols{k}, '^sender_agent_', ''));
        end
        W = table2array(T(:, sender_cols));
    else
        % Fallback: numeric columns from 2nd to end
        agent_ids = 1:height(T);
        W = table2array(T(:, 2:end));
    end

    % Ensure diagonal is exactly zero
    N = size(W, 1);
    W(1:N+1:end) = 0;
end

% =========================================================================
% PLOTTING FUNCTIONS
% =========================================================================

function fig = plot_gradient_flow_trajectories(sim_res, agent_ids, ref_agent_id, offset, max_cw, min_cw)
    fig = figure('Color', 'w', 'Position', [100, 100, 1050, 650], ...
        'Name', 'Gradient Flow Relaxation Dynamics of C_W(theta)');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    time = sim_res.time;
    N = numel(agent_ids);
    colors = lines(N);

    % --- Subplot 1: Maximization Relative Phase Trajectories ---
    ax1 = nexttile(1);
    hold(ax1, 'on');
    for k = 1:N
        y_val = sim_res.max_traj.relative_phase(:, k);
        % Insert NaNs across phase jump wrapping
        for j = 2:numel(y_val)
            if abs(y_val(j) - y_val(j-1)) > pi
                y_val(j) = NaN;
            end
        end
        plot(ax1, time, y_val, 'LineWidth', 1.6, 'Color', colors(k, :), ...
            'DisplayName', sprintf('$\\mathrm{Agent\\ %d}$', agent_ids(k) + offset));
    end
    format_angle_axis(ax1, time);
    ylabel(ax1, sprintf('$\\theta_j - \\theta_{%d}$ (rad)', ref_agent_id + offset), 'Interpreter', 'latex', 'FontSize', 12);
    title(ax1, 'Maximization Flow: Relative Phase Evolution', 'Interpreter', 'latex', 'FontSize', 12);
    legend(ax1, 'Location', 'eastoutside', 'Interpreter', 'latex');

    % --- Subplot 2: Minimization Relative Phase Trajectories ---
    ax2 = nexttile(2);
    hold(ax2, 'on');
    for k = 1:N
        y_val = sim_res.min_traj.relative_phase(:, k);
        for j = 2:numel(y_val)
            if abs(y_val(j) - y_val(j-1)) > pi
                y_val(j) = NaN;
            end
        end
        plot(ax2, time, y_val, 'LineWidth', 1.6, 'Color', colors(k, :), ...
            'DisplayName', sprintf('$\\mathrm{Agent\\ %d}$', agent_ids(k) + offset));
    end
    format_angle_axis(ax2, time);
    ylabel(ax2, sprintf('$\\theta_j - \\theta_{%d}$ (rad)', ref_agent_id + offset), 'Interpreter', 'latex', 'FontSize', 12);
    title(ax2, 'Minimization Flow: Relative Phase Evolution', 'Interpreter', 'latex', 'FontSize', 12);
    legend(ax2, 'Location', 'eastoutside', 'Interpreter', 'latex');

    % --- Subplot 3: Maximization C_W(t) Objective Evolution ---
    ax3 = nexttile(3);
    hold(ax3, 'on');
    plot(ax3, time, sim_res.max_traj.cw, 'b-', 'LineWidth', 2.0, 'DisplayName', '$\mathcal{C}_W(t)$');
    yline(ax3, max_cw, 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('$\\mathcal{C}_W^{\\max} = %+0.5f$', max_cw));
    grid(ax3, 'on'); box(ax3, 'on');
    xlim(ax3, [time(1), time(end)]);
    set(ax3, 'TickLabelInterpreter', 'latex');
    xlabel(ax3, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax3, '$\mathcal{C}_W(\theta)$', 'Interpreter', 'latex', 'FontSize', 13);
    title(ax3, 'Objective Ascent: $\mathcal{C}_W(t)$ Monotonic Increase', 'Interpreter', 'latex', 'FontSize', 12);
    legend(ax3, 'Location', 'southeast', 'Interpreter', 'latex');

    % --- Subplot 4: Minimization C_W(t) Objective Evolution ---
    ax4 = nexttile(4);
    hold(ax4, 'on');
    plot(ax4, time, sim_res.min_traj.cw, 'r-', 'LineWidth', 2.0, 'DisplayName', '$\mathcal{C}_W(t)$');
    yline(ax4, min_cw, 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('$\\mathcal{C}_W^{\\min} = %+0.5f$', min_cw));
    grid(ax4, 'on'); box(ax4, 'on');
    xlim(ax4, [time(1), time(end)]);
    set(ax4, 'TickLabelInterpreter', 'latex');
    xlabel(ax4, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax4, '$\mathcal{C}_W(\theta)$', 'Interpreter', 'latex', 'FontSize', 13);
    title(ax4, 'Objective Descent: $\mathcal{C}_W(t)$ Monotonic Decrease', 'Interpreter', 'latex', 'FontSize', 12);
    legend(ax4, 'Location', 'northeast', 'Interpreter', 'latex');

    apply_figure_styling(fig);
end

function fig = plot_optimal_phasor_diagrams(results, offset)
    fig = figure('Color', 'w', 'Position', [150, 150, 950, 460], ...
        'Name', 'Optimal Phase Configurations on Complex Unit Circle');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Subplot 1: Maximum Coherence Configuration
    ax1 = nexttile(1);
    render_single_phasor_circle(ax1, results.maximum, results.agent_ids, results.W_sym, offset, ...
        sprintf('Global Maximum: $\\mathcal{C}_W = %+0.5f$', results.maximum.cw_value));

    % Subplot 2: Minimum Coherence Configuration
    ax2 = nexttile(2);
    render_single_phasor_circle(ax2, results.minimum, results.agent_ids, results.W_sym, offset, ...
        sprintf('Global Minimum: $\\mathcal{C}_W = %+0.5f$', results.minimum.cw_value));

    apply_figure_styling(fig);
end

function render_single_phasor_circle(ax, state, agent_ids, W_sym, offset, title_str)
    hold(ax, 'on');
    axis(ax, 'equal');
    xlim(ax, [-1.45, 1.45]);
    ylim(ax, [-1.45, 1.45]);
    grid(ax, 'on'); box(ax, 'on');
    set(ax, 'TickLabelInterpreter', 'latex');

    % Draw unit circle
    circle_th = linspace(0, 2*pi, 300);
    plot(ax, cos(circle_th), sin(circle_th), 'Color', [0.75, 0.75, 0.75], 'LineWidth', 1.2);
    plot(ax, [-1.35, 1.35], [0, 0], 'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.8);
    plot(ax, [0, 0], [-1.35, 1.35], 'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.8);

    N = numel(agent_ids);
    th_vals = state.theta_rad;
    x_coords = cos(th_vals);
    y_coords = sin(th_vals);

    % Draw coupling interaction lines
    max_w = max(abs(W_sym(:))) + eps;
    for j = 1:N
        for k = (j+1):N
            w_jk = W_sym(j, k);
            if abs(w_jk) < 1e-6
                continue;
            end
            lw = 0.5 + 3.0 * (abs(w_jk) / max_w);
            if w_jk > 0
                line_color = [0.1, 0.45, 0.85, 0.6]; % Blue for positive coupling
            else
                line_color = [0.85, 0.2, 0.1, 0.6];  % Red for negative coupling
            end
            plot(ax, [x_coords(j), x_coords(k)], [y_coords(j), y_coords(k)], ...
                'Color', line_color, 'LineWidth', lw);
        end
    end

    % Draw agent nodes on the circle with grouped cluster labels
    colors = lines(N);
    xlim(ax, [-1.55, 1.55]);
    ylim(ax, [-1.55, 1.55]);

    visited = false(N, 1);
    for i = 1:N
        if visited(i), continue; end
        cluster_members = i;
        for j = i+1:N
            if abs(wrap_to_pi(th_vals(i) - th_vals(j))) < 0.15
                cluster_members = [cluster_members, j]; %#ok<AGROW>
            end
        end
        visited(cluster_members) = true;

        % Plot markers for all members
        for m = 1:numel(cluster_members)
            idx = cluster_members(m);
            plot(ax, x_coords(idx), y_coords(idx), 'o', 'MarkerSize', 11, ...
                'MarkerFaceColor', colors(idx, :), 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
        end

        % Compute centroid angle for label placement
        mean_th = angle(mean(exp(1i * th_vals(cluster_members))));
        cx = cos(mean_th);
        cy = sin(mean_th);

        % Construct unified label
        id_strs = arrayfun(@(id) sprintf('%d', id + offset), agent_ids(cluster_members), 'UniformOutput', false);
        id_combined = strjoin(id_strs, ',');
        deg_val = rad2deg(mean_th);

        tx = 1.25 * cx;
        ty = 1.25 * cy;
        text(ax, tx, ty, sprintf('$\\mathrm{ID\\ %s}$\n$(%+0.0f^\\circ)$', id_combined, deg_val), ...
            'Interpreter', 'latex', 'FontSize', 10, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'BackgroundColor', [1, 1, 1, 0.85], 'Margin', 1);
    end

    title(ax, title_str, 'Interpreter', 'latex', 'FontSize', 12);
    xlabel(ax, '$\mathrm{Re}(x_j)$', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax, '$\mathrm{Im}(x_j)$', 'Interpreter', 'latex', 'FontSize', 11);
end

function fig = plot_pairwise_energy_contributions(results, offset)
    fig = figure('Color', 'w', 'Position', [150, 150, 950, 400], ...
        'Name', 'Pairwise Energy Matrix & Spectral Bounds');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    agent_labels = arrayfun(@(id) sprintf('$\\mathrm{ID\\ %d}$', id + offset), results.agent_ids, 'UniformOutput', false);

    max_val = max([abs(results.maximum.energy_matrix(:)); abs(results.minimum.energy_matrix(:)); eps]);

    % Subplot 1: Maximum Energy Matrix
    ax1 = nexttile(1);
    imagesc(ax1, results.maximum.energy_matrix, [-max_val, max_val]);
    colormap(ax1, balance_colormap());
    colorbar(ax1);
    axis(ax1, 'square');
    set(ax1, 'XTick', 1:results.N, 'XTickLabel', agent_labels, 'YTick', 1:results.N, 'YTickLabel', agent_labels, ...
        'TickLabelInterpreter', 'latex');
    title(ax1, sprintf('Maximum Contributions $E_{jk}$\n$\\mathcal{C}_W = %+0.5f$', results.maximum.cw_value), ...
        'Interpreter', 'latex', 'FontSize', 11);
    xlabel(ax1, 'Agent $k$', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax1, 'Agent $j$', 'Interpreter', 'latex', 'FontSize', 11);

    % Subplot 2: Minimum Energy Matrix
    ax2 = nexttile(2);
    imagesc(ax2, results.minimum.energy_matrix, [-max_val, max_val]);
    colormap(ax2, balance_colormap());
    colorbar(ax2);
    axis(ax2, 'square');
    set(ax2, 'XTick', 1:results.N, 'XTickLabel', agent_labels, 'YTick', 1:results.N, 'YTickLabel', agent_labels, ...
        'TickLabelInterpreter', 'latex');
    title(ax2, sprintf('Minimum Contributions $E_{jk}$\n$\\mathcal{C}_W = %+0.5f$', results.minimum.cw_value), ...
        'Interpreter', 'latex', 'FontSize', 11);
    xlabel(ax2, 'Agent $k$', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax2, 'Agent $j$', 'Interpreter', 'latex', 'FontSize', 11);

    % Subplot 3: Spectral bounds comparison bar chart
    ax3 = nexttile(3);
    hold(ax3, 'on');
    grid(ax3, 'on'); box(ax3, 'on');
    categories = {'$\mathrm{Global\ Min}$', '$\mathrm{Spectral\ Min}$', '$\mathrm{Global\ Max}$', '$\mathrm{Spectral\ Max}$'};
    vals = [results.minimum.cw_value, results.spectral_bounds(1), ...
            results.maximum.cw_value, results.spectral_bounds(2)];
    bar_colors = [0.85, 0.2, 0.2; 0.95, 0.6, 0.6; 0.2, 0.45, 0.85; 0.6, 0.75, 0.95];
    b = bar(ax3, 1:4, vals, 'FaceColor', 'flat');
    b.CData = bar_colors;
    set(ax3, 'XTick', 1:4, 'XTickLabel', categories, 'TickLabelInterpreter', 'latex', 'XTickLabelRotation', 20);
    ylabel(ax3, 'Objective Value $\mathcal{C}_W$', 'Interpreter', 'latex', 'FontSize', 11);
    title(ax3, 'Optimal vs Spectral Bounds', 'Interpreter', 'latex', 'FontSize', 12);

    apply_figure_styling(fig);
end

function cmap = balance_colormap()
    N_c = 256;
    r = [linspace(0.1, 0.95, N_c/2), linspace(0.95, 0.8, N_c/2)].';
    g = [linspace(0.2, 0.95, N_c/2), linspace(0.95, 0.2, N_c/2)].';
    b = [linspace(0.8, 0.95, N_c/2), linspace(0.95, 0.1, N_c/2)].';
    cmap = [r, g, b];
end

function format_angle_axis(ax, time)
    grid(ax, 'on'); box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    ylim(ax, [-pi, pi]);
    yticks(ax, [-pi, -pi/2, 0, pi/2, pi]);
    yticklabels(ax, {'$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'});
    set(ax, 'TickLabelInterpreter', 'latex');
    xlabel(ax, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 11);
end

function apply_figure_styling(fig)
    set(fig, 'Color', 'w');
    set(fig, 'DefaultTextInterpreter', 'latex');
    set(fig, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(fig, 'DefaultLegendInterpreter', 'latex');

    all_axes = findobj(fig, 'Type', 'axes');
    for a = 1:numel(all_axes)
        ax = all_axes(a);
        set(ax, 'FontName', 'Times', 'Box', 'on', 'TickLabelInterpreter', 'latex');
        cur_fs = get(ax, 'FontSize');
        if isempty(cur_fs) || cur_fs > 13 || cur_fs < 10
            set(ax, 'FontSize', 11);
        end

        if ~isempty(ax.Title) && ~isempty(ax.Title.String)
            set(ax.Title, 'Interpreter', 'latex');
        end
        if ~isempty(ax.XLabel) && ~isempty(ax.XLabel.String)
            set(ax.XLabel, 'Interpreter', 'latex');
        end
        if ~isempty(ax.YLabel) && ~isempty(ax.YLabel.String)
            set(ax.YLabel, 'Interpreter', 'latex');
        end
    end

    all_legends = findobj(fig, 'Type', 'legend');
    for l = 1:numel(all_legends)
        set(all_legends(l), 'Interpreter', 'latex', 'FontSize', 10);
    end

    all_texts = findobj(fig, 'Type', 'text');
    for t = 1:numel(all_texts)
        set(all_texts(t), 'Interpreter', 'latex');
    end
end

function export_figure_to_png(fig, file_path)
    apply_figure_styling(fig);
    has_exportgraphics = ~isempty(which('exportgraphics')) || (exist('exportgraphics', 'builtin') == 5);
    if has_exportgraphics
        exportgraphics(fig, file_path, 'Resolution', 300);
    else
        saveas(fig, file_path);
    end
end

% =========================================================================
% PRINT & EXPORT HELPERS
% =========================================================================

function print_results_summary(res, offset)
    fprintf('-------------------------------------------------------------------------\n');
    fprintf('  OPTIMAL RESULTS SUMMARY\n');
    fprintf('-------------------------------------------------------------------------\n');

    % Print Maximum
    fprintf('  [GLOBAL MAXIMUM]\n');
    fprintf('    Objective C_W = %+8.6f  (Upper Spectral Bound = %+8.6f)\n', ...
        res.maximum.cw_value, res.spectral_bounds(2));
    fprintf('    Relative Phases (deg):\n');
    for i = 1:res.N
        fprintf('      Agent ID %2d (idx %d):  %+8.2f deg  (%+7.4f rad)\n', ...
            res.agent_ids(i) + offset, i, res.maximum.theta_deg(i), res.maximum.theta_rad(i));
    end
    fprintf('\n');

    % Print Minimum
    fprintf('  [GLOBAL MINIMUM]\n');
    fprintf('    Objective C_W = %+8.6f  (Lower Spectral Bound = %+8.6f)\n', ...
        res.minimum.cw_value, res.spectral_bounds(1));
    fprintf('    Relative Phases (deg):\n');
    for i = 1:res.N
        fprintf('      Agent ID %2d (idx %d):  %+8.2f deg  (%+7.4f rad)\n', ...
            res.agent_ids(i) + offset, i, res.minimum.theta_deg(i), res.minimum.theta_rad(i));
    end
    fprintf('\n');

    % Print all discovered stationary extrema / equilibria
    if isfield(res, 'extrema') && ~isempty(res.extrema)
        fprintf('-------------------------------------------------------------------------\n');
        fprintf('  ALL IDENTIFIED STATIONARY EXTREMA & EQUILIBRIA (N_extrema = %d)\n', numel(res.extrema));
        fprintf('-------------------------------------------------------------------------\n');
        fprintf('  #   Type                       C_W        |R|    Hits  Phases (deg) [Agent %s]\n', ...
            mat2str(res.agent_ids + offset));
        for e = 1:numel(res.extrema)
            ex = res.extrema(e);
            marker = ' ';
            if isfield(ex, 'is_global_max') && ex.is_global_max
                marker = '*';
            elseif isfield(ex, 'is_global_min') && ex.is_global_min
                marker = '-';
            end
            deg_str = sprintf('%+6.1f ', ex.theta_deg);
            fprintf('  %2d%c %-26s %+9.6f  %5.3f  %4d  [%s]\n', ...
                e, marker, ex.type, ex.cw, ex.order_param, ex.count, strtrim(deg_str));
        end
        fprintf('  (*: Global Maximum / Stable Attractor, -: Global Minimum / Repeller)\n');
    end
    fprintf('-------------------------------------------------------------------------\n\n');
end

function export = save_analysis_outputs(results, opts)
    out_dir = opts.OutputDir;
    if isempty(out_dir)
        source_dir = fileparts(results.source_path);
        out_dir = fullfile(source_dir, 'cw_optimal_phase_analysis');
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % 1. Save optimal phase summary CSV (global max & min)
    summary_csv = fullfile(out_dir, 'cw_optimal_phases_summary.csv');
    var_names = {'agent_id', 'max_phase_deg', 'max_phase_rad', 'min_phase_deg', 'min_phase_rad'};
    T_summary = table(results.agent_ids(:), results.maximum.theta_deg(:), results.maximum.theta_rad(:), ...
                      results.minimum.theta_deg(:), results.minimum.theta_rad(:), ...
                      'VariableNames', var_names);
    writetable(T_summary, summary_csv);

    % 2. Save all identified stationary extrema & equilibria CSV
    extrema_csv = fullfile(out_dir, 'cw_extrema_phase_relationships.csv');
    if isfield(results, 'extrema') && ~isempty(results.extrema)
        M_ext = numel(results.extrema);
        ext_ids = (1:M_ext).';
        ext_types = {results.extrema.type}.';
        ext_type_codes = {results.extrema.type_code}.';
        is_gmax = [results.extrema.is_global_max].';
        is_gmin = [results.extrema.is_global_min].';
        cw_vals = [results.extrema.cw].';
        order_params = [results.extrema.order_param].';
        grad_norms = [results.extrema.grad_norm].';
        basin_hits = [results.extrema.count].';

        T_ext = table(ext_ids, ext_types, ext_type_codes, is_gmax, is_gmin, ...
                      cw_vals, order_params, grad_norms, basin_hits, ...
                      'VariableNames', {'extrema_id', 'classification', 'type_code', ...
                      'is_global_max', 'is_global_min', 'cw_value', 'order_parameter', ...
                      'gradient_norm', 'basin_hits'});

        % Append phase degrees for each agent
        for a = 1:results.N
            col_deg = sprintf('phi_deg_agent_%d', results.agent_ids(a));
            col_rad = sprintf('phi_rad_agent_%d', results.agent_ids(a));
            vals_deg = zeros(M_ext, 1);
            vals_rad = zeros(M_ext, 1);
            for e = 1:M_ext
                vals_deg(e) = results.extrema(e).theta_deg(a);
                vals_rad(e) = results.extrema(e).theta_rad(a);
            end
            T_ext.(col_deg) = vals_deg;
            T_ext.(col_rad) = vals_rad;
        end
        writetable(T_ext, extrema_csv);
    else
        extrema_csv = '';
    end

    % 3. Save complete MAT results (exclude figure handles)
    mat_path = fullfile(out_dir, 'cw_optimal_phase_results.mat');
    res_save = results;
    if isfield(res_save, 'figures')
        res_save = rmfield(res_save, 'figures');
    end
    save(mat_path, 'res_save', '-v7.3');

    % 4. Save PNG figures if open
    saved_figs = {};
    if isfield(results, 'figures')
        if ~isempty(results.figures.trajectories) && isvalid(results.figures.trajectories)
            p_traj = fullfile(out_dir, 'cw_gradient_flow_trajectories.png');
            export_figure_to_png(results.figures.trajectories, p_traj);
            saved_figs{end+1} = p_traj; %#ok<AGROW>
        end
        if ~isempty(results.figures.phasors) && isvalid(results.figures.phasors)
            p_phasor = fullfile(out_dir, 'cw_optimal_phasor_diagrams.png');
            export_figure_to_png(results.figures.phasors, p_phasor);
            saved_figs{end+1} = p_phasor; %#ok<AGROW>
        end
        if ~isempty(results.figures.energy_matrix) && isvalid(results.figures.energy_matrix)
            p_energy = fullfile(out_dir, 'cw_pairwise_energy_matrix.png');
            export_figure_to_png(results.figures.energy_matrix, p_energy);
            saved_figs{end+1} = p_energy; %#ok<AGROW>
        end
    end

    export = struct('output_dir', out_dir, 'summary_csv', summary_csv, ...
                    'extrema_csv', extrema_csv, 'mat_path', mat_path, 'saved_figures', {saved_figs});
    fprintf('[INFO] Saved analysis outputs to: %s\n', out_dir);
    if ~isempty(extrema_csv)
        fprintf('[INFO] Extrema phase relationships table: %s\n', extrema_csv);
    end
end

function theta_norm = normalize_phase_vector(theta, ref_idx)
    theta = theta(:);
    % Gauge shift relative to reference agent
    shifted = theta - theta(ref_idx);
    theta_norm = wrap_to_pi(shifted);
    theta_norm(ref_idx) = 0; % Force exact zero
end

function phase_wrapped = wrap_to_pi(phase)
    phase_wrapped = atan2(sin(phase), cos(phase));
end

function opts = parse_options(def_num_starts, def_rand_seed, def_sim_time, def_dt, def_gain, ...
    def_ref_id, def_plot_traj, def_plot_phasors, def_plot_energy, def_offset, def_save, def_outdir, varargin)

    p = inputParser;
    p.CaseSensitive = false;
    p.KeepUnmatched = true;

    addParameter(p, 'NumStarts', def_num_starts, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'RandomSeed', def_rand_seed, @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'SimTimeSec', def_sim_time, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'SimDt', def_dt, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'GradGain', def_gain, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'ReferenceAgentId', def_ref_id, @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'PlotTrajectories', def_plot_traj, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotPhasors', def_plot_phasors, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotEnergyMatrix', def_plot_energy, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'AgentOffset', def_offset, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'SaveOutputs', def_save, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'OutputDir', def_outdir, @(x) ischar(x) || isstring(x));

    % Common aliases
    addParameter(p, 'SavePlots', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'SavePlot', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'SaveOutput', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotTraj', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotPhasor', [], @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PlotEnergy', [], @(x) islogical(x) || isnumeric(x));

    parse(p, varargin{:});
    opts = p.Results;

    % Resolve aliases
    if ~isempty(opts.SavePlots)
        opts.SaveOutputs = opts.SavePlots;
    end
    if ~isempty(opts.SavePlot)
        opts.SaveOutputs = opts.SavePlot;
    end
    if ~isempty(opts.SaveOutput)
        opts.SaveOutputs = opts.SaveOutput;
    end
    if ~isempty(opts.PlotTraj)
        opts.PlotTrajectories = opts.PlotTraj;
    end
    if ~isempty(opts.PlotPhasor)
        opts.PlotPhasors = opts.PlotPhasor;
    end
    if ~isempty(opts.PlotEnergy)
        opts.PlotEnergyMatrix = opts.PlotEnergy;
    end

    opts.OutputDir = char(opts.OutputDir);
    opts.PlotTrajectories = logical(opts.PlotTrajectories);
    opts.PlotPhasors = logical(opts.PlotPhasors);
    opts.PlotEnergyMatrix = logical(opts.PlotEnergyMatrix);
    opts.SaveOutputs = logical(opts.SaveOutputs);
end
