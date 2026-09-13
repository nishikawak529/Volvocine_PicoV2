function out = simulate_cp_rank1_relative_phase_dynamics(target_dir, M, varargin)
% SIMULATE_CP_RANK1_RELATIVE_PHASE_DYNAMICS
% Simulate relative phase dynamics using the Rank-1 Common Profile model
% identified by global_joint_cp_rank1_profile_free_network_svd.
%
% Model formulation:
%   s_{i <- j}(phi_i, phi_j) = W_{ij} * a(phi_i) * b(phi_j)
%   b(phi) = sqrt(2) * cos(phi - delta)
%   a(phi) = sum_{m=-M}^M A_m * exp(1i * m * phi)
%   W in R^{N x N} is the signed directed coupling network (W_{ii} = 0).
%
% Dynamics modes:
%   1. Phase-averaged Gamma Dynamics (use_original_system = false, default):
%      dphi_i/dt = omega_i + sigma * sum_{j=1}^N W_{ij} * Gamma_0(phi_i - phi_j)
%      where Gamma_0(psi) is the single global coupling function:
%      Gamma_0(psi) = (1 / 2*pi) * integral_0^{2*pi} z(theta) * a(theta) * b(theta - psi) dtheta
%      with z(theta) = -sin(theta) (PRC).
%
%   2. Original System Dynamics (use_original_system = true):
%      dphi_i/dt = omega_i + sigma * a(phi_i) * z(phi_i) * [W * b(phi)]_i
%                  + sigma * q_i(phi_i) * z(phi_i)  (if add_self_feedback = true)
%
% Network approximations:
%   - Full identified W matrix (default)
%   - Low-rank network approximation W^{(K)} = sum_{l=1}^K d_l * p_l * q_l.'
%     via SVD or Sparse PMD by setting 'network_mode_k', K.
%
% Examples:
%   % Default SStickFlat simulation
%   out = simulate_cp_rank1_relative_phase_dynamics();
%
%   % Simulation for Round dataset with M=10
%   out = simulate_cp_rank1_relative_phase_dynamics('Round', 10);
%
%   % Low-rank network simulation (Rank-1 network mode)
%   out = simulate_cp_rank1_relative_phase_dynamics('SStickFlat', 10, 'network_mode_k', 1);
%
%   % Original 2D dynamics with common profile
%   out = simulate_cp_rank1_relative_phase_dynamics('SStickFlat', 10, 'use_original_system', true);

    % =========================================================================
    % USER CONFIGURATION PRESETS (For F5 / Run without arguments)
    % Easily change target datasets, control gains, and simulation modes here!
    % =========================================================================
    
    % --- 1. Target Dataset & Fourier Order ---
    DEFAULT_DATASET = 'SStickFlat';   % 'SStickFlat', 'Round', 'Round6', 'Stick', 'SStick'
    DEFAULT_FOURIER_M = 10;           % Fourier truncation order M

    % --- 2. Control Gains (Overall & Self) ---
    DEFAULT_SIGMA = 10.0;              % Coupling / feedback control gain sigma (scalar or vector)
    DEFAULT_SIGMA_SELF = [];          % Self-feedback gain (empty [] uses DEFAULT_SIGMA)

    % --- 3. Dynamics & Reduction Mode ---
    DEFAULT_USE_ORIGINAL_SYSTEM = false;   % false: Phase-averaged Gamma dynamics, true: Original 2D dynamics
    DEFAULT_SUBTRACT_SELF_PROFILE = true;  % true: Account for individual self-profiles
    DEFAULT_ADD_SELF_FEEDBACK = true;      % true: Add self-profile feedback term
    DEFAULT_REMOVE_GAMMA_BIAS = true;      % true: Subtract mean bias from Gamma_0(psi)
    DEFAULT_USE_FIRST_HARMONIC = false;    % true: Approximate Gamma_0 with 1st harmonic (c0 + c1*sin + c2*cos)

    % --- 4. Network Approximation / Decomposition Mode ---
    DEFAULT_NETWORK_MODE_K = [];           % []: Full network W, or integer K (1, 2, ...) for Rank-K approximation
    DEFAULT_DECOMPOSITION = 'svd';         % 'svd' or 'sparse_pmd'
    DEFAULT_FORCE_ZERO_DIAGONAL = true;    % true: Force diagonal to 0 (no self-coupling in network matrix)

    % --- 5. Simulation Timing & Frequency ---
    DEFAULT_SIMULATION_DURATION_SEC = 500; % Simulation duration in seconds
    DEFAULT_SIMULATION_DT = 0.01;          % Integration step dt (s)
    DEFAULT_OMEGA_RAD_S = 2.5*pi;          % Natural frequency (rad/s), scalar or vector
    DEFAULT_REFERENCE_AGENT_ID = [];       % Reference agent ID for relative phase ([] uses 1st agent)

    % --- 6. Plotting & Visualization Flags ---
    DEFAULT_PLOT_RELATIVE_PHASE       = true;   % Plot relative phase trajectories
    DEFAULT_PLOT_ABSOLUTE_PHASES      = false;  % Plot absolute phase trajectories
    DEFAULT_PLOT_MODE_AMPLITUDES      = true;   % Plot SVD/PMD mode order parameter amplitudes |Z_l(t)| (like plot_relative_phase.m)
    DEFAULT_PLOT_AGENT_INPUT_AMP      = false;  % Plot agent total weighted input amplitude |G_i(t)|
    DEFAULT_PLOT_GAMMA                = false;  % Plot common profiles a(phi), b(phi) and Gamma_0(psi)
    DEFAULT_PLOT_NETWORK_WEIGHTS      = false;  % Plot network coupling matrix W heatmap
    DEFAULT_PLOT_ALL_GAMMA            = false;  % Plot all pairwise coupling curves
    DEFAULT_AGENT_DISPLAY_OFFSET      = 0;      % Display ID offset (e.g. -7 to show 8->1, 9->2, ...)
    DEFAULT_SAVE_OUTPUT               = false;  % Save outputs to CSV / MAT
    % =========================================================================

    % Pack defaults struct
    defaults = struct( ...
        'dataset', DEFAULT_DATASET, ...
        'M', DEFAULT_FOURIER_M, ...
        'sigma', DEFAULT_SIGMA, ...
        'sigma_self', DEFAULT_SIGMA_SELF, ...
        'use_original_system', DEFAULT_USE_ORIGINAL_SYSTEM, ...
        'subtract_self_profile', DEFAULT_SUBTRACT_SELF_PROFILE, ...
        'add_self_feedback', DEFAULT_ADD_SELF_FEEDBACK, ...
        'remove_gamma_bias', DEFAULT_REMOVE_GAMMA_BIAS, ...
        'use_first_harmonic', DEFAULT_USE_FIRST_HARMONIC, ...
        'network_mode_k', DEFAULT_NETWORK_MODE_K, ...
        'decomposition', DEFAULT_DECOMPOSITION, ...
        'force_zero_diagonal', DEFAULT_FORCE_ZERO_DIAGONAL, ...
        'simulation_duration_sec', DEFAULT_SIMULATION_DURATION_SEC, ...
        'simulation_dt', DEFAULT_SIMULATION_DT, ...
        'omega_rad_s', DEFAULT_OMEGA_RAD_S, ...
        'reference_agent_id', DEFAULT_REFERENCE_AGENT_ID, ...
        'plot_relative_phase', DEFAULT_PLOT_RELATIVE_PHASE, ...
        'plot_absolute_phases', DEFAULT_PLOT_ABSOLUTE_PHASES, ...
        'plot_mode_amplitudes', DEFAULT_PLOT_MODE_AMPLITUDES, ...
        'plot_agent_input_amp', DEFAULT_PLOT_AGENT_INPUT_AMP, ...
        'plot_gamma', DEFAULT_PLOT_GAMMA, ...
        'plot_network_weights', DEFAULT_PLOT_NETWORK_WEIGHTS, ...
        'plot_all_gamma', DEFAULT_PLOT_ALL_GAMMA, ...
        'agent_display_offset', DEFAULT_AGENT_DISPLAY_OFFSET, ...
        'save_output', DEFAULT_SAVE_OUTPUT);

    % Flexible argument dispatch:
    % Allow calls like:
    %   simulate_cp_rank1_relative_phase_dynamics()               -> uses presets
    %   simulate_cp_rank1_relative_phase_dynamics(8.5)            -> sigma = 8.5
    %   simulate_cp_rank1_relative_phase_dynamics(8.5, 'plot_relative_phase', false) -> sigma = 8.5
    %   simulate_cp_rank1_relative_phase_dynamics('SStickFlat', 10, 8.5) -> sigma = 8.5
    %   simulate_cp_rank1_relative_phase_dynamics('SStickFlat', 8.5)     -> sigma = 8.5, M = 10
    %   simulate_cp_rank1_relative_phase_dynamics(..., 'gain', 8.5)      -> alias for sigma
    %   simulate_cp_rank1_relative_phase_dynamics(..., 'sigma', 8.5, 'sigma_self', 2.0)
    raw_args = varargin;
    if nargin == 0
        target_dir = defaults.dataset;
        M = defaults.M;
    elseif isnumeric(target_dir) && ~isempty(target_dir)
        % First argument is numeric: user passed control gain sigma directly!
        sigma_val = target_dir;
        target_dir = defaults.dataset;
        if nargin >= 2 && isnumeric(M) && isscalar(M) && rem(M, 1) == 0 && M <= 30 && M > 0
            % e.g. (8.5, 10, 'plot_relative_phase', false)
            raw_args = [{'sigma', sigma_val}, raw_args];
        else
            % e.g. (8.5, 'plot_relative_phase', false)
            if nargin >= 2
                raw_args = [{'sigma', sigma_val}, {M}, raw_args];
            else
                raw_args = {'sigma', sigma_val};
            end
            M = defaults.M;
        end
    elseif nargin == 1
        if is_option_name(target_dir)
            raw_args = [{target_dir}];
            target_dir = defaults.dataset;
            M = defaults.M;
        else
            M = defaults.M;
        end
    elseif nargin == 2
        if is_option_name(target_dir)
            raw_args = [{target_dir}, {M}];
            target_dir = defaults.dataset;
            M = defaults.M;
        elseif isnumeric(M)
            % If M is non-integer or > 30, treat as control gain: simulate_cp_rank1_relative_phase_dynamics('SStickFlat', 8.5)
            if rem(M, 1) ~= 0 || M > 30 || M <= 0
                raw_args = {'sigma', M};
                M = defaults.M;
            end
        elseif ischar(M) || isstring(M)
            raw_args = [{M}];
            M = defaults.M;
        end
    else
        % nargin >= 3
        if is_option_name(target_dir)
            raw_args = [{target_dir}, {M}, raw_args];
            target_dir = defaults.dataset;
            M = defaults.M;
        elseif isnumeric(varargin{1}) && ~isempty(varargin{1}) && (isscalar(varargin{1}) || isvector(varargin{1}))
            % 3rd argument is numeric: simulate_cp_rank1_relative_phase_dynamics('SStickFlat', 10, 8.5, ...)
            sigma_val = varargin{1};
            raw_args = [{'sigma', sigma_val}, varargin(2:end)];
        elseif ~isnumeric(M) && (ischar(M) || isstring(M))
            raw_args = [{M}, raw_args];
            M = defaults.M;
        end
    end

    if isempty(target_dir)
        target_dir = defaults.dataset;
    end
    if isempty(M)
        M = defaults.M;
    end
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, mfilename, 'M');

    resolved_dir = resolve_dataset_directory(target_dir);

    % Parse options
    opts = parse_options(defaults, raw_args{:});
    agent_display_offset = opts.agent_display_offset;

    % 1. Obtain CP Rank-1 model (from input, cached files, or by running estimation)
    cp_model = obtain_cp_rank1_model(resolved_dir, M, opts);

    node_ids = cp_model.agent_ids(:);
    N = numel(node_ids);
    if N < 2
        error('At least 2 agents are required for network simulation.');
    end

    % 2. Determine coupling network matrix W to be used
    [W_sim, W_full, network_info] = build_simulation_network(cp_model, opts);

    % 3. Compute the common phase-interaction function Gamma_0(psi)
    gamma0 = compute_common_gamma_function(cp_model, M, opts);

    % 4. Load self-profiles if requested
    node_self_profiles = cell(N, 1);
    if opts.subtract_self_profile || opts.add_self_feedback
        node_self_profiles = load_agent_self_profiles(resolved_dir, M, node_ids, opts.self_profile_dir);
    end

    % 5. Setup simulation initial conditions and parameters
    if isempty(opts.initial_phases)
        phi0 = 2 * pi * rand(N, 1);
    else
        phi0 = opts.initial_phases(:);
        if numel(phi0) ~= N
            error('initial_phases must contain exactly %d values (got %d).', N, numel(phi0));
        end
    end

    if isscalar(opts.omega_rad_s)
        omega_rad_s = opts.omega_rad_s * ones(N, 1);
    else
        omega_rad_s = opts.omega_rad_s(:);
        if numel(omega_rad_s) ~= N
            error('omega_rad_s must be scalar or length %d vector.', N);
        end
    end

    time = (0:opts.simulation_dt:opts.simulation_duration_sec).';
    n_steps = numel(time);
    phase = nan(n_steps, N);
    phase(1, :) = phi0.';

    % Expand and validate control gains (sigma & sigma_self)
    if isscalar(opts.sigma)
        sigma = opts.sigma * ones(N, 1);
        gain_str = sprintf('\\sigma = %.2f', opts.sigma);
    else
        sigma = opts.sigma(:);
        if numel(sigma) ~= N
            error('Control gain sigma must be scalar or length %d vector (got %d).', N, numel(sigma));
        end
        gain_str = sprintf('\\sigma = [%s]', strjoin(arrayfun(@(v) sprintf('%.2f', v), sigma.', 'UniformOutput', false), ', '));
    end

    if isempty(opts.sigma_self)
        sigma_self = sigma;
    elseif isscalar(opts.sigma_self)
        sigma_self = opts.sigma_self * ones(N, 1);
    else
        sigma_self = opts.sigma_self(:);
        if numel(sigma_self) ~= N
            error('Self-feedback gain sigma_self must be scalar or length %d vector (got %d).', N, numel(sigma_self));
        end
    end

    network_info.gain_str = gain_str;
    network_info.sigma = sigma;
    network_info.sigma_self = sigma_self;

    fprintf('[INFO] Simulating CP Rank-1 Phase Dynamics (%s)\n', resolved_dir);
    fprintf('  Agents (%d): %s | Duration: %.1fs | dt: %.4fs\n', ...
        N, mat2str(node_ids.'), opts.simulation_duration_sec, opts.simulation_dt);
    fprintf('  Control Gain (sigma): %s | Self Gain (sigma_self): %s\n', ...
        mat2str(sigma.', 3), mat2str(sigma_self.', 3));
    fprintf('  System type: %s | Network mode: %s\n', ...
        ternary(opts.use_original_system, 'Original 2D Dynamics', 'Phase-averaged Gamma Dynamics'), ...
        network_info.desc);

    % 6. Time stepping integration
    dt = opts.simulation_dt;

    for t_idx = 1:(n_steps - 1)
        phi_curr = phase(t_idx, :).';
        if opts.use_original_system
            dphi = compute_phase_velocity_original(phi_curr, omega_rad_s, W_sim, ...
                cp_model, sigma, sigma_self, opts.add_self_feedback, node_self_profiles);
        else
            dphi = compute_phase_velocity_gamma(phi_curr, omega_rad_s, W_sim, ...
                gamma0, sigma, sigma_self, opts.add_self_feedback, node_self_profiles);
        end
        phase(t_idx + 1, :) = phase(t_idx, :) + dt * dphi.';
    end

    % 7. Relative phase calculation
    reference_agent_id = opts.reference_agent_id;
    if isempty(reference_agent_id)
        reference_agent_id = node_ids(1);
    end
    reference_idx = find(node_ids == reference_agent_id, 1, 'first');
    if isempty(reference_idx)
        error('reference_agent_id %d is not in simulated node set %s.', ...
            reference_agent_id, mat2str(node_ids.'));
    end

    relative_phase = wrap_to_pi(phase - phase(:, reference_idx));

    % 8. Mode Decomposition Time-Series Analysis (like plot_relative_phase.m)
    mode_analysis = compute_mode_amplitudes_time_series(phase, cp_model);

    % 9. Plotting
    sim_title = sprintf('%s, %s', network_info.desc, gain_str);
    figures = struct();
    if opts.plot_relative_phase
        figures.relative_phase = plot_relative_phase_trajectories( ...
            time, relative_phase, node_ids, reference_agent_id, agent_display_offset, sim_title);
    else
        figures.relative_phase = [];
    end

    if opts.plot_absolute_phases
        figures.absolute_phase = plot_absolute_phases(time, phase, node_ids, agent_display_offset);
    else
        figures.absolute_phase = [];
    end

    if opts.plot_mode_amplitudes && mode_analysis.available
        figures.mode_amplitudes = plot_mode_order_parameters( ...
            time, mode_analysis, sim_title);
    else
        figures.mode_amplitudes = [];
    end

    if opts.plot_agent_input_amp && mode_analysis.available
        figures.agent_input_amp = plot_agent_input_amplitudes( ...
            time, mode_analysis, node_ids, agent_display_offset, sim_title);
    else
        figures.agent_input_amp = [];
    end

    if opts.plot_gamma
        figures.common_profile = plot_common_profile_and_gamma(cp_model, gamma0, W_sim, node_ids, agent_display_offset);
    else
        figures.common_profile = [];
    end

    if opts.plot_network_weights
        figures.network_matrix = plot_network_matrix(W_sim, W_full, node_ids, agent_display_offset, network_info.desc);
    else
        figures.network_matrix = [];
    end

    if opts.plot_all_gamma
        figures.all_gamma = plot_pairwise_gamma_functions(W_sim, gamma0, node_ids, agent_display_offset);
    else
        figures.all_gamma = [];
    end

    % 10. Output struct assembling
    out = struct();
    out.target_dir = resolved_dir;
    out.M = M;
    out.options = opts;
    out.node_ids = node_ids;
    out.reference_agent_id = reference_agent_id;
    out.omega_rad_s = omega_rad_s;
    out.sigma = sigma;
    out.sigma_self = sigma_self;
    out.time = time;
    out.phase = phase;
    out.relative_phase = relative_phase;
    out.W = W_sim;
    out.W_full = W_full;
    out.network_info = network_info;
    out.cp_model = cp_model;
    out.gamma0 = gamma0;
    out.mode_analysis = mode_analysis;
    out.figures = figures;

    if opts.save_output
        out.export = save_simulation_outputs(out, opts);
    end

    fprintf('[INFO] Simulation completed successfully.\n');
end

% =========================================================================
% DYNAMICS COMPUTATION FUNCTIONS
% =========================================================================

function dphi = compute_phase_velocity_gamma(phi, omega_rad_s, W, gamma0, sigma, sigma_self, add_self_feedback, node_self_profiles)
% Phase-averaged Gamma Dynamics:
%   dphi_i/dt = omega_i + sigma_i * sum_j W_{ij} * Gamma_0(phi_i - phi_j) + sigma_self_i * q_i(phi_i) * z(phi_i)
    N = numel(phi);
    
    % Pairwise phase differences: Delta_Phi(i, j) = phi_i - phi_j
    % Row i: target/receiver, Column j: source/sender
    delta_phi = wrap_to_pi(phi - phi.');
    
    % Evaluate Gamma_0(delta_phi) using precomputed lookup table
    gamma_mat = interp1(gamma0.psi, gamma0.gamma0, delta_phi, 'linear', 'extrap');
    
    % Network coupling sum for each agent i: sum_j W_{ij} * Gamma_0(phi_i - phi_j)
    coupling_term = sum(W .* gamma_mat, 2);
    
    dphi = omega_rad_s + sigma .* coupling_term;

    % Optional self feedback: sigma_self * q_i(phi_i) * z(phi_i)
    if add_self_feedback && ~isempty(node_self_profiles)
        for i = 1:N
            sp = node_self_profiles{i};
            if ~isempty(sp)
                phi_w = mod(phi(i), 2*pi);
                q_val = interp1(sp.phi, sp.val, phi_w, 'linear', 'extrap');
                z_val = -sin(phi_w);
                dphi(i) = dphi(i) + sigma_self(i) * q_val * z_val;
            end
        end
    end
end

function dphi = compute_phase_velocity_original(phi, omega_rad_s, W, cp_model, sigma, sigma_self, add_self_feedback, node_self_profiles)
% Original 2D Dynamics with Common Profile:
%   s_{i <- j}(phi_i, phi_j) = W_{ij} * a(phi_i) * b(phi_j)
%   Coupling term: sigma_i * a(phi_i) * z(phi_i) * [W * b(phi)]_i + sigma_self_i * q_i(phi_i) * z(phi_i)
    N = numel(phi);

    % Sender collective profile: b_j = sqrt(2) * cos(phi_j - delta)
    b_vec = sqrt(2) * cos(phi - cp_model.delta);

    % Network aggregated sender signal at receivers: u = W * b
    u_vec = W * b_vec;

    % Receiver profile: a(phi_i)
    a_vec = evaluate_receiver_profile_fourier(phi, cp_model.A, cp_model.m_values);

    % PRC: z(phi_i) = -sin(phi_i)
    z_vec = -sin(phi);

    % Interaction term: a_i * z_i * [W * b]_i
    coupling_term = a_vec .* z_vec .* u_vec;

    dphi = omega_rad_s + sigma .* coupling_term;

    % Optional self feedback: sigma_self * q_i(phi_i) * z(phi_i)
    if add_self_feedback && ~isempty(node_self_profiles)
        for i = 1:N
            sp = node_self_profiles{i};
            if ~isempty(sp)
                phi_w = mod(phi(i), 2*pi);
                q_val = interp1(sp.phi, sp.val, phi_w, 'linear', 'extrap');
                z_val = -sin(phi_w);
                dphi(i) = dphi(i) + sigma_self(i) * q_val * z_val;
            end
        end
    end
end

function a_vals = evaluate_receiver_profile_fourier(phi, A, m_values)
    % a(phi) = real( sum_{m} A_m * exp(1i * m * phi) )
    phi_col = phi(:);
    m_row = m_values(:).';
    basis = exp(1i * (phi_col * m_row));
    a_vals = real(basis * A(:));
end

% =========================================================================
% COMMON COUPLING FUNCTION GAMMA_0 CALCULATION
% =========================================================================

function gamma0 = compute_common_gamma_function(cp_model, M, opts)
% Compute the single global phase-interaction function Gamma_0(psi)
% defined as:
%   Gamma_0(psi) = (1 / 2*pi) * integral_0^{2*pi} z(theta) * a(theta) * b(theta - psi) dtheta
% where z(theta) = -sin(theta), b(phi) = sqrt(2)*cos(phi - delta).

    n_theta = opts.n_theta;
    n_psi = opts.n_psi;

    theta = linspace(0, 2*pi, n_theta).';
    psi = linspace(-pi, pi, n_psi).';

    z_theta = -sin(theta);
    a_theta = evaluate_receiver_profile_fourier(theta, cp_model.A, cp_model.m_values);

    delta = cp_model.delta;

    gamma0_raw = zeros(size(psi));
    for k = 1:n_psi
        psi_k = psi(k);
        % b(theta - psi_k) = sqrt(2) * cos(theta - psi_k - delta)
        b_shifted = sqrt(2) * cos(theta - psi_k - delta);
        integrand = z_theta .* a_theta .* b_shifted;
        gamma0_raw(k) = trapz(theta, integrand) / (2 * pi);
    end

    gamma0_curve = gamma0_raw;
    if opts.remove_gamma_bias
        gamma0_curve = gamma0_curve - mean(gamma0_curve);
    end

    % Fit 1st harmonic: c0 + c1*sin(psi) + c2*cos(psi)
    fit_harmonic = fit_first_harmonic(psi, gamma0_curve);

    if opts.use_first_harmonic
        gamma0_curve = fit_harmonic.bias + ...
            fit_harmonic.sin_coefficient * sin(psi) + ...
            fit_harmonic.cos_coefficient * cos(psi);
    end

    gamma0 = struct();
    gamma0.psi = psi;
    gamma0.gamma0 = gamma0_curve;
    gamma0.gamma0_raw = gamma0_raw;
    gamma0.fit = fit_harmonic;
    gamma0.removed_bias = opts.remove_gamma_bias;
    gamma0.used_first_harmonic = opts.use_first_harmonic;
end

function fit = fit_first_harmonic(psi, y_values)
    psi = psi(:);
    y = y_values(:);
    valid = isfinite(psi) & isfinite(y);
    psi = psi(valid);
    y = y(valid);

    X = [ones(size(psi)), sin(psi), cos(psi)];
    coeff = X \ y;
    y_fit = X * coeff;
    residual = y - y_fit;
    centered = y - mean(y);
    ss_res = sum(residual.^2);
    ss_tot = sum(centered.^2);

    fit = struct();
    fit.bias = coeff(1);
    fit.sin_coefficient = coeff(2);
    fit.cos_coefficient = coeff(3);
    fit.amplitude = hypot(coeff(2), coeff(3));
    fit.phase_rad = atan2(coeff(3), coeff(2));
    fit.rmse = sqrt(mean(residual.^2));
    fit.r2 = 1 - ss_res / max(ss_tot, eps);
end

% =========================================================================
% MODEL LOADING & NETWORK SELECTION
% =========================================================================

function [W_sim, W_full, network_info] = build_simulation_network(cp_model, opts)
    W_full = cp_model.W;
    N = size(W_full, 1);

    network_info = struct();
    network_info.mode_k = opts.network_mode_k;
    network_info.decomposition = opts.decomposition;
    network_info.force_zero_diagonal = opts.force_zero_diagonal;

    if isempty(opts.network_mode_k) || strcmpi(string(opts.network_mode_k), 'full')
        % Full identified network W
        W_sim = W_full;
        network_info.desc = 'Full Network Matrix W';
    else
        K = double(opts.network_mode_k);
        validateattributes(K, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'network_mode_k');

        modes = cp_model.network_modes;
        if isempty(modes) || ~isfield(modes, 'P') || ~isfield(modes, 'Q') || ~isfield(modes, 'd')
            % Compute decomposition on the fly if missing
            [U, S_mat, V] = svd(W_full);
            modes = struct('P', U, 'Q', V, 'd', diag(S_mat), 'method', 'svd');
        end

        K_use = min(K, numel(modes.d));
        P_K = modes.P(:, 1:K_use);
        Q_K = modes.Q(:, 1:K_use);
        d_K = modes.d(1:K_use);

        W_sim = P_K * diag(d_K) * Q_K.';
        network_info.desc = sprintf('Rank-%d %s Network Approximation', K_use, upper(modes.method));
    end

    if opts.force_zero_diagonal
        W_sim(1:N+1:end) = 0;
    end
end

function cp_model = obtain_cp_rank1_model(resolved_dir, M, opts)
    % 1. Direct struct passed by user
    if ~isempty(opts.cp_results) && isstruct(opts.cp_results)
        cp_model = extract_model_from_results(opts.cp_results, M);
        return;
    end

    % 2. Try loading from saved CSV / MAT files if force_recompute_cp is false
    if ~opts.force_recompute_cp
        loaded_model = try_load_saved_cp_model(resolved_dir, M, opts);
        if ~isempty(loaded_model)
            cp_model = loaded_model;
            return;
        end
    end

    % 3. Run global_joint_cp_rank1_profile_free_network_svd to identify the model
    fprintf('[INFO] Identifying Rank-1 Common Profile model via global_joint_cp_rank1_profile_free_network_svd...\n');
    res = global_joint_cp_rank1_profile_free_network_svd(resolved_dir, M, ...
        'NetworkDecomposition', opts.decomposition, ...
        'NumStarts', opts.NumStarts, ...
        'MaxIter', opts.MaxIter, ...
        'Tol', opts.Tol, ...
        'RandomSeed', opts.RandomSeed, ...
        'SaveOutputs', true);

    cp_model = extract_model_from_results(res, M);
end

function cp_model = try_load_saved_cp_model(resolved_dir, M, opts)
    cp_model = [];

    if strcmpi(opts.decomposition, 'sparse_pmd')
        out_subdir = 'global_joint_cp_rank1_profile_free_network_sparse_pmd';
        mat_filename = 'rank1_profile_free_network_sparse_pmd_results.mat';
    else
        out_subdir = 'global_joint_cp_rank1_profile_free_network_svd';
        mat_filename = 'rank1_profile_free_network_svd_results.mat';
    end

    dir_analysis = fullfile(resolved_dir, 'low_rank_analysis', sprintf('M%d', M), out_subdir);

    % Try MAT file
    mat_path = fullfile(dir_analysis, mat_filename);
    if exist(mat_path, 'file')
        try
            loaded = load(mat_path);
            cp_model = struct();
            cp_model.A = loaded.A(:);
            cp_model.B = loaded.B(:);
            cp_model.delta = loaded.delta;
            cp_model.W = loaded.W;
            cp_model.agent_ids = loaded.agent_ids(:).';
            cp_model.m_values = (-M:M).';
            cp_model.network_modes = struct('P', loaded.P, 'Q', loaded.Q, 'd', loaded.d, 'method', loaded.method);
            fprintf('[INFO] Loaded CP Rank-1 model from MAT: %s\n', mat_path);
            return;
        catch
            cp_model = [];
        end
    end

    % Try CSV files
    fourier_csv = fullfile(dir_analysis, 'fourier_coefficients_A.csv');
    delta_csv = fullfile(dir_analysis, 'sender_phase_shift_delta.csv');
    w_csv = fullfile(dir_analysis, 'network_coupling_matrix_W.csv');

    if exist(fourier_csv, 'file') && exist(delta_csv, 'file') && exist(w_csv, 'file')
        try
            t_fourier = readtable(fourier_csv);
            t_delta = readtable(delta_csv);
            t_w = readtable(w_csv);

            A = t_fourier.Re_A + 1i * t_fourier.Im_A;
            delta = t_delta.delta_rad(1);

            % Extract agent IDs and W matrix from network_coupling_matrix_W.csv
            % Variable names: receiver_agent, sender_agent_8, sender_agent_9, ...
            var_names = t_w.Properties.VariableNames;
            sender_cols = var_names(startsWith(var_names, 'sender_agent_'));
            agent_ids = zeros(1, numel(sender_cols));
            for k = 1:numel(sender_cols)
                agent_ids(k) = str2double(regexprep(sender_cols{k}, '^sender_agent_', ''));
            end

            W_mat = table2array(t_w(:, sender_cols));

            % Check SVD modes summary CSV
            svd_csv = fullfile(dir_analysis, 'network_svd_modes_summary.csv');
            [U, S_mat, V] = svd(W_mat);
            modes = struct('P', U, 'Q', V, 'd', diag(S_mat), 'method', 'svd');

            cp_model = struct();
            cp_model.A = A;
            cp_model.B = [];
            cp_model.delta = delta;
            cp_model.W = W_mat;
            cp_model.agent_ids = agent_ids;
            cp_model.m_values = (-M:M).';
            cp_model.network_modes = modes;
            fprintf('[INFO] Loaded CP Rank-1 model from CSV files in %s\n', dir_analysis);
            return;
        catch ME
            warning('Failed reading CSV CP model: %s', ME.message);
            cp_model = [];
        end
    end
end

function cp_model = extract_model_from_results(res, M)
    cp_model = struct();
    if isfield(res, 'fit_R1')
        cp_model.A = res.fit_R1.A(:, 1);
        if isfield(res.fit_R1, 'B')
            cp_model.B = res.fit_R1.B(:, 1);
        else
            cp_model.B = [];
        end
        cp_model.delta = res.fit_R1.delta(1);
    elseif isfield(res, 'A') && isfield(res, 'delta')
        cp_model.A = res.A;
        cp_model.delta = res.delta;
    else
        error('Invalid CP results structure: missing A or delta fields.');
    end

    cp_model.W = res.W;
    cp_model.agent_ids = res.agent_ids(:).';
    cp_model.m_values = (-M:M).';

    if isfield(res, 'network_modes')
        cp_model.network_modes = res.network_modes;
    elseif isfield(res, 'svd')
        cp_model.network_modes = struct('P', res.svd.U, 'Q', res.svd.V, 'd', res.svd.singular_values, 'method', 'svd');
    else
        [U, S_mat, V] = svd(res.W);
        cp_model.network_modes = struct('P', U, 'Q', V, 'd', diag(S_mat), 'method', 'svd');
    end
end

function node_self_profiles = load_agent_self_profiles(resolved_dir, M, node_ids, custom_dir)
    N = numel(node_ids);
    node_self_profiles = cell(N, 1);

    self_dir = custom_dir;
    if isempty(self_dir)
        candidate_dirs = {
            fullfile(resolved_dir, 'low_rank_analysis', sprintf('M%d', M), 'agent_self_profiles'), ...
            fullfile(resolved_dir, 'low_rank_analysis', sprintf('M%d', M), 'global_joint_cp_rank1', 'agent_self_profiles'), ...
            fullfile('EstimateL', 'Round', 'low_rank_analysis', 'M10', 'agent_self_profiles')
        };
        for c = 1:numel(candidate_dirs)
            if exist(candidate_dirs{c}, 'dir')
                self_dir = candidate_dirs{c};
                break;
            end
        end
    end

    if isempty(self_dir) || ~exist(self_dir, 'dir')
        return;
    end

    for i = 1:N
        aid = node_ids(i);
        csv_path = fullfile(self_dir, sprintf('agent%d_self_profile_data.csv', aid));
        if isfile(csv_path)
            t = readtable(csv_path);
            node_self_profiles{i} = struct('phi', t.phi, 'val', t.mean_self_profile);
            fprintf('[INFO] Loaded mean self-profile for Agent %d\n', aid);
        end
    end
end

% =========================================================================
% PLOTTING FUNCTIONS
% =========================================================================

function fig = plot_relative_phase_trajectories(time, relative_phase, node_ids, reference_agent_id, agent_display_offset, net_title)
    if nargin < 5 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end
    if nargin < 6 || isempty(net_title)
        net_title = 'CP Rank-1 Model';
    end

    fig = figure('Color', 'w', 'Name', sprintf('Relative Phase Dynamics (%s)', net_title));
    ax = axes('Parent', fig);
    hold(ax, 'on');

    colors = lines(numel(node_ids));
    for k = 1:numel(node_ids)
        y_val = relative_phase(:, k);

        % Insert NaNs across phase wrapping boundaries (> pi gap)
        for j = 2:numel(y_val)
            if isnan(y_val(j)) || isnan(y_val(j-1))
                continue;
            end
            if abs(y_val(j) - y_val(j-1)) > pi
                y_val(j) = NaN;
            end
        end

        plot(ax, time, y_val, 'LineWidth', 1.6, 'Color', colors(k, :), ...
            'DisplayName', sprintf('ID %d', node_ids(k) + agent_display_offset));
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    ylim(ax, [-pi, pi]);
    yticks(ax, [-pi, -pi/2, 0, pi/2, pi]);
    yticklabels(ax, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
    xlabel(ax, 'Time (s)', 'FontSize', 11);
    ylabel(ax, sprintf('$$\\phi_j - \\phi_{%d}$$ (rad)', reference_agent_id + agent_display_offset), ...
        'Interpreter', 'latex', 'FontSize', 12);
    title(ax, sprintf('Relative Phase Dynamics [%s]', net_title), 'FontSize', 12);
    legend(ax, 'Location', 'eastoutside');

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_absolute_phases(time, phase, node_ids, agent_display_offset)
    if nargin < 4 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end

    fig = figure('Color', 'w', 'Name', 'Absolute Phase Dynamics');
    ax = axes('Parent', fig);
    hold(ax, 'on');

    colors = lines(numel(node_ids));
    for k = 1:numel(node_ids)
        plot(ax, time, phase(:, k), 'LineWidth', 1.2, 'Color', colors(k, :), ...
            'DisplayName', sprintf('ID %d: \\phi', node_ids(k) + agent_display_offset));
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    xlabel(ax, 'Time (s)', 'FontSize', 11);
    ylabel(ax, '$$\\phi_j$$ (rad)', 'Interpreter', 'latex', 'FontSize', 12);
    title(ax, 'Absolute Phase Dynamics', 'FontSize', 12);
    legend(ax, 'Location', 'best');

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_mode_order_parameters(time, mode_analysis, net_title)
    if nargin < 3 || isempty(net_title)
        net_title = 'CP Rank-1 Model';
    end

    fig = figure('Color', 'w', 'Name', sprintf('Weighted Order Parameter |Z_l(t)| (%s)', net_title));
    ax = axes('Parent', fig);
    hold(ax, 'on');

    K = mode_analysis.num_modes;
    d = mode_analysis.d;
    method = upper(mode_analysis.method);
    colors = lines(K);
    line_handles = gobjects(K, 1);
    mode_legend = cell(K, 1);

    for m = 1:K
        line_handles(m) = plot(ax, time, mode_analysis.Z_abs(:, m), ...
            'LineWidth', 1.6, 'Color', colors(m, :));
        if strcmpi(method, 'SPARSE_PMD')
            mode_legend{m} = sprintf('Sender Mode %d (d_{%d} = %.3f)', m, m, d(m));
        else
            mode_legend{m} = sprintf('Sender Mode %d (\\sigma_{%d} = %.3f)', m, m, d(m));
        end
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    xlabel(ax, 'Time (s)', 'FontSize', 11);
    ylabel(ax, '$$|Z_l(t)| = \left|\sum_j v_{jl} e^{i \phi_j(t)}\right|$$', ...
        'Interpreter', 'latex', 'FontSize', 12);
    title(ax, sprintf('Weighted Order Parameter |Z_l(t)| [%s]', net_title), 'FontSize', 12);
    legend(ax, line_handles, mode_legend, 'Location', 'eastoutside', 'FontSize', 10);

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_agent_input_amplitudes(time, mode_analysis, node_ids, agent_display_offset, net_title)
    if nargin < 4 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end
    if nargin < 5 || isempty(net_title)
        net_title = 'CP Rank-1 Model';
    end

    fig = figure('Color', 'w', 'Name', sprintf('Agent Weighted Input Amplitude |G_i(t)| (%s)', net_title));
    ax = axes('Parent', fig);
    hold(ax, 'on');

    N = numel(node_ids);
    colors = lines(N);
    line_handles = gobjects(N, 1);
    agent_legend = cell(N, 1);

    for i = 1:N
        line_handles(i) = plot(ax, time, mode_analysis.G_abs(:, i), ...
            'LineWidth', 1.5, 'Color', colors(i, :));
        agent_legend{i} = sprintf('ID %d', node_ids(i) + agent_display_offset);
    end

    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [time(1), time(end)]);
    xlabel(ax, 'Time (s)', 'FontSize', 11);
    ylabel(ax, '$$|G_i(t)| = \left|\sum_{\ell}\sigma_{\ell}u_{i\ell}Z_{\ell}(t)\right|$$', ...
        'Interpreter', 'latex', 'FontSize', 12);
    title(ax, sprintf('Agent Total Input Amplitude |G_i(t)| [%s]', net_title), 'FontSize', 12);
    legend(ax, line_handles, agent_legend, 'Location', 'eastoutside', 'FontSize', 10);

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function mode_analysis = compute_mode_amplitudes_time_series(phase, cp_model)
    mode_analysis = struct();
    mode_analysis.available = false;

    modes = cp_model.network_modes;
    if isempty(modes) || ~isfield(modes, 'P') || ~isfield(modes, 'Q') || ~isfield(modes, 'd')
        if isfield(cp_model, 'W') && ~isempty(cp_model.W)
            [U, S_mat, V] = svd(cp_model.W);
            modes = struct('P', U, 'Q', V, 'd', diag(S_mat), 'method', 'svd');
        else
            return;
        end
    end

    U = modes.P;
    V = modes.Q;
    d = modes.d(:);
    K = numel(d);

    % Complex phasors: exp(1i * phase) (T x N)
    phasors = exp(1i * phase);

    % Weighted order parameter: Z_l(t) = sum_j v_{jl} * exp(1i * phi_j(t))
    Z_complex = phasors * V;     % T x K
    Z_abs = abs(Z_complex);     % T x K

    % Receiver total mode input amplitude: G_i(t) = sum_l sigma_l * u_{il} * Z_l(t)
    G_complex = (Z_complex .* d.') * U.'; % T x N
    G_abs = abs(G_complex);               % T x N

    % Real collective sender signal: X_real (T x K)
    B_mat = sqrt(2) * cos(phase - cp_model.delta);
    X_real = B_mat * V;

    mode_analysis.available = true;
    mode_analysis.method = modes.method;
    mode_analysis.num_modes = K;
    mode_analysis.d = d;
    mode_analysis.U = U;
    mode_analysis.V = V;
    mode_analysis.Z_complex = Z_complex;
    mode_analysis.Z_abs = Z_abs;
    mode_analysis.G_complex = G_complex;
    mode_analysis.G_abs = G_abs;
    mode_analysis.X_real = X_real;
end

function fig = plot_common_profile_and_gamma(cp_model, gamma0, W, node_ids, agent_display_offset)
    if nargin < 5 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end

    fig = figure('Color', 'w', 'Position', [100, 100, 1000, 400], ...
        'Name', 'Rank-1 Common Profiles and Coupling Function');
    tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    phi_grid = linspace(0, 2*pi, 400).';
    a_grid = evaluate_receiver_profile_fourier(phi_grid, cp_model.A, cp_model.m_values);
    b_grid = sqrt(2) * cos(phi_grid - cp_model.delta);

    % Subplot 1: Target receiver profile a(phi)
    ax1 = nexttile;
    hold(ax1, 'on');
    plot(ax1, phi_grid, a_grid, 'b-', 'LineWidth', 1.8, 'DisplayName', 'a(\phi)');
    grid(ax1, 'on'); box(ax1, 'on');
    xlim(ax1, [0, 2*pi]);
    xticks(ax1, [0, pi/2, pi, 3*pi/2, 2*pi]);
    xticklabels(ax1, {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
    xlabel(ax1, '\phi (rad)');
    ylabel(ax1, 'Receiver Profile a(\phi)');
    title(ax1, 'Common Target Profile a(\phi)');

    % Subplot 2: Sender profile b(phi)
    ax2 = nexttile;
    hold(ax2, 'on');
    plot(ax2, phi_grid, b_grid, 'r-', 'LineWidth', 1.8, ...
        'DisplayName', sprintf('\\surd2 cos(\\phi - %.2f\\pi)', cp_model.delta/pi));
    grid(ax2, 'on'); box(ax2, 'on');
    xlim(ax2, [0, 2*pi]);
    xticks(ax2, [0, pi/2, pi, 3*pi/2, 2*pi]);
    xticklabels(ax2, {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
    xlabel(ax2, '\phi (rad)');
    ylabel(ax2, 'Sender Profile b(\phi)');
    title(ax2, sprintf('Common Sender b(\\phi) [\\delta = %.2f\\pi]', cp_model.delta/pi));
    legend(ax2, 'Location', 'northeast');

    % Subplot 3: Common coupling function Gamma_0(psi)
    ax3 = nexttile;
    hold(ax3, 'on');
    plot(ax3, gamma0.psi, gamma0.gamma0, 'k-', 'LineWidth', 2.0, 'DisplayName', '\Gamma_0(\psi)');
    if isfield(gamma0, 'fit') && ~isempty(gamma0.fit)
        psi_dense = linspace(-pi, pi, 200).';
        y_fit = gamma0.fit.bias + gamma0.fit.sin_coefficient*sin(psi_dense) + gamma0.fit.cos_coefficient*cos(psi_dense);
        plot(ax3, psi_dense, y_fit, 'm--', 'LineWidth', 1.2, ...
            'DisplayName', sprintf('1st Harm (R^2=%.3f)', gamma0.fit.r2));
    end
    grid(ax3, 'on'); box(ax3, 'on');
    xlim(ax3, [-pi, pi]);
    xticks(ax3, [-pi, -pi/2, 0, pi/2, pi]);
    xticklabels(ax3, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
    xlabel(ax3, '\psi = \phi_i - \phi_j (rad)');
    ylabel(ax3, '\Gamma_0(\psi)');
    title(ax3, 'Common Coupling Function \Gamma_0(\psi)');
    legend(ax3, 'Location', 'best');

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_network_matrix(W_sim, W_full, node_ids, agent_display_offset, net_title)
    if nargin < 4 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end
    if nargin < 5 || isempty(net_title)
        net_title = 'Network Matrix';
    end

    fig = figure('Color', 'w', 'Position', [100, 100, 750, 350], ...
        'Name', sprintf('Network Coupling Matrix W (%s)', net_title));
    tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    labels = arrayfun(@(id) sprintf('ID %d', id + agent_display_offset), node_ids, 'UniformOutput', false);

    max_val = max([abs(W_full(:)); abs(W_sim(:)); eps]);

    % Subplot 1: Full Identified W
    ax1 = nexttile;
    imagesc(ax1, W_full, [-max_val, max_val]);
    colormap(ax1, balance_colormap());
    colorbar(ax1);
    axis(ax1, 'square');
    set(ax1, 'XTick', 1:numel(node_ids), 'XTickLabel', labels, ...
             'YTick', 1:numel(node_ids), 'YTickLabel', labels);
    title(ax1, 'Full Identified Network W');
    xlabel(ax1, 'Sender j');
    ylabel(ax1, 'Receiver i');

    % Subplot 2: Simulation W
    ax2 = nexttile;
    imagesc(ax2, W_sim, [-max_val, max_val]);
    colormap(ax2, balance_colormap());
    colorbar(ax2);
    axis(ax2, 'square');
    set(ax2, 'XTick', 1:numel(node_ids), 'XTickLabel', labels, ...
             'YTick', 1:numel(node_ids), 'YTickLabel', labels);
    title(ax2, sprintf('Simulated Matrix (%s)', net_title));
    xlabel(ax2, 'Sender j');
    ylabel(ax2, 'Receiver i');

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function fig = plot_pairwise_gamma_functions(W, gamma0, node_ids, agent_display_offset)
    if nargin < 4 || isempty(agent_display_offset)
        agent_display_offset = 0;
    end

    N = numel(node_ids);
    n_pairs = N * (N - 1) / 2;
    fig = figure('Color', 'w', 'Position', [100, 100, 850, min(1000, 220*n_pairs)], ...
        'Name', 'Pairwise Gamma Functions: \Gamma_{ij}(\psi) = W_{ij} * \Gamma_0(\psi)');
    tiledlayout(fig, n_pairs, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:N
        for j = (i+1):N
            id_i = node_ids(i) + agent_display_offset;
            id_j = node_ids(j) + agent_display_offset;

            % Left tile: i <- j (W(i, j))
            ax1 = nexttile;
            hold(ax1, 'on');
            gamma_ij = W(i, j) * gamma0.gamma0;
            plot(ax1, gamma0.psi, gamma_ij, 'LineWidth', 1.5);
            grid(ax1, 'on'); box(ax1, 'on');
            xlim(ax1, [-pi, pi]);
            xticks(ax1, [-pi, -pi/2, 0, pi/2, pi]);
            xticklabels(ax1, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
            xlabel(ax1, '\psi');
            ylabel(ax1, sprintf('\\Gamma_{%d\\leftarrow%d}', id_i, id_j));
            title(ax1, sprintf('W_{%d\\leftarrow%d} = %.4f', id_i, id_j, W(i, j)));

            % Right tile: j <- i (W(j, i))
            ax2 = nexttile;
            hold(ax2, 'on');
            gamma_ji = W(j, i) * gamma0.gamma0;
            plot(ax2, gamma0.psi, gamma_ji, 'LineWidth', 1.5);
            grid(ax2, 'on'); box(ax2, 'on');
            xlim(ax2, [-pi, pi]);
            xticks(ax2, [-pi, -pi/2, 0, pi/2, pi]);
            xticklabels(ax2, {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'});
            xlabel(ax2, '\psi');
            ylabel(ax2, sprintf('\\Gamma_{%d\\leftarrow%d}', id_j, id_i));
            title(ax2, sprintf('W_{%d\\leftarrow%d} = %.4f', id_j, id_i, W(j, i)));
        end
    end

    if exist('tuneFigure', 'file') == 2 || exist('tuneFigure', 'builtin')
        figure(fig);
        tuneFigure();
    end
end

function cmap = balance_colormap()
    N = 256;
    r = [linspace(0.1, 0.95, N/2), linspace(0.95, 0.8, N/2)].';
    g = [linspace(0.2, 0.95, N/2), linspace(0.95, 0.2, N/2)].';
    b = [linspace(0.8, 0.95, N/2), linspace(0.95, 0.1, N/2)].';
    cmap = [r, g, b];
end

% =========================================================================
% OPTIONS PARSER & HELPERS
% =========================================================================

function opts = parse_options(defaults, varargin)
    % Map aliases to canonical names: 'gain', 'coupling_gain', 'control_gain' -> 'sigma'
    %                               'self_gain' -> 'sigma_self'
    args = varargin;
    for k = 1:2:numel(args)
        if ischar(args{k}) || isstring(args{k})
            key = lower(char(args{k}));
            if ismember(key, {'gain', 'coupling_gain', 'control_gain'})
                args{k} = 'sigma';
            elseif ismember(key, {'self_gain', 'self_feedback_gain'})
                args{k} = 'sigma_self';
            end
        end
    end

    p = inputParser;

    % Dynamics & Control Gain parameters
    addParameter(p, 'sigma', defaults.sigma, @(x) isnumeric(x) && all(isfinite(x)));
    addParameter(p, 'sigma_self', defaults.sigma_self, @(x) isempty(x) || (isnumeric(x) && all(isfinite(x))));
    addParameter(p, 'omega_rad_s', defaults.omega_rad_s, @(x) isnumeric(x) && isfinite(x));
    addParameter(p, 'simulation_duration_sec', defaults.simulation_duration_sec, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'simulation_dt', defaults.simulation_dt, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(p, 'initial_phases', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'reference_agent_id', defaults.reference_agent_id, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isfinite(x)));

    % System formulation flags
    addParameter(p, 'use_original_system', defaults.use_original_system, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'remove_gamma_bias', defaults.remove_gamma_bias, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'use_first_harmonic', defaults.use_first_harmonic, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'subtract_self_profile', defaults.subtract_self_profile, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'add_self_feedback', defaults.add_self_feedback, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'self_profile_dir', '', @(x) ischar(x) || isstring(x));

    % Network approximation options
    addParameter(p, 'network_mode_k', defaults.network_mode_k, @(x) isempty(x) || ischar(x) || isstring(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'decomposition', defaults.decomposition, @(x) ischar(x) || isstring(x));
    addParameter(p, 'force_zero_diagonal', defaults.force_zero_diagonal, @(x) islogical(x) || isnumeric(x));

    % CP Model estimation options (if model re-run needed)
    addParameter(p, 'cp_results', [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'force_recompute_cp', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'NumStarts', 20, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'MaxIter', 1000, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Tol', 1e-10, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'RandomSeed', 0, @(x) isnumeric(x) && isscalar(x));

    % Gamma grid resolution
    addParameter(p, 'n_theta', 2001, @(x) isnumeric(x) && isscalar(x) && x >= 3);
    addParameter(p, 'n_psi', 501, @(x) isnumeric(x) && isscalar(x) && x >= 3);

    % Plotting and export options
    addParameter(p, 'plot_relative_phase', defaults.plot_relative_phase, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_absolute_phases', defaults.plot_absolute_phases, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_mode_amplitudes', defaults.plot_mode_amplitudes, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_agent_input_amp', defaults.plot_agent_input_amp, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_gamma', defaults.plot_gamma, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_network_weights', defaults.plot_network_weights, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'plot_all_gamma', defaults.plot_all_gamma, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'agent_display_offset', defaults.agent_display_offset, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'save_output', defaults.save_output, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));

    parse(p, args{:});
    opts = p.Results;

    opts.decomposition = char(opts.decomposition);
    opts.self_profile_dir = char(opts.self_profile_dir);
    opts.output_dir = char(opts.output_dir);
    opts.use_original_system = logical(opts.use_original_system);
    opts.remove_gamma_bias = logical(opts.remove_gamma_bias);
    opts.use_first_harmonic = logical(opts.use_first_harmonic);
    opts.subtract_self_profile = logical(opts.subtract_self_profile);
    opts.add_self_feedback = logical(opts.add_self_feedback);
    opts.force_zero_diagonal = logical(opts.force_zero_diagonal);
    opts.force_recompute_cp = logical(opts.force_recompute_cp);
    opts.plot_relative_phase = logical(opts.plot_relative_phase);
    opts.plot_absolute_phases = logical(opts.plot_absolute_phases);
    opts.plot_mode_amplitudes = logical(opts.plot_mode_amplitudes);
    opts.plot_agent_input_amp = logical(opts.plot_agent_input_amp);
    opts.plot_gamma = logical(opts.plot_gamma);
    opts.plot_network_weights = logical(opts.plot_network_weights);
    opts.plot_all_gamma = logical(opts.plot_all_gamma);
    opts.save_output = logical(opts.save_output);
end

function phase_wrapped = wrap_to_pi(phase)
    phase_wrapped = atan2(sin(phase), cos(phase));
end

function val = ternary(cond, a, b)
    if cond
        val = a;
    else
        val = b;
    end
end

function resolved_dir = resolve_dataset_directory(d)
    if ischar(d) || isstring(d)
        d_str = char(d);
    else
        resolved_dir = d;
        return;
    end

    if exist(d_str, 'dir')
        resolved_dir = d_str;
        return;
    end

    base_root = fileparts(fileparts(mfilename('fullpath')));
    estimate_dir = fileparts(mfilename('fullpath'));
    candidates = {
        fullfile(estimate_dir, d_str), ...
        fullfile(base_root, 'EstimateL', d_str), ...
        fullfile('EstimateL', d_str), ...
        fullfile(base_root, d_str)
    };
    for c = 1:numel(candidates)
        if exist(candidates{c}, 'dir')
            resolved_dir = candidates{c};
            return;
        end
    end
    resolved_dir = d_str;
end

function export = save_simulation_outputs(out, opts)
    output_dir = opts.output_dir;
    if isempty(output_dir)
        output_dir = fullfile(out.target_dir, 'relative_phase_cp_rank1_exports');
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    csv_path = fullfile(output_dir, 'relative_phase_cp_rank1_sim.csv');
    final_relative = out.relative_phase(end, :);
    rel_table = array2table(final_relative, ...
        'VariableNames', arrayfun(@(id) sprintf('phi_%d_minus_ref', id), out.node_ids, 'UniformOutput', false));
    writetable(rel_table, csv_path);

    % Also save mode order parameter amplitudes if available
    if isfield(out, 'mode_analysis') && out.mode_analysis.available
        mode_csv_path = fullfile(output_dir, 'mode_order_parameters.csv');
        Z_table = array2table([out.time, out.mode_analysis.Z_abs], ...
            'VariableNames', [{'time_sec'}, arrayfun(@(m) sprintf('mode_%d_abs', m), 1:out.mode_analysis.num_modes, 'UniformOutput', false)]);
        writetable(Z_table, mode_csv_path);
    end

    mat_path = fullfile(output_dir, 'relative_phase_cp_rank1_simulation_results.mat');
    save(mat_path, 'out', '-v7.3');

    export = struct('output_dir', output_dir, 'csv_path', csv_path, 'mat_path', mat_path);
    fprintf('[INFO] Saved simulation results to %s\n', output_dir);
end

function tf = is_option_name(x)
    if ~ischar(x) && ~isstring(x)
        tf = false;
        return;
    end
    known_opts = { ...
        'sigma', 'gain', 'coupling_gain', 'control_gain', ...
        'sigma_self', 'self_gain', 'self_feedback_gain', ...
        'omega_rad_s', 'simulation_duration_sec', 'simulation_dt', ...
        'initial_phases', 'reference_agent_id', 'use_original_system', ...
        'remove_gamma_bias', 'use_first_harmonic', 'subtract_self_profile', ...
        'add_self_feedback', 'self_profile_dir', 'network_mode_k', ...
        'decomposition', 'force_zero_diagonal', 'cp_results', ...
        'force_recompute_cp', 'NumStarts', 'MaxIter', 'Tol', 'RandomSeed', ...
        'n_theta', 'n_psi', 'plot_relative_phase', 'plot_absolute_phases', ...
        'plot_mode_amplitudes', 'plot_agent_input_amp', ...
        'plot_gamma', 'plot_network_weights', 'plot_all_gamma', ...
        'agent_display_offset', 'save_output', 'output_dir' ...
    };
    tf = any(strcmpi(char(x), known_opts));
end

