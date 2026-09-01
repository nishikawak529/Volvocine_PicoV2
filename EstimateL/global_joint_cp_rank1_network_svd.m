function results = global_joint_cp_rank1_network_svd(round_dir, M, varargin)
%GLOBAL_JOINT_CP_RANK1_NETWORK_SVD Wrapper for Rank-1 Joint CP Model with Sinusoidal Sender Profile & Network SVD Analysis.
%
% Usage:
%   results = global_joint_cp_rank1_network_svd();
%   results = global_joint_cp_rank1_network_svd(fullfile('EstimateL','Round6'), 10);
%
% Output:
%   - Saves max 4 consolidated PNG files:
%       1. rank1_profile_network_svd_summary.png
%       2. rank1_profile_network_svd_modes.png
%       3. rank1_profile_network_svd_approximation.png
%       4. rank1_profile_collective_signals.png (if phase time series available)
%   - Saves 1 lightweight MAT file: rank1_profile_network_svd_results.mat (SaveCompactMat=true)

    if nargin < 1 || isempty(round_dir)
        round_dir = fullfile('EstimateL', 'Round6');
    end
    if nargin < 2 || isempty(M)
        M = 10;
    end

    results = global_joint_cp_rank_sweep_sinusoidal_sender_rank1_network( ...
        round_dir, M, 'Mode', 'R1_only', 'SaveCompactMat', true, varargin{:});
end
