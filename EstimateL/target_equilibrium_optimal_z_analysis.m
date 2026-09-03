function result = target_equilibrium_optimal_z_analysis(varargin)
%TARGET_EQUILIBRIUM_OPTIMAL_Z_ANALYSIS Optimize a PRC at a target equilibrium.
%
% result = target_equilibrium_optimal_z_analysis( ...
%     'PsiTarget', 0, 'Nu', 0, 'Power', pi, ...
%     'ManualTau', 0, 'NGrid', 4096, 'Interactive', true)
%
% The one-way phase-difference model used here is
%
%   psi = phi_i - Phi_i,
%   F_z(psi) = nu + Gamma_z(psi),
%   Gamma_z(psi) = <z(theta), a(theta) cos(theta-psi-delta)>.
%
% Phi_i is the collective-mode phase before applying the sender shift
% delta.  The actual sensor oscillation is cos(Phi_i-delta), so its phase
% difference is psi_phys = psi + delta.  Consequently, psi_d = 0 is in
% phase with Phi_i, whereas psi_d = -delta is in phase with the sensor.
%
% This is a fixed collective mode driving one receiver.  It is not a
% symmetric pair of identical oscillators; no odd-part construction such
% as Gamma(psi)-Gamma(-psi) is used.
%
% The optimization does not maximize or minimize Gamma at psi_d.  It first
% enforces the equilibrium condition Gamma(psi_d) = -nu exactly and then
% maximizes the negative local slope kappa = -Gamma'(psi_d), subject to
% integral_0^(2*pi) z(theta)^2 dtheta = Power.

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'PsiTarget', 0, @isFiniteRealScalar);
    addParameter(parser, 'Nu', 0, @isFiniteRealScalar);
    addParameter(parser, 'Power', pi, @isNonnegativeFiniteRealScalar);
    addParameter(parser, 'ManualTau', 0, @isFiniteRealScalar);
    addParameter(parser, 'NGrid', 4096, @isValidGridSize);
    addParameter(parser, 'Interactive', true, @isLogicalScalar);
    parse(parser, varargin{:});
    options = parser.Results;
    options.Interactive = logical(options.Interactive);

    if options.Interactive && abs(options.Nu) > 0.8
        error('target_equilibrium_optimal_z_analysis:InteractiveNuRange', ...
            ['Interactive mode uses the required nu slider range [-0.8, 0.8]. ' ...
             'Use Interactive=false for a value outside this range.']);
    end

    % Resolve inputs relative to this function, never relative to pwd.
    baseDir = fileparts(mfilename('fullpath'));
    inputDir = fullfile(baseDir, 'Round', 'low_rank_analysis', 'M10', ...
        'global_joint_cp_rank1_profile_free_network_svd');
    profilePath = fullfile(inputDir, 'target_receiver_profile_a_phi.csv');
    deltaPath = fullfile(inputDir, 'sender_phase_shift_delta.csv');

    tolerances = defaultTolerances();
    [phi, a, delta, inputDiagnostics] = loadPeriodicInputs( ...
        profilePath, deltaPath, options.NGrid, tolerances);

    psiTarget = wrapAngle(options.PsiTarget);
    manualTau = wrapAngle(options.ManualTau);
    analysis = computeAnalysis(phi, a, delta, psiTarget, options.Nu, ...
        options.Power, manualTau, tolerances);
    selfChecks = runInternalChecks(phi, a, delta, tolerances);
    result = assembleResult(analysis, inputDiagnostics, selfChecks, tolerances);

    if options.Interactive
        result.figure = createInteractiveFigure(phi, a, delta, analysis, ...
            inputDiagnostics, selfChecks, tolerances, baseDir);
    else
        result.figure = createStaticFigure(analysis, selfChecks);
        result.figure.UserData = struct('result', resultWithoutFigure(result));
    end
end

% =========================================================================
% Input and periodic interpolation
% =========================================================================

function [phiGrid, aGrid, delta, diagnostics] = loadPeriodicInputs( ...
        profilePath, deltaPath, nGrid, tolerances)
    if ~isfile(profilePath)
        error('target_equilibrium_optimal_z_analysis:MissingProfile', ...
            'Receiver profile CSV not found: %s', profilePath);
    end
    if ~isfile(deltaPath)
        error('target_equilibrium_optimal_z_analysis:MissingDelta', ...
            'Sender phase-shift CSV not found: %s', deltaPath);
    end

    profileTable = readtable(profilePath);
    profileNames = profileTable.Properties.VariableNames;
    requiredProfileNames = {'phi', 'a_phi'};
    if ~all(ismember(requiredProfileNames, profileNames))
        error('target_equilibrium_optimal_z_analysis:ProfileColumns', ...
            'Profile CSV must contain the exact columns phi and a_phi.');
    end

    phiRaw = profileTable.phi;
    aRaw = profileTable.a_phi;
    if ~isnumeric(phiRaw) || ~isnumeric(aRaw) || ...
            ~isvector(phiRaw) || ~isvector(aRaw) || ...
            numel(phiRaw) ~= numel(aRaw) || numel(phiRaw) < 4 || ...
            any(~isfinite(phiRaw)) || any(~isfinite(aRaw)) || ...
            ~isreal(phiRaw) || ~isreal(aRaw)
        error('target_equilibrium_optimal_z_analysis:ProfileData', ...
            'Columns phi and a_phi must be finite real numeric vectors of equal length.');
    end
    phiRaw = phiRaw(:);
    aRaw = aRaw(:);

    deltaTable = readtable(deltaPath);
    deltaNames = deltaTable.Properties.VariableNames;
    requiredDeltaNames = {'delta_rad', 'delta_deg', 'delta_over_pi'};
    if ~all(ismember(requiredDeltaNames, deltaNames))
        error('target_equilibrium_optimal_z_analysis:DeltaColumns', ...
            ['Phase-shift CSV must contain the exact columns delta_rad, ' ...
             'delta_deg, and delta_over_pi.']);
    end
    if height(deltaTable) ~= 1 || ~isnumeric(deltaTable.delta_rad) || ...
            any(~isfinite([deltaTable.delta_rad, deltaTable.delta_deg, ...
                           deltaTable.delta_over_pi]))
        error('target_equilibrium_optimal_z_analysis:DeltaData', ...
            'The phase-shift CSV must contain exactly one finite numeric data row.');
    end
    delta = double(deltaTable.delta_rad(1));
    deltaDegreeResidual = abs(deltaTable.delta_deg(1) - delta * 180 / pi);
    deltaPiResidual = abs(deltaTable.delta_over_pi(1) - delta / pi);
    if deltaDegreeResidual > 1e-9 || deltaPiResidual > 1e-12
        error('target_equilibrium_optimal_z_analysis:DeltaUnits', ...
            'The three delta columns are not mutually consistent.');
    end

    % Wrap samples to [0,2*pi) without rotating the phase origin.  A
    % tolerance is essential: the supplied last phase is 2*pi plus a few
    % ulps, rather than being bitwise equal to 2*pi.
    twoPi = 2 * pi;
    phaseMergeTolerance = max(tolerances.angle, ...
        100 * eps(max(1, max(abs(phiRaw)))));
    phiWrapped = mod(phiRaw, twoPi);
    nearPeriodicOrigin = abs(phiWrapped) <= phaseMergeTolerance | ...
        abs(phiWrapped - twoPi) <= phaseMergeTolerance;
    phiWrapped(nearPeriodicOrigin) = 0;

    [phiWrapped, order] = sort(phiWrapped);
    aSorted = aRaw(order);
    [phiUnique, ~, groupIndex] = unique(phiWrapped, 'sorted');
    aUnique = accumarray(groupIndex, aSorted, [], @mean);
    duplicateCount = numel(phiRaw) - numel(phiUnique);
    if numel(phiUnique) < 4 || any(diff(phiUnique) <= 0)
        error('target_equilibrium_optimal_z_analysis:PhaseGrid', ...
            'The periodic profile must contain at least four distinct phase samples.');
    end

    % Add one periodic neighbor on either side before interpolation.  The
    % target grid deliberately excludes 2*pi, so no endpoint is integrated
    % twice.  Linear interpolation matches the repository's sampled-profile
    % convention and introduces no additional toolbox dependency.
    phiExtended = [phiUnique(end) - twoPi; phiUnique; phiUnique(1) + twoPi];
    aExtended = [aUnique(end); aUnique; aUnique(1)];
    phiGrid = (0:(nGrid - 1)).' * (twoPi / nGrid);
    aGrid = interp1(phiExtended, aExtended, phiGrid, 'linear');
    if any(~isfinite(aGrid))
        error('target_equilibrium_optimal_z_analysis:Interpolation', ...
            'Periodic interpolation produced nonfinite values.');
    end

    diagnostics = struct();
    diagnostics.profilePath = profilePath;
    diagnostics.deltaPath = deltaPath;
    diagnostics.profileColumns = profileNames;
    diagnostics.deltaColumns = deltaNames;
    diagnostics.rawSampleCount = numel(phiRaw);
    diagnostics.uniquePeriodicSampleCount = numel(phiUnique);
    diagnostics.duplicatePeriodicSamplesCollapsed = duplicateCount;
    diagnostics.rawPhaseRange = [min(phiRaw), max(phiRaw)];
    diagnostics.rawProfileRange = [min(aRaw), max(aRaw)];
    diagnostics.gridSize = nGrid;
    diagnostics.gridRange = [phiGrid(1), phiGrid(end)];
    diagnostics.interpolationMethod = 'linear periodic';
    diagnostics.phaseMergeTolerance = phaseMergeTolerance;
    diagnostics.deltaDegreeResidual = deltaDegreeResidual;
    diagnostics.deltaPiResidual = deltaPiResidual;
end

% =========================================================================
% Core analysis
% =========================================================================

function analysis = computeAnalysis(phi, a, delta, psiTarget, nu, powerValue, ...
        manualTau, tolerances)
    p = powerValue / (2 * pi);
    h = a .* cos(phi - psiTarget - delta);
    k = a .* sin(phi - psiTarget - delta);

    [zOptimal, optimalInfo] = solveOptimalProfile(phi, h, k, nu, p, tolerances);
    [zBestSinusoid, bestTau, sinusoidInfo] = solveBestSinusoid( ...
        phi, h, k, nu, powerValue, tolerances);
    zManual = sqrt(powerValue / pi) .* sin(phi + manualTau);

    psi = linspace(-pi, pi, numel(phi) + 1).';
    optimal = evaluateDesign('Optimal', zOptimal, optimalInfo.feasible, ...
        phi, a, delta, psi, psiTarget, nu, powerValue, tolerances);
    bestSinusoid = evaluateDesign('Best sinusoid', zBestSinusoid, ...
        sinusoidInfo.feasible, phi, a, delta, psi, psiTarget, nu, ...
        powerValue, tolerances);
    manual = evaluateDesign('Manual sinusoid', zManual, true, phi, a, ...
        delta, psi, psiTarget, nu, powerValue, tolerances);

    optimal.status = constrainedDesignStatus(optimal, optimalInfo, tolerances);
    bestSinusoid.status = sinusoidDesignStatus(bestSinusoid, sinusoidInfo, tolerances);
    manual.status = manualDesignStatus(manual, tolerances);

    analysis = struct();
    analysis.phi = phi;
    analysis.a = a;
    analysis.delta = delta;
    analysis.psiTarget = psiTarget;
    analysis.nu = nu;
    analysis.power = powerValue;
    analysis.manualTau = manualTau;
    analysis.psi = psi;
    analysis.h = h;
    analysis.k = k;
    analysis.optimal = optimal;
    analysis.bestSinusoid = bestSinusoid;
    analysis.manual = manual;
    analysis.bestTau = bestTau;
    analysis.optimalInfo = optimalInfo;
    analysis.sinusoidInfo = sinusoidInfo;
end

function [z, info] = solveOptimalProfile(phi, h, k, nu, p, tolerances)
    hNorm = mean(h .* h);
    hkInnerProduct = mean(h .* k);
    kNorm = mean(k .* k);
    metricZeroTolerance = tolerances.abs;

    info = struct();
    info.H = hNorm;
    info.B = hkInnerProduct;
    info.K = kNorm;
    info.KPerp = NaN;
    info.p = p;
    info.feasible = false;
    info.degenerate = false;
    info.nonunique = false;
    info.remainingPower = NaN;
    info.feasibilityMargin = p * hNorm - nu^2;
    info.feasibilityTolerance = scaledTolerance(tolerances, ...
        [p * hNorm, nu^2]);
    info.note = '';
    z = nan(size(h));

    if hNorm > metricZeroTolerance
        kPerp = k - (hkInnerProduct / hNorm) .* h;
        kPerpNorm = mean(kPerp .* kPerp);
        info.KPerp = kPerpNorm;

        if info.feasibilityMargin < -info.feasibilityTolerance
            info.note = 'Infeasible because nu^2 exceeds p H.';
            return;
        end

        info.feasible = true;
        remainingPower = max(0, info.feasibilityMargin / hNorm);
        info.remainingPower = remainingPower;
        zBase = -(nu / hNorm) .* h;
        powerTolerance = scaledTolerance(tolerances, [p, remainingPower]);

        if remainingPower <= powerTolerance
            % Do not form 0/sqrt(KPerp); this also covers a tangent energy
            % constraint with KPerp=0.
            z = zBase;
            info.note = 'Unique minimum-norm target solution; no residual power remains.';
        elseif kPerpNorm > metricZeroTolerance
            % Numerically stable form of
            % -sqrt((p-nu^2/H)/KPerp)*kPerp.
            z = zBase - sqrt(remainingPower) .* (kPerp ./ sqrt(kPerpNorm));
            info.note = 'Closed-form optimum.';
        else
            % The objective is flat in h-perpendicular directions.  Add a
            % deterministic unit direction to satisfy the energy equality.
            q = deterministicOrthogonalUnit(phi, h, tolerances);
            z = zBase + sqrt(remainingPower) .* q;
            info.degenerate = true;
            info.nonunique = true;
            info.note = ['Feasible degenerate case: KPerp is zero, so the ' ...
                         'energy-completing optimum is nonunique.'];
        end
        return;
    end

    % Degenerate h: the target value is independent of z.  It can only be
    % zero, after which the steepest negative slope is along -k.
    info.degenerate = true;
    equilibriumTolerance = scaledTolerance(tolerances, nu);
    if abs(nu) > equilibriumTolerance
        info.note = 'Infeasible because h is zero while nu is nonzero.';
        return;
    end

    info.feasible = true;
    info.feasibilityMargin = 0;
    info.remainingPower = p;
    if p <= scaledTolerance(tolerances, p)
        z = zeros(size(h));
        info.note = 'Degenerate zero-power solution.';
    elseif kNorm > metricZeroTolerance
        z = -sqrt(p / kNorm) .* k;
        info.note = 'Degenerate h=0 solution optimized directly along -k.';
    else
        z = sqrt(p) .* deterministicOrthogonalUnit(phi, h, tolerances);
        info.nonunique = true;
        info.note = 'Degenerate h=k=0 case; every profile with the specified power is optimal.';
    end
end

function [z, tau, info] = solveBestSinusoid(phi, h, k, nu, powerValue, tolerances)
    amplitude = sqrt(powerValue / pi);
    u = amplitude .* sin(phi);
    v = amplitude .* cos(phi);
    e = [mean(u .* h); mean(v .* h)];
    d = [mean(u .* k); mean(v .* k)];
    eNorm = hypot(e(1), e(2));
    feasibilityTolerance = scaledTolerance(tolerances, [eNorm, nu]);

    info = struct();
    info.e = e;
    info.d = d;
    info.eNorm = eNorm;
    info.feasible = false;
    info.degenerate = false;
    info.candidateTau = nan(2, 1);
    info.candidateSlope = nan(2, 1);
    info.note = '';
    tau = NaN;
    z = nan(size(phi));

    if eNorm <= tolerances.abs
        info.degenerate = true;
        if abs(nu) > feasibilityTolerance
            info.note = 'No sinusoid can satisfy the target equilibrium.';
            return;
        end

        info.feasible = true;
        dNorm = hypot(d(1), d(2));
        if dNorm > tolerances.abs
            direction = -d / dNorm;
            info.note = 'All sinusoid phases satisfy the target; the steepest one was selected.';
        else
            direction = [1; 0];
            info.note = 'All sinusoid phases have the same target value and slope; tau=0 was selected.';
        end
        tau = wrapAngle(atan2(direction(2), direction(1)));
        z = u .* cos(tau) + v .* sin(tau);
        info.candidateTau(:) = tau;
        info.candidateSlope(:) = d.' * direction;
        return;
    end

    if abs(nu) > eNorm + feasibilityTolerance
        info.note = 'No sinusoid can satisfy the target equilibrium.';
        return;
    end

    info.feasible = true;
    eUnit = e / eNorm;
    perpendicular = [-eUnit(2); eUnit(1)];
    constrainedProjection = min(1, max(-1, -nu / eNorm));
    transverseMagnitude = sqrt(max(0, 1 - constrainedProjection^2));
    candidates = [constrainedProjection .* eUnit + transverseMagnitude .* perpendicular, ...
                  constrainedProjection .* eUnit - transverseMagnitude .* perpendicular];
    slopes = d.' * candidates;
    [~, bestIndex] = min(slopes);
    direction = candidates(:, bestIndex);
    tau = wrapAngle(atan2(direction(2), direction(1)));
    z = u .* cos(tau) + v .* sin(tau);
    info.candidateTau = wrapAngle(atan2(candidates(2, :).', candidates(1, :).'));
    info.candidateSlope = slopes(:);
    info.note = 'Best of the two analytic target-constrained sinusoid phases.';
end

function design = evaluateDesign(label, z, available, phi, a, delta, psi, ...
        psiTarget, nu, requestedPower, tolerances)
    design = emptyDesign(label, size(phi), size(psi));
    design.available = available;
    design.z = z;
    if ~available
        return;
    end

    cosineKernel = a .* cos(phi - delta);
    sineKernel = a .* sin(phi - delta);
    design.C = mean(z .* cosineKernel);
    design.D = mean(z .* sineKernel);
    design.gamma = design.C .* cos(psi) + design.D .* sin(psi);
    design.phaseVelocity = nu + design.gamma;
    design.gammaAtTarget = design.C * cos(psiTarget) + design.D * sin(psiTarget);
    design.gammaPrimeAtTarget = -design.C * sin(psiTarget) + ...
        design.D * cos(psiTarget);
    design.kappa = -design.gammaPrimeAtTarget;
    design.lockingHalfWidth = hypot(design.C, design.D);
    design.energy = 2 * pi * mean(z .* z);
    design.equilibriumResidual = abs(nu + design.gammaAtTarget);
    design.energyResidual = abs(design.energy - requestedPower);
    design.trigonometricIdentityResidual = abs(design.lockingHalfWidth^2 - ...
        (design.gammaAtTarget^2 + design.gammaPrimeAtTarget^2));
    design.targetLockingIdentityResidual = abs(design.lockingHalfWidth^2 - ...
        (nu^2 + design.gammaPrimeAtTarget^2));
    design.targetEquilibriumTolerance = scaledTolerance(tolerances, ...
        [nu, design.gammaAtTarget, design.lockingHalfWidth]);
    design.isTargetEquilibrium = ...
        design.equilibriumResidual <= design.targetEquilibriumTolerance;
    design.equilibria = findEquilibria(nu, design.C, design.D, tolerances);
end

function design = emptyDesign(label, phiSize, psiSize)
    design = struct();
    design.label = label;
    design.available = false;
    design.z = nan(phiSize);
    design.C = NaN;
    design.D = NaN;
    design.gamma = nan(psiSize);
    design.phaseVelocity = nan(psiSize);
    design.gammaAtTarget = NaN;
    design.gammaPrimeAtTarget = NaN;
    design.kappa = NaN;
    design.lockingHalfWidth = NaN;
    design.energy = NaN;
    design.equilibriumResidual = NaN;
    design.energyResidual = NaN;
    design.trigonometricIdentityResidual = NaN;
    design.targetLockingIdentityResidual = NaN;
    design.targetEquilibriumTolerance = NaN;
    design.isTargetEquilibrium = false;
    design.equilibria = emptyEquilibriumSet();
    design.status = 'Unavailable.';
end

function status = constrainedDesignStatus(design, solverInfo, tolerances)
    if ~solverInfo.feasible
        status = ['Infeasible: ' solverInfo.note];
        return;
    end
    if ~design.isTargetEquilibrium
        status = 'Numerical failure: the computed profile misses the target equilibrium.';
        return;
    end
    slopeTolerance = scaledTolerance(tolerances, design.lockingHalfWidth);
    if design.gammaPrimeAtTarget < -slopeTolerance
        status = 'Feasible and stable.';
    elseif design.gammaPrimeAtTarget > slopeTolerance
        status = ['Feasible equilibrium, but it cannot be stabilized under this ' ...
                  'constraint (even the optimal slope is positive).'];
    else
        status = ['Feasible equilibrium, but it cannot be asymptotically stabilized ' ...
                  'under this constraint (optimal slope is neutral).'];
    end
end

function status = sinusoidDesignStatus(design, solverInfo, tolerances)
    if ~solverInfo.feasible
        status = 'Infeasible within the fixed-energy sinusoid family.';
        return;
    end
    if ~design.isTargetEquilibrium
        status = 'Numerical failure: the selected sinusoid misses the target equilibrium.';
        return;
    end
    slopeTolerance = scaledTolerance(tolerances, design.lockingHalfWidth);
    if design.gammaPrimeAtTarget < -slopeTolerance
        status = 'Feasible within the sinusoid family and stable.';
    elseif design.gammaPrimeAtTarget > slopeTolerance
        status = 'Feasible within the sinusoid family, but unstable.';
    else
        status = 'Feasible within the sinusoid family, but neutral.';
    end
end

function status = manualDesignStatus(design, tolerances)
    if ~design.isTargetEquilibrium
        status = 'Manual comparison only: psi_d is not an equilibrium.';
        return;
    end
    slopeTolerance = scaledTolerance(tolerances, design.lockingHalfWidth);
    if design.gammaPrimeAtTarget < -slopeTolerance
        status = 'Manual comparison: target equilibrium is stable.';
    elseif design.gammaPrimeAtTarget > slopeTolerance
        status = 'Manual comparison: target equilibrium is unstable.';
    else
        status = 'Manual comparison: target equilibrium is neutral.';
    end
end

% =========================================================================
% Equilibrium enumeration and classification
% =========================================================================

function equilibria = findEquilibria(nu, cCoefficient, dCoefficient, tolerances)
    equilibria = emptyEquilibriumSet();
    amplitude = hypot(cCoefficient, dCoefficient);
    rootTolerance = scaledTolerance(tolerances, [nu, amplitude]);
    equilibria.amplitude = amplitude;

    if amplitude <= rootTolerance
        if abs(nu) <= rootTolerance
            equilibria.continuum = true;
            equilibria.hasEquilibria = true;
            equilibria.note = 'F(psi) is identically zero; every phase is a neutral equilibrium.';
        else
            equilibria.note = 'No equilibrium: the interaction is zero and nu is nonzero.';
        end
        return;
    end

    rootRatio = -nu / amplitude;
    if abs(rootRatio) > 1 + rootTolerance
        equilibria.note = 'No equilibrium because |nu| exceeds the locking half-width.';
        return;
    end

    rootRatio = min(1, max(-1, rootRatio));
    interactionPhase = atan2(dCoefficient, cCoefficient);
    angularOffset = acos(rootRatio);
    roots = wrapAngle([interactionPhase + angularOffset; ...
                       interactionPhase - angularOffset]);
    if angularDistance(roots(1), roots(2)) <= tolerances.angle
        roots = roots(1);
    end
    roots = sort(roots(:));

    slopes = -cCoefficient .* sin(roots) + dCoefficient .* cos(roots);
    slopeTolerance = scaledTolerance(tolerances, amplitude);
    classifications = cell(numel(roots), 1);
    for index = 1:numel(roots)
        if slopes(index) < -slopeTolerance
            classifications{index} = 'stable';
        elseif slopes(index) > slopeTolerance
            classifications{index} = 'unstable';
        else
            classifications{index} = 'tangent';
        end
    end

    equilibria.psi = roots;
    equilibria.psiOverPi = roots / pi;
    equilibria.slope = slopes;
    equilibria.classification = classifications;
    equilibria.hasEquilibria = true;
    equilibria.note = sprintf('%d isolated periodic equilibrium point(s).', numel(roots));
end

function equilibria = emptyEquilibriumSet()
    equilibria = struct();
    equilibria.psi = zeros(0, 1);
    equilibria.psiOverPi = zeros(0, 1);
    equilibria.slope = zeros(0, 1);
    equilibria.classification = cell(0, 1);
    equilibria.continuum = false;
    equilibria.hasEquilibria = false;
    equilibria.amplitude = NaN;
    equilibria.note = '';
end

% =========================================================================
% Diagnostics and output structure
% =========================================================================

function checks = runInternalChecks(phi, a, delta, tolerances)
    % Use the requested reference case independently of the caller's current
    % slider/input state, so the diagnostics always exercise the sign and
    % normalization conventions.
    reference = computeAnalysis(phi, a, delta, 0, 0, pi, 0, tolerances);
    referenceEquilibriumPass = reference.optimal.equilibriumResidual < 1e-9;
    referenceEnergyPass = reference.optimal.energyResidual < 1e-9;
    referenceSlopePass = reference.optimal.gammaAtTarget == ...
        reference.optimal.gammaAtTarget && ... % explicit finite check
        abs(reference.optimal.gammaAtTarget) < 1e-9 && ...
        reference.optimal.gammaPrimeAtTarget < 0;

    constantProfile = computeAnalysis(phi, ones(size(phi)), delta, ...
        0, 0, pi, 0, tolerances);
    expectedConstantProfile = -sin(phi - delta);
    relativeProfileError = sqrt(mean((constantProfile.optimal.z - ...
        expectedConstantProfile).^2)) / max(sqrt(mean(expectedConstantProfile.^2)), eps);
    syntheticPass = relativeProfileError < 1e-10 && ...
        constantProfile.optimal.equilibriumResidual < 1e-10 && ...
        constantProfile.optimal.energyResidual < 1e-10 && ...
        constantProfile.optimal.gammaPrimeAtTarget < 0;

    expectedKappaOptimal = 0.302;
    expectedKappaSinusoid = 0.153;
    expectedTauOverPi = 0.679;
    benchmarkPass = abs(reference.optimal.kappa - expectedKappaOptimal) < 5e-3 && ...
        abs(reference.bestSinusoid.kappa - expectedKappaSinusoid) < 5e-3 && ...
        abs(wrapAngle(reference.bestTau - expectedTauOverPi * pi)) / pi < 5e-3;

    checks = struct();
    checks.referenceNuZeroPsiZero = struct( ...
        'equilibriumResidual', reference.optimal.equilibriumResidual, ...
        'energyResidual', reference.optimal.energyResidual, ...
        'gammaAtTarget', reference.optimal.gammaAtTarget, ...
        'gammaPrimeAtTarget', reference.optimal.gammaPrimeAtTarget, ...
        'pass', referenceEquilibriumPass && referenceEnergyPass && referenceSlopePass);
    checks.syntheticConstantProfile = struct( ...
        'relativeProfileError', relativeProfileError, ...
        'equilibriumResidual', constantProfile.optimal.equilibriumResidual, ...
        'energyResidual', constantProfile.optimal.energyResidual, ...
        'kappa', constantProfile.optimal.kappa, ...
        'expectedKappa', 0.5, ...
        'pass', syntheticPass);
    checks.defaultCsvBenchmark = struct( ...
        'kappaOptimal', reference.optimal.kappa, ...
        'kappaBestSinusoid', reference.bestSinusoid.kappa, ...
        'bestTauOverPi', reference.bestTau / pi, ...
        'pass', benchmarkPass);
    checks.allPassed = checks.referenceNuZeroPsiZero.pass && ...
        checks.syntheticConstantProfile.pass && checks.defaultCsvBenchmark.pass;
end

function result = assembleResult(analysis, inputDiagnostics, selfChecks, tolerances)
    result = struct();
    result.phi = analysis.phi;
    result.a = analysis.a;
    result.delta = analysis.delta;
    result.psiTarget = analysis.psiTarget;
    result.nu = analysis.nu;
    result.power = analysis.power;
    result.psi = analysis.psi;
    result.zOptimal = analysis.optimal.z;
    result.zBestSinusoid = analysis.bestSinusoid.z;
    result.zManual = analysis.manual.z;
    result.bestTau = analysis.bestTau;
    result.manualTau = analysis.manualTau;
    result.gammaOptimal = analysis.optimal.gamma;
    result.gammaBestSinusoid = analysis.bestSinusoid.gamma;
    result.gammaManual = analysis.manual.gamma;
    result.phaseVelocityOptimal = analysis.optimal.phaseVelocity;
    result.phaseVelocityBestSinusoid = analysis.bestSinusoid.phaseVelocity;
    result.phaseVelocityManual = analysis.manual.phaseVelocity;
    result.kappaOptimal = analysis.optimal.kappa;
    result.kappaBestSinusoid = analysis.bestSinusoid.kappa;
    result.kappaManual = analysis.manual.kappa;
    result.lockingHalfWidthOptimal = analysis.optimal.lockingHalfWidth;
    result.lockingHalfWidthBestSinusoid = analysis.bestSinusoid.lockingHalfWidth;
    result.lockingHalfWidthManual = analysis.manual.lockingHalfWidth;
    result.equilibria = struct( ...
        'optimal', analysis.optimal.equilibria, ...
        'bestSinusoid', analysis.bestSinusoid.equilibria, ...
        'manual', analysis.manual.equilibria);

    diagnostics = struct();
    diagnostics.input = inputDiagnostics;
    diagnostics.tolerances = tolerances;
    diagnostics.optimalFeasible = analysis.optimalInfo.feasible;
    diagnostics.optimalStatus = analysis.optimal.status;
    diagnostics.bestSinusoidFeasible = analysis.sinusoidInfo.feasible;
    diagnostics.bestSinusoidStatus = analysis.bestSinusoid.status;
    diagnostics.manualStatus = analysis.manual.status;
    diagnostics.H = analysis.optimalInfo.H;
    diagnostics.B = analysis.optimalInfo.B;
    diagnostics.KPerp = analysis.optimalInfo.KPerp;
    diagnostics.feasibilityMargin = analysis.optimalInfo.feasibilityMargin;
    diagnostics.optimalEquilibriumResidual = analysis.optimal.equilibriumResidual;
    diagnostics.optimalEnergyResidual = analysis.optimal.energyResidual;
    diagnostics.optimalLockingIdentityResidual = ...
        analysis.optimal.targetLockingIdentityResidual;
    diagnostics.bestSinusoidEquilibriumResidual = ...
        analysis.bestSinusoid.equilibriumResidual;
    diagnostics.bestSinusoidEnergyResidual = analysis.bestSinusoid.energyResidual;
    diagnostics.bestSinusoidLockingIdentityResidual = ...
        analysis.bestSinusoid.targetLockingIdentityResidual;
    diagnostics.manualEquilibriumResidual = analysis.manual.equilibriumResidual;
    diagnostics.manualEnergyResidual = analysis.manual.energyResidual;
    diagnostics.selfChecks = selfChecks;
    diagnostics.phaseConvention = struct( ...
        'modePhaseDifference', 'psi = phi_i - Phi_i', ...
        'physicalSensorPhaseDifference', 'psi_phys = psi + delta', ...
        'modeInPhaseTarget', 0, ...
        'sensorInPhaseTarget', wrapAngle(-analysis.delta));
    result.diagnostics = diagnostics;
end

function reduced = resultWithoutFigure(result)
    reduced = result;
    if isfield(reduced, 'figure')
        reduced = rmfield(reduced, 'figure');
    end
end

% =========================================================================
% Interactive and static displays
% =========================================================================

function fig = createInteractiveFigure(phi, a, delta, initialAnalysis, ...
        inputDiagnostics, selfChecks, tolerances, baseDir)
    state.psiTarget = initialAnalysis.psiTarget;
    state.nu = initialAnalysis.nu;
    state.manualTau = initialAnalysis.manualTau;
    state.power = initialAnalysis.power;
    state.analysis = initialAnalysis;

    fig = uifigure('Name', 'Target-equilibrium optimal z analysis', ...
        'Color', 'white', 'Position', [70, 50, 1450, 900]);
    rootGrid = uigridlayout(fig, [3, 2]);
    rootGrid.RowHeight = {155, '1x', 190};
    rootGrid.ColumnWidth = {'1x', '1x'};
    rootGrid.Padding = [10, 10, 10, 10];
    rootGrid.RowSpacing = 8;
    rootGrid.ColumnSpacing = 8;

    controlsPanel = uipanel(rootGrid, 'Title', 'Controls');
    controlsPanel.Layout.Row = 1;
    controlsPanel.Layout.Column = [1, 2];
    controlsGrid = uigridlayout(controlsPanel, [4, 6]);
    controlsGrid.RowHeight = {26, 26, 26, 28};
    controlsGrid.ColumnWidth = {135, '1x', 75, 135, '1x', 110};
    controlsGrid.Padding = [8, 4, 8, 4];

    psiLabel = uilabel(controlsGrid, 'Text', 'Target psi_d / pi');
    psiLabel.Layout.Row = 1;
    psiLabel.Layout.Column = 1;
    psiSlider = uislider(controlsGrid, 'Limits', [-1, 1], ...
        'Value', state.psiTarget / pi, 'MajorTicks', -1:0.5:1);
    psiSlider.Layout.Row = 1;
    psiSlider.Layout.Column = 2;
    psiValue = uilabel(controlsGrid, 'Text', sprintf('%.5f', state.psiTarget / pi), ...
        'HorizontalAlignment', 'right');
    psiValue.Layout.Row = 1;
    psiValue.Layout.Column = 3;

    nuLabel = uilabel(controlsGrid, 'Text', 'Normalized nu');
    nuLabel.Layout.Row = 1;
    nuLabel.Layout.Column = 4;
    nuSlider = uislider(controlsGrid, 'Limits', [-0.8, 0.8], ...
        'Value', state.nu, 'MajorTicks', -0.8:0.4:0.8);
    nuSlider.Layout.Row = 1;
    nuSlider.Layout.Column = 5;
    nuValue = uilabel(controlsGrid, 'Text', sprintf('%.5f', state.nu), ...
        'HorizontalAlignment', 'right');
    nuValue.Layout.Row = 1;
    nuValue.Layout.Column = 6;

    tauLabel = uilabel(controlsGrid, 'Text', 'Manual tau / pi');
    tauLabel.Layout.Row = 2;
    tauLabel.Layout.Column = 1;
    tauSlider = uislider(controlsGrid, 'Limits', [-1, 1], ...
        'Value', state.manualTau / pi, 'MajorTicks', -1:0.5:1);
    tauSlider.Layout.Row = 2;
    tauSlider.Layout.Column = 2;
    tauValue = uilabel(controlsGrid, 'Text', sprintf('%.5f', state.manualTau / pi), ...
        'HorizontalAlignment', 'right');
    tauValue.Layout.Row = 2;
    tauValue.Layout.Column = 3;

    setModeButton = uibutton(controlsGrid, 'push', ...
        'Text', 'Set psi_d = 0');
    setModeButton.Layout.Row = 2;
    setModeButton.Layout.Column = 4;
    setSensorButton = uibutton(controlsGrid, 'push', ...
        'Text', 'Set psi_d = -delta');
    setSensorButton.Layout.Row = 2;
    setSensorButton.Layout.Column = 5;
    exportButton = uibutton(controlsGrid, 'push', 'Text', 'Export PNG');
    exportButton.Layout.Row = 2;
    exportButton.Layout.Column = 6;
    exportPrcButton = uibutton(controlsGrid, 'push', 'Text', 'Export PRC Txt');
    exportPrcButton.Layout.Row = 3;
    exportPrcButton.Layout.Column = 6;

    showOptimal = uicheckbox(controlsGrid, 'Text', 'Show optimal', 'Value', true);
    showOptimal.Layout.Row = 3;
    showOptimal.Layout.Column = 1;
    showBest = uicheckbox(controlsGrid, 'Text', 'Show best sinusoid', 'Value', true);
    showBest.Layout.Row = 3;
    showBest.Layout.Column = 2;
    showManual = uicheckbox(controlsGrid, 'Text', 'Show manual sinusoid', 'Value', true);
    showManual.Layout.Row = 3;
    showManual.Layout.Column = 3;

    conventionLabel = uilabel(controlsGrid, ...
        'Text', ['psi = phi_i - Phi_i (mode);  psi_phys = psi + delta (actual sensor).  ' ...
                 'psi_d=0: mode in phase;  psi_d=-delta: sensor in phase.'], ...
        'WordWrap', 'on');
    conventionLabel.Layout.Row = [3, 4];
    conventionLabel.Layout.Column = [4, 5];

    powerLabel = uilabel(controlsGrid, ...
        'Text', sprintf('Power P = %.9g; delta/pi = %.9g', state.power, delta / pi));
    powerLabel.Layout.Row = 4;
    powerLabel.Layout.Column = [1, 3];

    zAxes = uiaxes(rootGrid);
    zAxes.Layout.Row = 2;
    zAxes.Layout.Column = 1;
    phaseAxes = uiaxes(rootGrid);
    phaseAxes.Layout.Row = 2;
    phaseAxes.Layout.Column = 2;

    diagnosticsArea = uitextarea(rootGrid, 'Editable', 'off', ...
        'FontName', 'Consolas');
    diagnosticsArea.Layout.Row = 3;
    diagnosticsArea.Layout.Column = [1, 2];

    psiSlider.ValueChangingFcn = @onPsiChanging;
    psiSlider.ValueChangedFcn = @onPsiChanged;
    nuSlider.ValueChangingFcn = @onNuChanging;
    nuSlider.ValueChangedFcn = @onNuChanged;
    tauSlider.ValueChangingFcn = @onTauChanging;
    tauSlider.ValueChangedFcn = @onTauChanged;
    setModeButton.ButtonPushedFcn = @setModeTarget;
    setSensorButton.ButtonPushedFcn = @setSensorTarget;
    exportButton.ButtonPushedFcn = @exportPng;
    exportPrcButton.ButtonPushedFcn = @exportPrcTxt;
    showOptimal.ValueChangedFcn = @visibilityChanged;
    showBest.ValueChangedFcn = @visibilityChanged;
    showManual.ValueChangedFcn = @visibilityChanged;

    plotHandles = initInteractivePlotHandles(zAxes, phaseAxes, initialAnalysis, ...
        [showOptimal.Value, showBest.Value, showManual.Value]);
    refreshDisplay(false);

    function onPsiChanging(~, event)
        state.psiTarget = pi * event.Value;
        psiValue.Text = sprintf('%.5f', event.Value);
        refreshDisplay(true);
    end

    function onPsiChanged(source, ~)
        state.psiTarget = pi * source.Value;
        psiValue.Text = sprintf('%.5f', source.Value);
        refreshDisplay(false);
    end

    function onNuChanging(~, event)
        state.nu = event.Value;
        nuValue.Text = sprintf('%.5f', event.Value);
        refreshDisplay(true);
    end

    function onNuChanged(source, ~)
        state.nu = source.Value;
        nuValue.Text = sprintf('%.5f', source.Value);
        refreshDisplay(false);
    end

    function onTauChanging(~, event)
        state.manualTau = pi * event.Value;
        tauValue.Text = sprintf('%.5f', event.Value);
        refreshDisplay(true);
    end

    function onTauChanged(source, ~)
        state.manualTau = pi * source.Value;
        tauValue.Text = sprintf('%.5f', source.Value);
        refreshDisplay(false);
    end

    function setModeTarget(~, ~)
        state.psiTarget = 0;
        psiSlider.Value = 0;
        psiValue.Text = '0.00000';
        refreshDisplay(false);
    end

    function setSensorTarget(~, ~)
        state.psiTarget = wrapAngle(-delta);
        psiSlider.Value = state.psiTarget / pi;
        psiValue.Text = sprintf('%.5f', state.psiTarget / pi);
        refreshDisplay(false);
    end

    function visibilityChanged(~, ~)
        refreshDisplay(false);
    end

    function refreshDisplay(limitRate)
        state.analysis = computeAnalysis(phi, a, delta, state.psiTarget, ...
            state.nu, state.power, state.manualTau, tolerances);
        visibility = [showOptimal.Value, showBest.Value, showManual.Value];
        updateInteractivePlotHandles(plotHandles, state.analysis, visibility);
        diagnosticsArea.Value = formatDiagnostics(state.analysis, selfChecks);
        if limitRate
            drawnow limitrate;
        else
            currentResult = assembleResult(state.analysis, inputDiagnostics, ...
                selfChecks, tolerances);
            fig.UserData = struct('result', currentResult);
            drawnow;
        end
    end

    function exportPng(~, ~)
        [fileName, folderName] = uiputfile({'*.png', 'PNG image (*.png)'}, ...
            'Export target-equilibrium analysis', ...
            fullfile(baseDir, 'target_equilibrium_optimal_z_analysis.png'));
        if isequal(fileName, 0)
            return;
        end
        outputPath = fullfile(folderName, fileName);
        [~, ~, extension] = fileparts(outputPath);
        if isempty(extension)
            outputPath = [outputPath '.png'];
        end
        try
            exportapp(fig, outputPath);
        catch exportError
            uialert(fig, exportError.message, 'PNG export failed');
        end
    end

    function exportPrcTxt(~, ~)
        psiTargetOverPi = state.analysis.psiTarget / pi;
        defaultFileName = sprintf('prc_snippet_target_optimal_z_psid_%.2fpi.txt', psiTargetOverPi);
        defaultPath = fullfile(baseDir, '..', 'gamma_exports', defaultFileName);
        [fileName, folderName] = uiputfile({'*.txt', 'PRC snippet file (*.txt)'}, ...
            'Export PRC snippet for ServerResponse.py', defaultPath);
        if isequal(fileName, 0)
            return;
        end
        outputPath = fullfile(folderName, fileName);
        try
            writePrcSnippetFile(state.analysis, 10, outputPath);
            uialert(fig, sprintf('Successfully exported PRC snippet to:\n%s', outputPath), ...
                'Export PRC Snippet');
        catch exportError
            uialert(fig, exportError.message, 'PRC export failed');
        end
    end
end

function handles = initInteractivePlotHandles(zAxes, phaseAxes, initialAnalysis, visibility)
    colors = [0.0000, 0.4470, 0.7410; ...
              0.8500, 0.3250, 0.0980; ...
              0.4660, 0.6740, 0.1880];
    lineStyles = {'-', '--', ':'};
    labels = {'Optimal', 'Best constrained sinusoid', 'Manual sinusoid'};

    delete(allchild(zAxes));
    delete(allchild(phaseAxes));

    hold(zAxes, 'on');
    handles.zLines = gobjects(3, 1);
    for i = 1:3
        handles.zLines(i) = plot(zAxes, initialAnalysis.phi / pi, nan(size(initialAnalysis.phi)), ...
            'Color', colors(i, :), 'LineStyle', lineStyles{i}, 'LineWidth', 1.6, ...
            'DisplayName', labels{i});
    end
    yline(zAxes, 0, 'k:', 'HandleVisibility', 'off');
    hold(zAxes, 'off');
    grid(zAxes, 'on');
    box(zAxes, 'on');
    xlim(zAxes, [0, 2]);
    ylim(zAxes, [-2.5, 2.5]);
    xlabel(zAxes, '\phi / \pi');
    ylabel(zAxes, 'z(\phi)');
    title(zAxes, 'Phase sensitivity functions');

    hold(phaseAxes, 'on');
    handles.phaseLines = gobjects(3, 1);
    for i = 1:3
        handles.phaseLines(i) = plot(phaseAxes, initialAnalysis.psi / pi, nan(size(initialAnalysis.psi)), ...
            'Color', colors(i, :), 'LineStyle', lineStyles{i}, 'LineWidth', 1.6, ...
            'DisplayName', labels{i});
    end
    yline(phaseAxes, 0, 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    handles.xTargetLine = xline(phaseAxes, initialAnalysis.psiTarget / pi, 'k--', ...
        'LineWidth', 1.1, 'HandleVisibility', 'off');

    handles.eqMarkers = gobjects(3, 3);
    for i = 1:3
        handles.eqMarkers(i, 1) = plot(phaseAxes, nan, nan, 'o', 'Color', colors(i, :), ...
            'MarkerFaceColor', colors(i, :), 'MarkerSize', 7, 'LineWidth', 1.2, ...
            'HandleVisibility', 'off');
        handles.eqMarkers(i, 2) = plot(phaseAxes, nan, nan, 'o', 'Color', colors(i, :), ...
            'MarkerFaceColor', 'white', 'MarkerSize', 7, 'LineWidth', 1.2, ...
            'HandleVisibility', 'off');
        handles.eqMarkers(i, 3) = plot(phaseAxes, nan, nan, 'd', 'Color', colors(i, :), ...
            'MarkerFaceColor', colors(i, :), 'MarkerSize', 7, 'LineWidth', 1.2, ...
            'HandleVisibility', 'off');
    end

    handles.stableKey = plot(phaseAxes, nan, nan, 'o', 'Color', 'k', ...
        'MarkerFaceColor', 'k', 'MarkerSize', 6);
    handles.unstableKey = plot(phaseAxes, nan, nan, 'o', 'Color', 'k', ...
        'MarkerFaceColor', 'white', 'MarkerSize', 6);
    handles.tangentKey = plot(phaseAxes, nan, nan, 'd', 'Color', 'k', ...
        'MarkerFaceColor', [0.7, 0.7, 0.7], 'MarkerSize', 6);
    hold(phaseAxes, 'off');
    grid(phaseAxes, 'on');
    box(phaseAxes, 'on');
    xlim(phaseAxes, [-1, 1]);
    ylim(phaseAxes, [-1, 1]);
    xlabel(phaseAxes, '\psi / \pi');
    ylabel(phaseAxes, 'F_z(\psi) = \nu + \Gamma_z(\psi)');
    title(phaseAxes, 'One-way phase-difference dynamics');

    handles.zAxes = zAxes;
    handles.phaseAxes = phaseAxes;
    handles.labels = labels;

    updateInteractivePlotHandles(handles, initialAnalysis, visibility);
end

function updateInteractivePlotHandles(handles, analysis, visibility)
    designs = {analysis.optimal, analysis.bestSinusoid, analysis.manual};
    zVisHandles = gobjects(0, 1);
    zVisNames = cell(0, 1);
    phaseVisHandles = gobjects(0, 1);
    phaseVisNames = cell(0, 1);

    for i = 1:3
        show = visibility(i) && designs{i}.available;
        if show
            handles.zLines(i).YData = designs{i}.z;
            handles.zLines(i).Visible = 'on';
            zVisHandles(end + 1, 1) = handles.zLines(i); %#ok<AGROW>
            zVisNames{end + 1, 1} = handles.labels{i}; %#ok<AGROW>

            handles.phaseLines(i).YData = designs{i}.phaseVelocity;
            handles.phaseLines(i).Visible = 'on';
            phaseVisHandles(end + 1, 1) = handles.phaseLines(i); %#ok<AGROW>
            phaseVisNames{end + 1, 1} = handles.labels{i}; %#ok<AGROW>

            updateEquilibriumMarkers(handles.eqMarkers(i, :), designs{i}.equilibria);
        else
            handles.zLines(i).Visible = 'off';
            handles.phaseLines(i).Visible = 'off';
            for c = 1:3
                handles.eqMarkers(i, c).XData = nan;
                handles.eqMarkers(i, c).YData = nan;
                handles.eqMarkers(i, c).Visible = 'off';
            end
        end
    end

    handles.xTargetLine.Value = analysis.psiTarget / pi;

    if isempty(zVisHandles)
        legend(handles.zAxes, 'off');
    else
        legend(handles.zAxes, zVisHandles, zVisNames, 'Location', 'best');
    end

    legendHandles = [phaseVisHandles; handles.stableKey; handles.unstableKey; handles.tangentKey];
    legendNames = [phaseVisNames; {'Stable equilibrium'; 'Unstable equilibrium'; 'Tangent equilibrium'}];
    legend(handles.phaseAxes, legendHandles, legendNames, 'Location', 'best');
end

function updateEquilibriumMarkers(markerHandles, equilibria)
    if equilibria.continuum || isempty(equilibria.psi)
        for c = 1:3
            markerHandles(c).XData = nan;
            markerHandles(c).YData = nan;
            markerHandles(c).Visible = 'off';
        end
        return;
    end

    classifications = {'stable', 'unstable', 'tangent'};
    for c = 1:3
        selection = strcmp(equilibria.classification, classifications{c});
        if ~any(selection)
            markerHandles(c).XData = nan;
            markerHandles(c).YData = nan;
            markerHandles(c).Visible = 'off';
            continue;
        end
        xValues = equilibria.psi(selection) / pi;
        edgeSelection = abs(abs(equilibria.psi(selection)) - pi) < 1e-8;
        if any(edgeSelection)
            edgeValues = xValues(edgeSelection);
            xValues = [xValues; -sign(edgeValues)]; %#ok<AGROW>
        end
        markerHandles(c).XData = xValues;
        markerHandles(c).YData = zeros(size(xValues));
        markerHandles(c).Visible = 'on';
    end
end

function fig = createStaticFigure(analysis, selfChecks)
    fig = figure('Name', 'Target-equilibrium optimal z analysis', ...
        'Color', 'white', 'Position', [80, 60, 1450, 850]);
    layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', ...
        'Padding', 'compact');
    zAxes = nexttile(layout, 1);
    phaseAxes = nexttile(layout, 2);
    drawAnalysisPlots(zAxes, phaseAxes, analysis, [true, true, true]);

    informationAxes = nexttile(layout, 3, [1, 2]);
    informationAxes.Visible = 'off';
    text(informationAxes, 0, 1, strjoin(formatDiagnostics(analysis, selfChecks), newline), ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'FontName', 'Consolas', 'Interpreter', 'none');
end

function drawAnalysisPlots(zAxes, phaseAxes, analysis, visibility)
    designs = {analysis.optimal, analysis.bestSinusoid, analysis.manual};
    labels = {'Optimal', 'Best constrained sinusoid', 'Manual sinusoid'};
    colors = [0.0000, 0.4470, 0.7410; ...
              0.8500, 0.3250, 0.0980; ...
              0.4660, 0.6740, 0.1880];
    lineStyles = {'-', '--', ':'};

    delete(allchild(zAxes));
    hold(zAxes, 'on');
    zHandles = gobjects(0, 1);
    zNames = cell(0, 1);
    for index = 1:3
        if visibility(index) && designs{index}.available
            handle = plot(zAxes, analysis.phi / pi, designs{index}.z, ...
                'Color', colors(index, :), 'LineStyle', lineStyles{index}, ...
                'LineWidth', 1.6);
            zHandles(end + 1, 1) = handle; %#ok<AGROW>
            zNames{end + 1, 1} = labels{index}; %#ok<AGROW>
        end
    end
    yline(zAxes, 0, 'k:', 'HandleVisibility', 'off');
    hold(zAxes, 'off');
    grid(zAxes, 'on');
    box(zAxes, 'on');
    xlim(zAxes, [0, 2]);
    ylim(zAxes, [-2, 2]);
    xlabel(zAxes, '\phi / \pi');
    ylabel(zAxes, 'z(\phi)');
    title(zAxes, 'Phase sensitivity functions');
    if isempty(zHandles)
        legend(zAxes, 'off');
    else
        legend(zAxes, zHandles, zNames, 'Location', 'best');
    end

    delete(allchild(phaseAxes));
    hold(phaseAxes, 'on');
    phaseHandles = gobjects(0, 1);
    phaseNames = cell(0, 1);
    for index = 1:3
        if visibility(index) && designs{index}.available
            handle = plot(phaseAxes, analysis.psi / pi, ...
                designs{index}.phaseVelocity, 'Color', colors(index, :), ...
                'LineStyle', lineStyles{index}, 'LineWidth', 1.6);
            phaseHandles(end + 1, 1) = handle; %#ok<AGROW>
            phaseNames{end + 1, 1} = labels{index}; %#ok<AGROW>
            plotEquilibriumMarkers(phaseAxes, designs{index}.equilibria, ...
                colors(index, :));
        end
    end
    yline(phaseAxes, 0, 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    xline(phaseAxes, analysis.psiTarget / pi, 'k--', 'LineWidth', 1.1, ...
        'HandleVisibility', 'off');

    stableKey = plot(phaseAxes, nan, nan, 'o', 'Color', 'k', ...
        'MarkerFaceColor', 'k', 'MarkerSize', 6);
    unstableKey = plot(phaseAxes, nan, nan, 'o', 'Color', 'k', ...
        'MarkerFaceColor', 'white', 'MarkerSize', 6);
    tangentKey = plot(phaseAxes, nan, nan, 'd', 'Color', 'k', ...
        'MarkerFaceColor', [0.7, 0.7, 0.7], 'MarkerSize', 6);
    hold(phaseAxes, 'off');
    grid(phaseAxes, 'on');
    box(phaseAxes, 'on');
    xlim(phaseAxes, [-1, 1]);
    ylim(phaseAxes, [-1, 1]);
    xlabel(phaseAxes, '\psi / \pi');
    ylabel(phaseAxes, 'F_z(\psi) = \nu + \Gamma_z(\psi)');
    title(phaseAxes, 'One-way phase-difference dynamics');
    legendHandles = [phaseHandles; stableKey; unstableKey; tangentKey];
    legendNames = [phaseNames; {'Stable equilibrium'; 'Unstable equilibrium'; ...
                               'Tangent equilibrium'}];
    legend(phaseAxes, legendHandles, legendNames, 'Location', 'best');
end

function plotEquilibriumMarkers(axesHandle, equilibria, color)
    if equilibria.continuum || isempty(equilibria.psi)
        return;
    end
    classifications = {'stable', 'unstable', 'tangent'};
    markers = {'o', 'o', 'd'};
    for classIndex = 1:numel(classifications)
        selection = strcmp(equilibria.classification, classifications{classIndex});
        if ~any(selection)
            continue;
        end
        xValues = equilibria.psi(selection) / pi;
        edgeSelection = abs(abs(equilibria.psi(selection)) - pi) < 1e-8;
        if any(edgeSelection)
            edgeValues = xValues(edgeSelection);
            xValues = [xValues; -sign(edgeValues)]; %#ok<AGROW>
        end
        if classIndex == 1
            faceColor = color;
        elseif classIndex == 2
            faceColor = 'white';
        else
            faceColor = color;
        end
        plot(axesHandle, xValues, zeros(size(xValues)), markers{classIndex}, ...
            'Color', color, 'MarkerFaceColor', faceColor, 'MarkerSize', 7, ...
            'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
end

function lines = formatDiagnostics(analysis, selfChecks)
    lines = {
        sprintf(['Phase: psi=phi_i-Phi_i; psi_phys=psi+delta.  ' ...
                 'delta/pi=%.9g, psi_d/pi=%.9g, nu=%.9g, P=%.9g'], ...
                analysis.delta / pi, analysis.psiTarget / pi, ...
                analysis.nu, analysis.power)
        'Targets: psi_d=0 -> collective mode in phase; psi_d=-delta -> actual sensor in phase.'
        sprintf('Optimal: %s', analysis.optimal.status)
        sprintf(['  residual(eq)=%.3e, residual(energy)=%.3e, kappa=%.9g, ' ...
                 'A=%.9g, identity residual=%.3e'], ...
                analysis.optimal.equilibriumResidual, analysis.optimal.energyResidual, ...
                analysis.optimal.kappa, analysis.optimal.lockingHalfWidth, ...
                analysis.optimal.targetLockingIdentityResidual)
        sprintf('Best sinusoid: %s', analysis.bestSinusoid.status)
        sprintf(['  tau/pi=%.9g, residual(eq)=%.3e, residual(energy)=%.3e, ' ...
                 'kappa=%.9g, A=%.9g, identity residual=%.3e'], ...
                analysis.bestTau / pi, analysis.bestSinusoid.equilibriumResidual, ...
                analysis.bestSinusoid.energyResidual, analysis.bestSinusoid.kappa, ...
                analysis.bestSinusoid.lockingHalfWidth, ...
                analysis.bestSinusoid.targetLockingIdentityResidual)
        sprintf('Manual sinusoid: %s', analysis.manual.status)
        sprintf(['  tau/pi=%.9g, residual(eq)=%.3e, residual(energy)=%.3e, ' ...
                 'kappa(at target)=%.9g, A=%.9g'], ...
                analysis.manualTau / pi, analysis.manual.equilibriumResidual, ...
                analysis.manual.energyResidual, analysis.manual.kappa, ...
                analysis.manual.lockingHalfWidth)
        sprintf(['Internal checks: %s  [CSV default: kappaOpt=%.9g, ' ...
                 'kappaSin=%.9g, tau/pi=%.9g; constant-a relative error=%.3e]'], ...
                passFail(selfChecks.allPassed), ...
                selfChecks.defaultCsvBenchmark.kappaOptimal, ...
                selfChecks.defaultCsvBenchmark.kappaBestSinusoid, ...
                selfChecks.defaultCsvBenchmark.bestTauOverPi, ...
                selfChecks.syntheticConstantProfile.relativeProfileError)
        };
end

% =========================================================================
% Small numerical helpers and validators
% =========================================================================

function q = deterministicOrthogonalUnit(phi, constraint, tolerances)
    candidates = {ones(size(phi))};
    for harmonic = 1:8
        candidates{end + 1} = sin(harmonic .* phi); %#ok<AGROW>
        candidates{end + 1} = cos(harmonic .* phi); %#ok<AGROW>
    end

    constraint = constraint(:);
    constraintNorm = mean(constraint .* constraint);
    for index = 1:numel(candidates)
        qCandidate = candidates{index}(:);
        if constraintNorm > tolerances.abs
            % A second pass removes roundoff left by the first projection.
            for pass = 1:2
                qCandidate = qCandidate - ...
                    (mean(qCandidate .* constraint) / constraintNorm) .* constraint;
            end
        end
        candidateNorm = sqrt(mean(qCandidate .* qCandidate));
        if candidateNorm > sqrt(tolerances.abs)
            q = qCandidate / candidateNorm;
            return;
        end
    end
    error('target_equilibrium_optimal_z_analysis:DegenerateBasis', ...
        'Could not construct an energy-completing direction.');
end

function tolerances = defaultTolerances()
    tolerances = struct();
    tolerances.abs = 1e-12;
    tolerances.rel = 1e-10;
    tolerances.angle = 1e-9;
end

function value = scaledTolerance(tolerances, quantities)
    finiteQuantities = abs(quantities(isfinite(quantities)));
    if isempty(finiteQuantities)
        scale = 1;
    else
        scale = max([1; finiteQuantities(:)]);
    end
    value = tolerances.abs + tolerances.rel * scale;
end

function wrapped = wrapAngle(angleValue)
    wrapped = atan2(sin(angleValue), cos(angleValue));
end

function distance = angularDistance(firstAngle, secondAngle)
    distance = abs(wrapAngle(firstAngle - secondAngle));
end

function textValue = passFail(condition)
    if condition
        textValue = 'PASS';
    else
        textValue = 'FAIL';
    end
end

function valid = isFiniteRealScalar(value)
    valid = isnumeric(value) && isscalar(value) && isreal(value) && isfinite(value);
end

function valid = isNonnegativeFiniteRealScalar(value)
    valid = isFiniteRealScalar(value) && value >= 0;
end

function valid = isValidGridSize(value)
    valid = isFiniteRealScalar(value) && value >= 32 && value == floor(value);
end

function valid = isLogicalScalar(value)
    valid = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
        isreal(value) && isfinite(value) && (value == 0 || value == 1);
end

function writePrcSnippetFile(analysis, maxHarmonics, outputPath)
    phi = analysis.phi;
    zOpt = analysis.optimal.z;
    if any(~isfinite(zOpt))
        error('target_equilibrium_optimal_z_analysis:ExportNonfinite', ...
            'Optimal z profile is nonfinite or unavailable for export.');
    end

    a_coeff = zeros(maxHarmonics + 1, 1);
    b_coeff = zeros(maxHarmonics + 1, 1);
    a_coeff(1) = mean(zOpt);
    b_coeff(1) = 0;
    for n = 1:maxHarmonics
        a_coeff(n + 1) = 2 * mean(zOpt .* cos(n * phi));
        b_coeff(n + 1) = 2 * mean(zOpt .* sin(n * phi));
    end

    folder = fileparts(outputPath);
    if ~isempty(folder) && ~exist(folder, 'dir')
        mkdir(folder);
    end

    fid = fopen(outputPath, 'w', 'n', 'UTF-8');
    if fid == -1
        error('target_equilibrium_optimal_z_analysis:FileOpenError', ...
            'Could not open file for writing: %s', outputPath);
    end
    cleanup = onCleanup(@() fclose(fid));

    psiTargetOverPi = analysis.psiTarget / pi;
    psiPhysTarget = wrapAngle(analysis.psiTarget + analysis.delta);
    psiPhysTargetOverPi = psiPhysTarget / pi;

    fprintf(fid, '# Auto-generated from target_equilibrium_optimal_z_analysis\n');
    fprintf(fid, '# Target equilibrium point (mode phase difference): psi_d = %.9g rad (psi_d/pi = %.9g)\n', ...
        analysis.psiTarget, psiTargetOverPi);
    fprintf(fid, '# Target equilibrium point (physical sensor phase): psi_phys_d = %.9g rad (psi_phys_d/pi = %.9g)\n', ...
        psiPhysTarget, psiPhysTargetOverPi);
    fprintf(fid, '# Parameters: nu = %.9g, power = %.9g, delta/pi = %.9g\n#\n', ...
        analysis.nu, analysis.power, analysis.delta / pi);
    fprintf(fid, 'target_psi_d_rad = %.10f\n', analysis.psiTarget);
    fprintf(fid, 'target_psi_d_over_pi = %.10f\n', psiTargetOverPi);
    fprintf(fid, 'target_psi_phys_d_rad = %.10f\n', psiPhysTarget);
    fprintf(fid, 'target_psi_phys_d_over_pi = %.10f\n\n', psiPhysTargetOverPi);

    fprintf(fid, 'prc_harmonics = %d\n', maxHarmonics);
    fprintf(fid, 'prc_a = [0.0] * (prc_harmonics + 1)\n');
    fprintf(fid, 'prc_b = [0.0] * (prc_harmonics + 1)\n\n');

    for n = 0:maxHarmonics
        fprintf(fid, 'prc_a[%d] = %.10f\n', n, a_coeff(n + 1));
        fprintf(fid, 'prc_b[%d] = %.10f\n', n, b_coeff(n + 1));
    end
end
