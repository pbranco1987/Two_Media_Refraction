% =========================================================================
% PARAMETER SWEEP: n2 (soil layer 1), n3 (soil layer 2), depth1 (DEM2)
% =========================================================================
% Automated sweep over all combinations of three physical parameters:
%   - n2: refractive index of soil layer 1 (medium 2)
%   - n3: refractive index of soil layer 2 (medium 3)
%   - depth1: depth of the second DEM interface below the first (meters)
%
% For each (n2, n3, depth1) combination, this script:
%   1. Runs the hybrid backprojection (bp_hybrid) to produce a 3D image
%   2. Runs the quality analysis (analyze_bp_quality_fn) to compute 35+ metrics
%   3. Stores all metrics in a summary table and saves incrementally
%
% Optimization: when n2 == n3, there is no refraction at the soil1-soil2
% interface (both layers have the same refractive index), making depth1
% irrelevant. These redundant trials are skipped: only one depth1 is
% computed, and results are copied to all other depth1 entries. This
% reduces the total from 9x9x8 = 648 to 585 computed trials.
%
% Results are saved incrementally after every trial (both .mat and .csv),
% allowing progress monitoring while the sweep is still running.
%
% Outputs:
%   - Per-combination folders under 'sweep_results/' with /mat and /plots
%   - sweep_results/sweep_summary.mat  (MATLAB table + parameter vectors)
%   - sweep_results/sweep_summary.csv  (same data in CSV format)
%   - Summary heatmap plots (n2 vs n3 heatmaps for each depth1 value)
%   - Cross-depth summary plot (best sharpness vs depth1)
% =========================================================================

clc; clear; close all;

%% ======================== DEFINE PARAMETER RANGES =======================
% Refractive indices: 9 evenly spaced values from 3.2 to 4.0 (step = 0.1)
% These represent typical soil permittivity values for GPR applications.
n2_values = linspace(3.2, 4.0, 9);   % Soil layer 1 refractive index (9 values)
n3_values = linspace(3.2, 4.0, 9);   % Soil layer 2 refractive index (9 values)

% Depth of the second interface (DEM2) below the first (DEM1), in meters.
% 15 values from 20 cm to 1.60 m with 10 cm spacing covers the expected
% range of the soil layer 1 thickness at fine resolution.
depth1_values = 0.20:0.10:1.60;

%% ======================== SETUP OUTPUT DIRECTORY ========================
baseOutputDir = fullfile(pwd, 'sweep_results');
if ~exist(baseOutputDir, 'dir')
    mkdir(baseOutputDir);
end

%% ======================== BUILD FULL CARTESIAN PRODUCT ===================
% Create the full 3D grid of parameter combinations using ndgrid.
% ndgrid ordering: n2 varies along dim 1, n3 along dim 2, depth1 along dim 3.
% After flattening, depth1 varies fastest, then n3, then n2.
[N2_grid, N3_grid, D1_grid] = ndgrid(n2_values, n3_values, depth1_values);
n2_list     = N2_grid(:);     % Flattened list of all n2 values [nCombinations x 1]
n3_list     = N3_grid(:);     % Flattened list of all n3 values [nCombinations x 1]
depth1_list = D1_grid(:);     % Flattened list of all depth1 values [nCombinations x 1]
nCombinations = numel(n2_list);  % Total number of parameter combinations

% --- n2 == n3 skip optimization ---
% When n2 equals n3, both soil layers have identical refractive indices,
% so there is no refraction at the DEM2 interface. The depth of DEM2 has
% no physical effect, making all depth1 values produce identical results.
% We compute only the first depth1 and copy results to all other entries.
sameN = (abs(n2_list - n3_list) < 1e-10);                  % Logical: n2 == n3
refDepth1 = depth1_values(1);                               % Reference depth1 used for computation
skipMask = sameN & (abs(depth1_list - refDepth1) > 1e-10);  % True = skip (redundant trial)
nSkipped = sum(skipMask);
nToRun   = nCombinations - nSkipped;

% Build a lookup table: for each skipped trial, store the index of the
% reference trial (same n2==n3 with depth1 == refDepth1) to copy from.
% The ndgrid ordering guarantees that the reference trial (first depth1)
% is always processed before any of its skipped copies.
refTrialIdx = zeros(nCombinations, 1);   % 0 = not skipped, >0 = index of reference trial
for k = 1:nCombinations
    if skipMask(k)
        % Find the reference trial: same n2, same n3, depth1 == refDepth1
        match = find(abs(n2_list - n2_list(k)) < 1e-10 & ...
                     abs(n3_list - n3_list(k)) < 1e-10 & ...
                     abs(depth1_list - refDepth1) < 1e-10);
        refTrialIdx(k) = match(1);
    end
end

fprintf('============================================\n');
fprintf('       PARAMETER SWEEP (3D Cartesian)       \n');
fprintf('============================================\n');
fprintf('n2 values (%d): %s\n', numel(n2_values), mat2str(n2_values, 4));
fprintf('n3 values (%d): %s\n', numel(n3_values), mat2str(n3_values, 4));
fprintf('depth1 values (%d): %s\n', numel(depth1_values), mat2str(depth1_values, 4));
fprintf('Full cartesian product: %d x %d x %d = %d\n', ...
    numel(n2_values), numel(n3_values), numel(depth1_values), nCombinations);
fprintf('Skipping %d redundant trials (n2 == n3, depth1 irrelevant)\n', nSkipped);
fprintf('Trials to compute: %d\n\n', nToRun);

%% ======================== PRE-ALLOCATE RESULTS ==========================
% Pre-allocate all metric storage arrays with NaN. Each array has one
% element per parameter combination. NaN values indicate trials that
% have not yet been computed or that failed during execution.
N = nCombinations;

% Trial identification columns
trialIdx      = (1:N)';           % Sequential trial number [1..N]
n2_col        = n2_list;          % n2 value for each trial
n3_col        = n3_list;          % n3 value for each trial
depth1_col    = depth1_list;      % depth1 value for each trial
compTime      = NaN(N, 1);       % Computation time (seconds)
status_col    = repmat({'pending'}, N, 1);  % Trial status string
folder_col    = repmat({''}, N, 1);         % Output folder name

% --- Peak location and magnitude ---
peakMag       = NaN(N, 1);       % Maximum absolute amplitude in the volume
peakX         = NaN(N, 1);       % X-coordinate of peak (meters)
peakY         = NaN(N, 1);       % Y-coordinate of peak (meters)
peakZ         = NaN(N, 1);       % Z-coordinate of peak (meters)

% --- Entropy (energy dispersion) ---
entropy_amp_norm   = NaN(N, 1);  % Amplitude-based normalized entropy [0,1]
entropy_int_norm   = NaN(N, 1);  % Intensity-based normalized entropy [0,1]
entropy_amp_raw    = NaN(N, 1);  % Amplitude-based raw Shannon entropy (nats)
entropy_int_raw    = NaN(N, 1);  % Intensity-based raw Shannon entropy (nats)

% --- Sharpness (peak dominance) ---
sharp_peak_mean    = NaN(N, 1);  % Peak / mean amplitude ratio
sharp_peak_median  = NaN(N, 1);  % Peak / median amplitude ratio
sharp_peak_rms     = NaN(N, 1);  % Peak / RMS amplitude ratio

% --- Resolution -3 dB (half-power width) ---
res3_x             = NaN(N, 1);  % X-direction -3dB width (meters)
res3_y             = NaN(N, 1);  % Y-direction -3dB width (meters)
res3_z             = NaN(N, 1);  % Z-direction -3dB width (meters)
res3_vol           = NaN(N, 1);  % Resolution cell volume (m^3)
res3_equiv_diam    = NaN(N, 1);  % Equivalent sphere diameter (meters)

% --- Resolution -6 dB (quarter-power width) ---
res6_x             = NaN(N, 1);  % X-direction -6dB width (meters)
res6_y             = NaN(N, 1);  % Y-direction -6dB width (meters)
res6_z             = NaN(N, 1);  % Z-direction -6dB width (meters)

% --- Contrast (target vs background) ---
TBR_dB_col         = NaN(N, 1);  % Target-to-Background Ratio (dB)
SCR_dB_col         = NaN(N, 1);  % Signal-to-Clutter Ratio (dB)
CNR_col            = NaN(N, 1);  % Contrast-to-Noise Ratio (linear)

% --- Sidelobes ---
PSLR_dB_col        = NaN(N, 1);  % Peak Sidelobe Level Ratio (dB)
ISLR_dB_col        = NaN(N, 1);  % Integrated Sidelobe Level Ratio (dB)

% --- Statistical distribution metrics ---
kurtosis_col       = NaN(N, 1);  % Kurtosis (peakedness, >3 = leptokurtic)
skewness_col       = NaN(N, 1);  % Skewness (asymmetry, >0 = right tail)
CV_col             = NaN(N, 1);  % Coefficient of Variation (std/mean)
gini_col           = NaN(N, 1);  % Gini coefficient (inequality, 0-1)

% --- Energy concentration ---
pct_vox_50         = NaN(N, 1);  % % of voxels containing 50% of total energy
pct_vox_90         = NaN(N, 1);  % % of voxels containing 90% of total energy
pct_vox_99         = NaN(N, 1);  % % of voxels containing 99% of total energy
eff_scatterers     = NaN(N, 1);  % Effective number of scatterers (participation ratio)
eff_scatterers_pct = NaN(N, 1);  % Effective scatterers as % of total voxels

% --- Gradient (edge sharpness) ---
grad_mean          = NaN(N, 1);  % Global mean gradient magnitude
grad_mean_peak     = NaN(N, 1);  % Mean gradient near the peak
grad_max           = NaN(N, 1);  % Maximum gradient magnitude

% --- PSF shape ---
psf_asym_max       = NaN(N, 1);  % Maximum PSF asymmetry ratio (1 = isotropic)

% --- Dynamic range ---
dyn_range_dB       = NaN(N, 1);  % Peak / noise floor in dB

% Column names for the summary table (defined once, reused in all saves)
sweepVarNames = { ...
    'Trial', 'n2', 'n3', 'depth1_m', 'ComputationTime_s', ...
    'PeakMagnitude', 'PeakX', 'PeakY', 'PeakZ', ...
    'EntropyAmpNorm', 'EntropyIntNorm', 'EntropyAmpRaw', 'EntropyIntRaw', ...
    'SharpnessPkMean', 'SharpnessPkMedian', 'SharpnessPkRMS', ...
    'Res3dB_X', 'Res3dB_Y', 'Res3dB_Z', 'Res3dB_Volume', 'Res3dB_EquivDiam', ...
    'Res6dB_X', 'Res6dB_Y', 'Res6dB_Z', ...
    'TBR_dB', 'SCR_dB', 'CNR', ...
    'PSLR_dB', 'ISLR_dB', ...
    'Kurtosis', 'Skewness', 'CV', 'Gini', ...
    'PctVox50', 'PctVox90', 'PctVox99', 'EffScatterers', 'EffScatterersPct', ...
    'GradMean', 'GradMeanPeak', 'GradMax', ...
    'PSF_AsymMax', ...
    'DynamicRange_dB', ...
    'Status', 'Folder'};

% Paths for the live-updated summary files
summaryMatFile = fullfile(baseOutputDir, 'sweep_summary.mat');
summaryCsvFile = fullfile(baseOutputDir, 'sweep_summary.csv');

%% ======================== RUN SWEEP =====================================
% Main loop: iterate over all parameter combinations, skipping redundant
% n2==n3 trials and computing the rest.
sweepStartTime = tic;
nComputed = 0;   % Counter for actually computed (non-skipped) trials

for k = 1:nCombinations
    n2_val     = n2_list(k);
    n3_val     = n3_list(k);
    depth1_val = depth1_list(k);

    % Folder name encodes all three parameters
    folderName = sprintf('n2_%.2f_n3_%.2f_d1_%.2f', n2_val, n3_val, depth1_val);
    trialOutputDir = fullfile(baseOutputDir, folderName);
    folder_col{k} = folderName;

    % ---- Skip redundant trial when n2 == n3 ----
    if skipMask(k)
        ref = refTrialIdx(k);
        % Copy all metrics from the reference trial (same n2==n3, first depth1)
        compTime(k)           = compTime(ref);
        peakMag(k)            = peakMag(ref);
        peakX(k)              = peakX(ref);
        peakY(k)              = peakY(ref);
        peakZ(k)              = peakZ(ref);
        entropy_amp_norm(k)   = entropy_amp_norm(ref);
        entropy_int_norm(k)   = entropy_int_norm(ref);
        entropy_amp_raw(k)    = entropy_amp_raw(ref);
        entropy_int_raw(k)    = entropy_int_raw(ref);
        sharp_peak_mean(k)    = sharp_peak_mean(ref);
        sharp_peak_median(k)  = sharp_peak_median(ref);
        sharp_peak_rms(k)     = sharp_peak_rms(ref);
        res3_x(k)             = res3_x(ref);
        res3_y(k)             = res3_y(ref);
        res3_z(k)             = res3_z(ref);
        res3_vol(k)           = res3_vol(ref);
        res3_equiv_diam(k)    = res3_equiv_diam(ref);
        res6_x(k)             = res6_x(ref);
        res6_y(k)             = res6_y(ref);
        res6_z(k)             = res6_z(ref);
        TBR_dB_col(k)         = TBR_dB_col(ref);
        SCR_dB_col(k)         = SCR_dB_col(ref);
        CNR_col(k)            = CNR_col(ref);
        PSLR_dB_col(k)        = PSLR_dB_col(ref);
        ISLR_dB_col(k)        = ISLR_dB_col(ref);
        kurtosis_col(k)       = kurtosis_col(ref);
        skewness_col(k)       = skewness_col(ref);
        CV_col(k)             = CV_col(ref);
        gini_col(k)           = gini_col(ref);
        pct_vox_50(k)         = pct_vox_50(ref);
        pct_vox_90(k)         = pct_vox_90(ref);
        pct_vox_99(k)         = pct_vox_99(ref);
        eff_scatterers(k)     = eff_scatterers(ref);
        eff_scatterers_pct(k) = eff_scatterers_pct(ref);
        grad_mean(k)          = grad_mean(ref);
        grad_mean_peak(k)     = grad_mean_peak(ref);
        grad_max(k)           = grad_max(ref);
        psf_asym_max(k)       = psf_asym_max(ref);
        dyn_range_dB(k)       = dyn_range_dB(ref);
        status_col{k}         = status_col{ref};
        folder_col{k}         = folder_col{ref};   % point to same folder

        fprintf('=== Trial %d/%d: n2=%.2f, n3=%.2f, depth1=%.2f m -> SKIPPED (n2==n3, copied from trial %d) ===\n', ...
            k, nCombinations, n2_val, n3_val, depth1_val, ref);

    else
        % ---- Actually compute this trial ----
        nComputed = nComputed + 1;
        if ~exist(trialOutputDir, 'dir')
            mkdir(trialOutputDir);
        end

        fprintf('=== Trial %d/%d [compute %d/%d]: n2=%.2f, n3=%.2f, depth1=%.2f m ===\n', ...
            k, nCombinations, nComputed, nToRun, n2_val, n3_val, depth1_val);

        try
            % ---- Step 1: Run backprojection ----
            [Img, tTotal] = bp_hybrid(n2_val, n3_val, depth1_val, trialOutputDir);
            compTime(k) = tTotal;

            % Load grid variables from saved .mat
            matPath = fullfile(trialOutputDir, 'mat', 'bp_hybrid_results.mat');
            loaded = load(matPath, 'xGrid', 'yGrid', 'zGrid');
            xGrid = loaded.xGrid;
            yGrid = loaded.yGrid;
            zGrid = loaded.zGrid;

            % ---- Step 2: Run full quality analysis ----
            R = analyze_bp_quality_fn(Img, xGrid, yGrid, zGrid, ...
                    n2_val, n3_val, depth1_val, trialOutputDir);

            % ---- Step 3: Extract all metrics into columns ----
            % Peak
            peakMag(k) = R.peak.value;
            peakX(k)   = R.peak.location(1);
            peakY(k)   = R.peak.location(2);
            peakZ(k)   = R.peak.location(3);

            % Entropy
            entropy_amp_norm(k) = R.entropy.normalized;
            entropy_int_norm(k) = R.entropy.intensity_normalized;
            entropy_amp_raw(k)  = R.entropy.shannon;
            entropy_int_raw(k)  = R.entropy.intensity;

            % Sharpness
            sharp_peak_mean(k)   = R.sharpness.peak_mean;
            sharp_peak_median(k) = R.sharpness.peak_median;
            sharp_peak_rms(k)    = R.sharpness.peak_rms;

            % Resolution -3 dB
            res3_x(k)          = R.resolution_3dB.x;
            res3_y(k)          = R.resolution_3dB.y;
            res3_z(k)          = R.resolution_3dB.z;
            res3_vol(k)        = R.resolution_3dB.volume;
            res3_equiv_diam(k) = R.resolution_3dB.equiv_diameter;

            % Resolution -6 dB
            res6_x(k) = R.resolution_6dB.x;
            res6_y(k) = R.resolution_6dB.y;
            res6_z(k) = R.resolution_6dB.z;

            % Contrast
            TBR_dB_col(k) = R.contrast.TBR_dB;
            SCR_dB_col(k) = R.contrast.SCR_dB;
            CNR_col(k)    = R.contrast.CNR;

            % Sidelobes
            PSLR_dB_col(k) = R.sidelobes.PSLR_dB;
            ISLR_dB_col(k) = R.sidelobes.ISLR_dB;

            % Statistical
            kurtosis_col(k) = R.statistical.kurtosis;
            skewness_col(k) = R.statistical.skewness;
            CV_col(k)       = R.statistical.CV;
            gini_col(k)     = R.statistical.gini;

            % Energy concentration
            pct_vox_50(k)         = R.energy.pct_voxels_50;
            pct_vox_90(k)         = R.energy.pct_voxels_90;
            pct_vox_99(k)         = R.energy.pct_voxels_99;
            eff_scatterers(k)     = R.energy.effective_scatterers;
            eff_scatterers_pct(k) = R.energy.effective_scatterers_pct;

            % Gradient
            grad_mean(k)      = R.gradient.mean;
            grad_mean_peak(k) = R.gradient.mean_peak;
            grad_max(k)       = R.gradient.max;

            % PSF
            psf_asym_max(k) = R.psf.asymmetry_max;

            % Dynamic range
            dyn_range_dB(k) = R.dynamic_range_dB;

            status_col{k} = 'success';
            fprintf('  -> Done in %.1fs | Peak=%.4f at (%.2f, %.2f, %.2f)\n', ...
                tTotal, peakMag(k), peakX(k), peakY(k), peakZ(k));

        catch ME
            status_col{k} = sprintf('error: %s', ME.message);
            fprintf('  -> FAILED: %s\n', ME.message);

            % Save error details
            errorFile = fullfile(trialOutputDir, 'error_log.txt');
            fid = fopen(errorFile, 'w');
            fprintf(fid, 'Error for n2=%.2f, n3=%.2f, depth1=%.2f\n', n2_val, n3_val, depth1_val);
            fprintf(fid, 'Message: %s\n', ME.message);
            fprintf(fid, 'Identifier: %s\n', ME.identifier);
            for s = 1:numel(ME.stack)
                fprintf(fid, '  File: %s, Line: %d, Function: %s\n', ...
                    ME.stack(s).file, ME.stack(s).line, ME.stack(s).name);
            end
            fclose(fid);
        end

        % Free figures between trials
        close all;
    end

    % ---- INCREMENTAL SAVE ----
    % Save the summary table after every trial (both computed and skipped).
    % This allows monitoring progress and partial results while the sweep runs.
    elapsedSoFar = toc(sweepStartTime);
    sweepTable = table( ...
        trialIdx, n2_col, n3_col, depth1_col, compTime, ...
        peakMag, peakX, peakY, peakZ, ...
        entropy_amp_norm, entropy_int_norm, entropy_amp_raw, entropy_int_raw, ...
        sharp_peak_mean, sharp_peak_median, sharp_peak_rms, ...
        res3_x, res3_y, res3_z, res3_vol, res3_equiv_diam, ...
        res6_x, res6_y, res6_z, ...
        TBR_dB_col, SCR_dB_col, CNR_col, ...
        PSLR_dB_col, ISLR_dB_col, ...
        kurtosis_col, skewness_col, CV_col, gini_col, ...
        pct_vox_50, pct_vox_90, pct_vox_99, eff_scatterers, eff_scatterers_pct, ...
        grad_mean, grad_mean_peak, grad_max, ...
        psf_asym_max, dyn_range_dB, ...
        status_col, folder_col, ...
        'VariableNames', sweepVarNames);
    save(summaryMatFile, 'sweepTable', 'n2_values', 'n3_values', 'depth1_values', ...
        'elapsedSoFar', 'k', 'nCombinations', 'nToRun', 'nSkipped');
    writetable(sweepTable, summaryCsvFile);
    fprintf('  [Saved progress: %d/%d done | %.1f min elapsed]\n\n', ...
        k, nCombinations, elapsedSoFar / 60);
end

totalSweepTime = toc(sweepStartTime);

%% ======================== FINAL SUMMARY TABLE ===========================
sweepTable = table( ...
    trialIdx, n2_col, n3_col, depth1_col, compTime, ...
    peakMag, peakX, peakY, peakZ, ...
    entropy_amp_norm, entropy_int_norm, entropy_amp_raw, entropy_int_raw, ...
    sharp_peak_mean, sharp_peak_median, sharp_peak_rms, ...
    res3_x, res3_y, res3_z, res3_vol, res3_equiv_diam, ...
    res6_x, res6_y, res6_z, ...
    TBR_dB_col, SCR_dB_col, CNR_col, ...
    PSLR_dB_col, ISLR_dB_col, ...
    kurtosis_col, skewness_col, CV_col, gini_col, ...
    pct_vox_50, pct_vox_90, pct_vox_99, eff_scatterers, eff_scatterers_pct, ...
    grad_mean, grad_mean_peak, grad_max, ...
    psf_asym_max, dyn_range_dB, ...
    status_col, folder_col, ...
    'VariableNames', sweepVarNames);

fprintf('\n========== SWEEP SUMMARY ==========\n');
disp(sweepTable(:, 1:12));  % print compact view (first 12 columns)

save(summaryMatFile, 'sweepTable', 'n2_values', 'n3_values', 'depth1_values', ...
    'totalSweepTime');
fprintf('Summary saved: %s\n', summaryMatFile);

writetable(sweepTable, summaryCsvFile);
fprintf('Summary CSV saved: %s\n', summaryCsvFile);

%% ======================== SUMMARY VISUALIZATIONS ========================
% Generate n2-vs-n3 heatmaps for each depth1 value and each metric.
% These heatmaps provide a visual overview of how image quality varies
% across the parameter space, making it easy to identify optimal regions.
successMask = strcmp(sweepTable.Status, 'success');

if ~any(successMask)
    fprintf('No successful trials to plot.\n');
else
    successTable = sweepTable(successMask, :);

    % Metrics to plot as heatmaps (column name, display title, colormap)
    hmDefs = {
        'SharpnessPkMean',     'Sharpness (Peak/Mean)',      'jet'
        'EntropyAmpNorm',      'Entropy (Amplitude, Norm)',  'jet'
        'EntropyIntNorm',      'Entropy (Intensity, Norm)',  'jet'
        'Res3dB_X',            'Resolution X -3dB (m)',      'jet'
        'Res3dB_Y',            'Resolution Y -3dB (m)',      'jet'
        'Res3dB_Z',            'Resolution Z -3dB (m)',      'jet'
        'Res3dB_Volume',       'Resolution Volume (m^3)',    'jet'
        'SCR_dB',              'SCR (dB)',                   'jet'
        'PSLR_dB',             'PSLR (dB)',                  'jet'
        'ISLR_dB',             'ISLR (dB)',                  'jet'
        'CNR',                 'CNR',                        'jet'
        'Gini',                'Gini Coefficient',           'jet'
        'PctVox90',            '% Voxels for 90% Energy',    'jet'
        'DynamicRange_dB',     'Dynamic Range (dB)',         'hot'
        'ComputationTime_s',   'Computation Time (s)',       'hot'
    };

    for di = 1:numel(depth1_values)
        d1 = depth1_values(di);
        dMask = abs(successTable.depth1_m - d1) < 1e-6;
        subT = successTable(dMask, :);
        if isempty(subT), continue; end

        for mi = 1:size(hmDefs, 1)
            colName  = hmDefs{mi, 1};
            colTitle = hmDefs{mi, 2};
            cmap     = hmDefs{mi, 3};

            figure('Color', 'w', 'Position', [100 100 650 520], 'Visible', 'off');
            mat = NaN(numel(n3_values), numel(n2_values));
            for row = 1:height(subT)
                ci = find(abs(n2_values - subT.n2(row)) < 1e-6);
                ri = find(abs(n3_values - subT.n3(row)) < 1e-6);
                if ~isempty(ci) && ~isempty(ri)
                    mat(ri, ci) = subT.(colName)(row);
                end
            end
            imagesc(n2_values, n3_values, mat);
            axis xy; colorbar; colormap(cmap);
            xlabel('n2 (soil layer 1)');
            ylabel('n3 (soil layer 2)');
            title(sprintf('%s | depth1 = %.2f m', colTitle, d1));

            fname = sprintf('heatmap_%s_d1_%.2f', lower(colName), d1);
            saveas(gcf, fullfile(baseOutputDir, [fname '.png']));
            saveas(gcf, fullfile(baseOutputDir, [fname '.fig']));
            close(gcf);
        end
    end

    % --- Cross-depth summary: find the optimal (n2, n3) per depth1 ---
    % For each depth1 value, identify which (n2, n3) combination yields
    % the highest sharpness. This reveals how the optimal parameters
    % shift as the layer thickness changes.
    figure('Color', 'w', 'Position', [100 100 700 450], 'Visible', 'off');
    bestSharp = NaN(numel(depth1_values), 1);
    bestN2    = NaN(numel(depth1_values), 1);
    bestN3    = NaN(numel(depth1_values), 1);
    for di = 1:numel(depth1_values)
        d1 = depth1_values(di);
        dMask = abs(successTable.depth1_m - d1) < 1e-6;
        subT = successTable(dMask, :);
        if isempty(subT), continue; end
        [bestSharp(di), idx] = max(subT.SharpnessPkMean);
        bestN2(di) = subT.n2(idx);
        bestN3(di) = subT.n3(idx);
    end

    subplot(2,1,1);
    bar(depth1_values, bestSharp, 'FaceColor', [0.2 0.6 0.9]);
    xlabel('depth1 (m)'); ylabel('Best Sharpness');
    title('Best Sharpness vs DEM2 Depth'); grid on;

    subplot(2,1,2);
    plot(depth1_values, bestN2, '-o', 'DisplayName', 'n2', 'LineWidth', 1.5);
    hold on;
    plot(depth1_values, bestN3, '-s', 'DisplayName', 'n3', 'LineWidth', 1.5);
    xlabel('depth1 (m)'); ylabel('Refractive index');
    title('Optimal n2, n3 at Best Sharpness per depth1');
    legend('Location', 'best'); grid on;

    saveas(gcf, fullfile(baseOutputDir, 'summary_best_sharpness_vs_depth.png'));
    saveas(gcf, fullfile(baseOutputDir, 'summary_best_sharpness_vs_depth.fig'));
    close(gcf);
end

%% ======================== FINAL REPORT ==================================
fprintf('\n============================================\n');
fprintf('          SWEEP COMPLETE                    \n');
fprintf('============================================\n');
fprintf('Total combinations (cartesian product): %d\n', nCombinations);
fprintf('Skipped (n2==n3 redundant depth1): %d\n', nSkipped);
fprintf('Actually computed: %d\n', nToRun);
fprintf('Successful: %d\n', sum(successMask));
fprintf('Failed: %d\n', sum(~successMask & ~skipMask));
fprintf('Total sweep time: %.1f min (%.1f hrs)\n', totalSweepTime/60, totalSweepTime/3600);
fprintf('Results directory: %s\n', baseOutputDir);
fprintf('============================================\n');

