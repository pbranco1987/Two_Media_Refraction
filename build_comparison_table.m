% =========================================================================
% BUILD_COMPARISON_TABLE
% =========================================================================
% Post-processing script that builds a comprehensive comparison table of
% all parameter sweep trials, including every evaluation metric.
%
% This script can work in two modes:
%   Mode 1: Load from sweep_summary.mat (fast, requires completed sweep)
%   Mode 2: Scan individual trial folders and load bp_analysis_results.mat
%           files (fallback, works even if sweep was interrupted)
%
% The output is a clean, analysis-ready dataset with:
%   - One row per (n2, n3, depth1) combination
%   - All 35+ quality metrics in separate columns
%   - Redundant n2==n3 rows collapsed (deduplicated)
%   - Composite quality score for ranking parameter combinations
%   - Per-metric normalized scores (0 = worst, 1 = best) for transparency
%
% Output files:
%   sweep_results/comparison_all_trials.csv       All trials, full table
%   sweep_results/comparison_unique_trials.csv    Deduplicated (n2==n3 collapsed)
%   sweep_results/comparison_ranked.csv           Ranked by composite score
%   sweep_results/comparison_*.mat                Same data as .mat for MATLAB
%
% The composite score is a weighted average of 19 normalized metrics:
%   - Metrics where higher is better: sharpness, SCR, TBR, CNR, dynamic range, etc.
%   - Metrics where lower is better: entropy, resolution, PSLR, ISLR, etc.
%   - Weights emphasize sharpness (1.5x), resolution volume (1.2x), and SCR (1.2x)
% =========================================================================

clc; clear; close all;

baseOutputDir = fullfile(pwd, 'sweep_results');
summaryMatFile = fullfile(baseOutputDir, 'sweep_summary.mat');

%% ======================== LOAD DATA ======================================
% Try Mode 1 first (fast path); fall back to Mode 2 if summary doesn't exist.
if exist(summaryMatFile, 'file')
    % --- Mode 1: Load the pre-built summary table from the sweep ---
    % This is the fast path; the sweep script saves this file incrementally.
    fprintf('Loading from sweep_summary.mat ...\n');
    S = load(summaryMatFile);
    sweepTable = S.sweepTable;
    n2_values = S.n2_values;
    n3_values = S.n3_values;
    depth1_values = S.depth1_values;
else
    % --- Mode 2: Scan individual trial folders ---
    % Fallback when sweep_summary.mat doesn't exist (e.g., sweep was interrupted
    % or results were transferred without the summary file).
    fprintf('sweep_summary.mat not found. Scanning trial folders ...\n');

    n2_values = linspace(3.2, 4.0, 9);
    n3_values = linspace(3.2, 4.0, 9);
    depth1_values = 0.20:0.10:1.60;  % 15 values

    [N2_grid, N3_grid, D1_grid] = ndgrid(n2_values, n3_values, depth1_values);
    n2_list     = N2_grid(:);
    n3_list     = N3_grid(:);
    depth1_list = D1_grid(:);
    nTotal      = numel(n2_list);

    % Pre-allocate cell array of results structs
    allResults = cell(nTotal, 1);
    compTimes  = NaN(nTotal, 1);
    statusList = repmat({'not_found'}, nTotal, 1);

    for k = 1:nTotal
        n2_val     = n2_list(k);
        n3_val     = n3_list(k);
        depth1_val = depth1_list(k);

        folderName = sprintf('n2_%.2f_n3_%.2f_d1_%.2f', n2_val, n3_val, depth1_val);
        analysisFile = fullfile(baseOutputDir, folderName, 'mat', 'bp_analysis_results.mat');
        bpFile       = fullfile(baseOutputDir, folderName, 'mat', 'bp_hybrid_results.mat');

        % For n2==n3 cases, check the reference folder (first depth1)
        if abs(n2_val - n3_val) < 1e-10 && abs(depth1_val - depth1_values(1)) > 1e-10
            refFolder = sprintf('n2_%.2f_n3_%.2f_d1_%.2f', n2_val, n3_val, depth1_values(1));
            analysisFile = fullfile(baseOutputDir, refFolder, 'mat', 'bp_analysis_results.mat');
            bpFile       = fullfile(baseOutputDir, refFolder, 'mat', 'bp_hybrid_results.mat');
        end

        if exist(analysisFile, 'file')
            try
                tmp = load(analysisFile, 'results');
                allResults{k} = tmp.results;
                statusList{k} = 'success';

                % Try to get computation time
                if exist(bpFile, 'file')
                    bpData = load(bpFile, 'tTotal');
                    if isfield(bpData, 'tTotal')
                        compTimes(k) = bpData.tTotal;
                    end
                end
            catch
                statusList{k} = 'load_error';
            end
        end

        if mod(k, 50) == 0
            fprintf('  Scanned %d / %d folders ...\n', k, nTotal);
        end
    end

    % Build the table from individual results
    N = nTotal;
    trialIdx      = (1:N)';
    n2_col        = n2_list;
    n3_col        = n3_list;
    depth1_col    = depth1_list;
    compTime      = compTimes;
    status_col    = statusList;

    % Pre-allocate metric columns
    peakMag            = NaN(N,1); peakX = NaN(N,1); peakY = NaN(N,1); peakZ = NaN(N,1);
    entropy_amp_norm   = NaN(N,1); entropy_int_norm = NaN(N,1);
    entropy_amp_raw    = NaN(N,1); entropy_int_raw  = NaN(N,1);
    sharp_peak_mean    = NaN(N,1); sharp_peak_median = NaN(N,1); sharp_peak_rms = NaN(N,1);
    res3_x = NaN(N,1); res3_y = NaN(N,1); res3_z = NaN(N,1);
    res3_vol = NaN(N,1); res3_equiv_diam = NaN(N,1);
    res6_x = NaN(N,1); res6_y = NaN(N,1); res6_z = NaN(N,1);
    TBR_dB_col = NaN(N,1); SCR_dB_col = NaN(N,1); CNR_col = NaN(N,1);
    PSLR_dB_col = NaN(N,1); ISLR_dB_col = NaN(N,1);
    kurtosis_col = NaN(N,1); skewness_col = NaN(N,1); CV_col = NaN(N,1); gini_col = NaN(N,1);
    pct_vox_50 = NaN(N,1); pct_vox_90 = NaN(N,1); pct_vox_99 = NaN(N,1);
    eff_scatterers = NaN(N,1); eff_scatterers_pct = NaN(N,1);
    grad_mean = NaN(N,1); grad_mean_peak = NaN(N,1); grad_max = NaN(N,1);
    psf_asym_max = NaN(N,1); dyn_range_dB = NaN(N,1);

    for k = 1:N
        R = allResults{k};
        if isempty(R), continue; end
        peakMag(k) = R.peak.value;
        peakX(k) = R.peak.location(1); peakY(k) = R.peak.location(2); peakZ(k) = R.peak.location(3);
        entropy_amp_norm(k) = R.entropy.normalized;
        entropy_int_norm(k) = R.entropy.intensity_normalized;
        entropy_amp_raw(k)  = R.entropy.shannon;
        entropy_int_raw(k)  = R.entropy.intensity;
        sharp_peak_mean(k) = R.sharpness.peak_mean;
        sharp_peak_median(k) = R.sharpness.peak_median;
        sharp_peak_rms(k) = R.sharpness.peak_rms;
        res3_x(k) = R.resolution_3dB.x; res3_y(k) = R.resolution_3dB.y; res3_z(k) = R.resolution_3dB.z;
        res3_vol(k) = R.resolution_3dB.volume; res3_equiv_diam(k) = R.resolution_3dB.equiv_diameter;
        res6_x(k) = R.resolution_6dB.x; res6_y(k) = R.resolution_6dB.y; res6_z(k) = R.resolution_6dB.z;
        TBR_dB_col(k) = R.contrast.TBR_dB; SCR_dB_col(k) = R.contrast.SCR_dB; CNR_col(k) = R.contrast.CNR;
        PSLR_dB_col(k) = R.sidelobes.PSLR_dB; ISLR_dB_col(k) = R.sidelobes.ISLR_dB;
        kurtosis_col(k) = R.statistical.kurtosis; skewness_col(k) = R.statistical.skewness;
        CV_col(k) = R.statistical.CV; gini_col(k) = R.statistical.gini;
        pct_vox_50(k) = R.energy.pct_voxels_50;
        pct_vox_90(k) = R.energy.pct_voxels_90;
        pct_vox_99(k) = R.energy.pct_voxels_99;
        eff_scatterers(k) = R.energy.effective_scatterers;
        eff_scatterers_pct(k) = R.energy.effective_scatterers_pct;
        grad_mean(k) = R.gradient.mean; grad_mean_peak(k) = R.gradient.mean_peak; grad_max(k) = R.gradient.max;
        psf_asym_max(k) = R.psf.asymmetry_max;
        dyn_range_dB(k) = R.dynamic_range_dB;
    end

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
        'Status'};

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
        status_col, ...
        'VariableNames', sweepVarNames);
end

%% ======================== FILTER SUCCESSFUL TRIALS =======================
% Remove failed or pending trials, keeping only successfully computed ones.
% Also strip bookkeeping columns (Status, Folder) that are not needed
% in the analysis-ready output.
if ismember('Status', sweepTable.Properties.VariableNames)
    successMask = strcmp(sweepTable.Status, 'success');
    T = sweepTable(successMask, :);
    fprintf('Successful trials: %d / %d\n', sum(successMask), height(sweepTable));
else
    T = sweepTable;
end

% Remove non-metric columns for a cleaner comparison table
colsToRemove = intersect(T.Properties.VariableNames, {'Status', 'Folder'});
T(:, colsToRemove) = [];

fprintf('Comparison table: %d rows x %d columns\n', height(T), width(T));

%% ======================== SAVE FULL TABLE ================================
fullCsvFile = fullfile(baseOutputDir, 'comparison_all_trials.csv');
writetable(T, fullCsvFile);
fprintf('Saved: %s\n', fullCsvFile);

% Also save as .mat for MATLAB users
save(fullfile(baseOutputDir, 'comparison_all_trials.mat'), 'T');

%% ======================== DEDUPLICATED TABLE (unique results) ============
% When n2 == n3, all depth1 values produce identical results (no refraction
% at the second interface). Remove these duplicate rows, keeping only the
% first depth1 entry for each n2==n3 pair to avoid inflating statistics.
sameN_mask = abs(T.n2 - T.n3) < 1e-10;
firstDepth = min(T.depth1_m);
redundant = sameN_mask & (abs(T.depth1_m - firstDepth) > 1e-10);
T_unique = T(~redundant, :);

fprintf('Unique trials (n2==n3 deduplicated): %d rows\n', height(T_unique));

uniqueCsvFile = fullfile(baseOutputDir, 'comparison_unique_trials.csv');
writetable(T_unique, uniqueCsvFile);
fprintf('Saved: %s\n', uniqueCsvFile);
save(fullfile(baseOutputDir, 'comparison_unique_trials.mat'), 'T_unique');

%% ======================== RANKED TABLE ===================================
% Compute a composite quality score to rank all parameter combinations.
%
% Scoring strategy:
%   1. Normalize each metric to [0, 1] across all trials (min-max scaling)
%   2. For "lower is better" metrics (entropy, resolution, PSLR, etc.),
%      invert the normalization: score = 1 - normalized
%   3. Compute a weighted average of all normalized scores
%   4. Sort by composite score (highest = best overall quality)
%
% Metric direction:
%   Higher is better: sharpness, SCR, CNR, TBR, dynamic range, kurtosis, gini, CV, gradient
%   Lower is better:  entropy, resolution widths, PSLR, ISLR, PctVox90, PSF asymmetry

metricDefs = {
    % Column name               Direction   Weight   Display name
    'SharpnessPkMean',           'higher',   1.5,    'Sharpness (Pk/Mean)'
    'EntropyAmpNorm',            'lower',    1.0,    'Entropy (Amp, Norm)'
    'EntropyIntNorm',            'lower',    1.0,    'Entropy (Int, Norm)'
    'Res3dB_X',                  'lower',    1.0,    'Resolution X (m)'
    'Res3dB_Y',                  'lower',    1.0,    'Resolution Y (m)'
    'Res3dB_Z',                  'lower',    1.0,    'Resolution Z (m)'
    'Res3dB_Volume',             'lower',    1.2,    'Resolution Volume (m^3)'
    'SCR_dB',                    'higher',   1.2,    'SCR (dB)'
    'TBR_dB',                    'higher',   1.0,    'TBR (dB)'
    'CNR',                       'higher',   1.0,    'CNR'
    'PSLR_dB',                   'lower',    1.0,    'PSLR (dB)'
    'ISLR_dB',                   'lower',    1.0,    'ISLR (dB)'
    'Kurtosis',                  'higher',   0.8,    'Kurtosis'
    'Gini',                      'higher',   0.8,    'Gini'
    'PctVox90',                  'lower',    0.8,    'PctVox 90% Energy'
    'DynamicRange_dB',           'higher',   1.0,    'Dynamic Range (dB)'
    'PSF_AsymMax',               'lower',    0.5,    'PSF Asymmetry (max)'
    'CV',                        'higher',   0.5,    'CV'
    'GradMean',                  'higher',   0.5,    'Gradient Mean'
};

nMetrics = size(metricDefs, 1);
nRows = height(T_unique);
normScores = NaN(nRows, nMetrics);
weights = zeros(nMetrics, 1);

for m = 1:nMetrics
    colName   = metricDefs{m, 1};
    direction = metricDefs{m, 2};
    w         = metricDefs{m, 3};
    weights(m) = w;

    vals = T_unique.(colName);
    vMin = min(vals, [], 'omitnan');
    vMax = max(vals, [], 'omitnan');
    range = vMax - vMin;

    if range < eps
        normScores(:, m) = 0.5;   % all equal -> neutral score
    else
        normalized = (vals - vMin) / range;   % 0 = worst, 1 = best when higher is better
        if strcmp(direction, 'lower')
            normalized = 1 - normalized;      % invert: 0 = worst (highest), 1 = best (lowest)
        end
        normScores(:, m) = normalized;
    end
end

% Weighted average composite score
compositeScore = (normScores * weights) / sum(weights);
T_unique.CompositeScore = compositeScore;

% Per-metric normalized scores (for transparency)
for m = 1:nMetrics
    scoreName = ['Score_' metricDefs{m, 1}];
    T_unique.(scoreName) = normScores(:, m);
end

% Sort by composite score (best first)
T_ranked = sortrows(T_unique, 'CompositeScore', 'descend');

% Add rank column
T_ranked.Rank = (1:height(T_ranked))';
% Move Rank to first column
T_ranked = [T_ranked(:, end), T_ranked(:, 1:end-1)];

rankedCsvFile = fullfile(baseOutputDir, 'comparison_ranked.csv');
writetable(T_ranked, rankedCsvFile);
fprintf('Saved: %s\n', rankedCsvFile);
save(fullfile(baseOutputDir, 'comparison_ranked.mat'), 'T_ranked');

%% ======================== PRINT TOP 20 ===================================
fprintf('\n');
fprintf('=====================================================================\n');
fprintf('                    TOP 20 PARAMETER COMBINATIONS                    \n');
fprintf('=====================================================================\n');
fprintf('%4s | %5s | %5s | %7s | %10s | %10s | %8s | %8s | %8s | %7s\n', ...
    'Rank', 'n2', 'n3', 'depth1', 'Sharpness', 'EntropyAmp', 'Res3dB_V', 'SCR_dB', 'PSLR_dB', 'Score');
fprintf('-----+-------+-------+---------+------------+------------+----------+----------+----------+--------\n');
nShow = min(20, height(T_ranked));
for r = 1:nShow
    fprintf('%4d | %5.3f | %5.3f | %5.2f m | %10.2f | %10.4f | %8.4f | %8.2f | %8.2f | %6.4f\n', ...
        T_ranked.Rank(r), T_ranked.n2(r), T_ranked.n3(r), T_ranked.depth1_m(r), ...
        T_ranked.SharpnessPkMean(r), T_ranked.EntropyAmpNorm(r), ...
        T_ranked.Res3dB_Volume(r), T_ranked.SCR_dB(r), T_ranked.PSLR_dB(r), ...
        T_ranked.CompositeScore(r));
end
fprintf('=====================================================================\n');

%% ======================== SUMMARY STATISTICS =============================
fprintf('\n');
fprintf('=====================================================================\n');
fprintf('                      METRIC SUMMARY STATISTICS                      \n');
fprintf('=====================================================================\n');

% Columns that are actual metrics (not parameters/bookkeeping)
metricCols = { ...
    'PeakMagnitude', ...
    'EntropyAmpNorm', 'EntropyIntNorm', 'EntropyAmpRaw', 'EntropyIntRaw', ...
    'SharpnessPkMean', 'SharpnessPkMedian', 'SharpnessPkRMS', ...
    'Res3dB_X', 'Res3dB_Y', 'Res3dB_Z', 'Res3dB_Volume', 'Res3dB_EquivDiam', ...
    'Res6dB_X', 'Res6dB_Y', 'Res6dB_Z', ...
    'TBR_dB', 'SCR_dB', 'CNR', ...
    'PSLR_dB', 'ISLR_dB', ...
    'Kurtosis', 'Skewness', 'CV', 'Gini', ...
    'PctVox50', 'PctVox90', 'PctVox99', 'EffScatterers', 'EffScatterersPct', ...
    'GradMean', 'GradMeanPeak', 'GradMax', ...
    'PSF_AsymMax', 'DynamicRange_dB'};

fprintf('%-22s | %12s | %12s | %12s | %12s | %12s\n', ...
    'Metric', 'Min', 'Max', 'Mean', 'Median', 'Std');
fprintf('%s\n', repmat('-', 1, 99));

for c = 1:numel(metricCols)
    col = metricCols{c};
    if ~ismember(col, T_unique.Properties.VariableNames), continue; end
    vals = T_unique.(col);
    fprintf('%-22s | %12.4g | %12.4g | %12.4g | %12.4g | %12.4g\n', ...
        col, min(vals, [], 'omitnan'), max(vals, [], 'omitnan'), ...
        mean(vals, 'omitnan'), median(vals, 'omitnan'), std(vals, 'omitnan'));
end
fprintf('=====================================================================\n');

%% ======================== DONE ==========================================
fprintf('\nComparison tables built successfully.\n');
fprintf('Files saved:\n');
fprintf('  %s  (%d rows - all successful trials)\n', fullCsvFile, height(T));
fprintf('  %s  (%d rows - unique results only)\n', uniqueCsvFile, height(T_unique));
fprintf('  %s  (%d rows - ranked by composite score)\n', rankedCsvFile, height(T_ranked));
fprintf('\nTo load in MATLAB:  load(''%s'');\n', ...
    fullfile(baseOutputDir, 'comparison_ranked.mat'));
fprintf('To load in Python:  pd.read_csv(''%s'')\n', rankedCsvFile);
