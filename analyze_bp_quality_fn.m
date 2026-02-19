function results = analyze_bp_quality_fn(Img, xGrid, yGrid, zGrid, n2, n3, depth1, outputDir)
% =========================================================================
% ANALYZE_BP_QUALITY_FN  -  Callable function for SAR image quality analysis
% =========================================================================
% Computes a comprehensive set of 35+ image quality metrics for a 3D
% backprojection (BP) volume and generates diagnostic figures.
%
% This function is called by the parameter sweep script (run_parameter_sweep.m)
% after each backprojection trial to quantitatively evaluate the resulting
% image. It produces two output figures and saves all intermediate variables
% for full reproducibility.
%
% Metrics computed:
%   - Entropy (amplitude & intensity, raw & normalized)
%   - Sharpness (peak/mean, peak/median, peak/RMS)
%   - Resolution (-3dB and -6dB widths in x, y, z; volume; equivalent diameter)
%   - Contrast (TBR, SCR, CNR in dB and linear)
%   - Sidelobes (PSLR, ISLR in dB and linear)
%   - Statistical (kurtosis, skewness, CV, Gini coefficient)
%   - Energy concentration (% voxels for 50/90/99% of total energy)
%   - Gradient (mean, peak-region mean, max)
%   - PSF asymmetry (max ratio across all axis pairs)
%   - Dynamic range (peak/noise floor in dB)
%   - Effective number of scatterers (participation ratio)
%
% Inputs:
%   Img       - Complex 3D image volume [Nx x Ny x Nz] (single precision)
%   xGrid     - X-axis coordinate vector [1 x Nx] (meters)
%   yGrid     - Y-axis coordinate vector [1 x Ny] (meters)
%   zGrid     - Z-axis coordinate vector [1 x Nz] (meters, negative = depth)
%   n2        - Refractive index of soil layer 1
%   n3        - Refractive index of soil layer 2
%   depth1    - Depth of DEM2 below DEM1 (meters)
%   outputDir - Directory for saving results (figures go to outputDir/plots/)
%
% Output:
%   results   - Struct with all computed metrics (scalar values)
%
% Output files:
%   outputDir/mat/bp_analysis_results.mat    - Results struct
%   outputDir/mat/bp_analysis_variables.mat  - All intermediate variables
%   outputDir/plots/quality_analysis.png/fig - Figure 1: image quality analysis
%   outputDir/plots/dem_interfaces.png/fig   - Figure 2: DEM interface overlay
% =========================================================================

if nargin < 8 || isempty(outputDir), outputDir = '.'; end

plotDir = fullfile(outputDir, 'plots');
if ~exist(plotDir, 'dir'), mkdir(plotDir); end

% Load both DEM surfaces from the trial's own results file.
% The backprojection function saves xDEM, yDEM, zDEM1, zDEM2 alongside
% the image so that the analysis can overlay the DEM interfaces on the
% image slices without needing the original input data file.
bpResultFile = fullfile(outputDir, 'mat', 'bp_hybrid_results.mat');
demData = load(bpResultFile, 'xDEM', 'yDEM', 'zDEM1', 'zDEM2');
xDEM  = demData.xDEM;
yDEM  = demData.yDEM;
zDEM1 = demData.zDEM1;
zDEM2 = demData.zDEM2;

%% ======================== BASIC INFO ====================================
% Compute the magnitude (absolute value) of the complex BP image.
% All quality metrics operate on the magnitude volume M.
M = abs(Img);
[Nx, Ny, Nz] = size(M);

% Grid spacing in each direction (uniform spacing assumed)
dx = xGrid(2) - xGrid(1);
dy = yGrid(2) - yGrid(1);
dz = zGrid(2) - zGrid(1);

%% ======================== FIND PEAK =====================================
% Locate the global maximum (strongest scatterer) in the 3D magnitude volume.
% The peak location serves as the reference point for all region-based metrics.
[peakVal, peakIdx] = max(M(:));
[ix_peak, iy_peak, iz_peak] = ind2sub([Nx, Ny, Nz], peakIdx);

x_peak = xGrid(ix_peak);
y_peak = yGrid(iy_peak);
z_peak = zGrid(iz_peak);

%% ======================== ENTROPY =======================================
% Entropy measures the dispersion of energy across the image. Lower entropy
% indicates more focused energy (desirable for well-focused BP images).
M_flat = M(:);
M_nonzero = M_flat(M_flat > 0);
n_vox = numel(M_nonzero);

% Amplitude-based Shannon entropy: treats normalized magnitudes as a
% probability distribution. H_norm = H / H_max maps to [0, 1].
p = M_nonzero / sum(M_nonzero);
entropy_shannon = -sum(p .* log(p));
entropy_max = log(n_vox);
entropy_normalized = entropy_shannon / entropy_max;

% Intensity-weighted entropy (power-based): uses squared magnitudes (power)
% as the distribution. More sensitive to dominant scatterers.
M2 = M_nonzero.^2;
p2 = M2 / sum(M2);
entropy_intensity = -sum(p2 .* log(p2));
entropy_intensity_max = log(numel(M2));
entropy_intensity_normalized = entropy_intensity / entropy_intensity_max;

%% ======================== SHARPNESS =====================================
% Sharpness ratios quantify how dominant the peak is relative to the
% background level. Higher sharpness = better focused image.
sharpness_peak_mean = peakVal / mean(M_nonzero);     % Peak-to-mean ratio
sharpness_peak_median = peakVal / median(M_nonzero);  % Peak-to-median ratio
sharpness_peak_rms = peakVal / sqrt(mean(M_nonzero.^2)); % Peak-to-RMS ratio

%% ======================== RESOLUTION (-3dB / -6dB) ======================
% Resolution is measured as the width of the Point Spread Function (PSF)
% at -3 dB and -6 dB thresholds along each axis. The PSF is approximated
% by 1D cuts through the peak along x, y, and z.
M_norm = M / peakVal;  % Peak-normalized magnitude [0, 1]

% Extract 1D profiles (cuts) through the peak along each axis
cut_x = squeeze(M_norm(:, iy_peak, iz_peak));  % X-cut at peak Y, Z
cut_y = squeeze(M_norm(ix_peak, :, iz_peak));  % Y-cut at peak X, Z
cut_z = squeeze(M_norm(ix_peak, iy_peak, :));  % Z-cut at peak X, Y

% Thresholds in linear amplitude (corresponding to -3 dB and -6 dB)
thresh_3dB = 10^(-3/20);  % ~0.7079 (half-power point)
thresh_6dB = 10^(-6/20);  % ~0.5012 (quarter-power point)

% Compute -3 dB widths by interpolating where the profile crosses the threshold
res_x = computeResolution_local(xGrid, cut_x, thresh_3dB);
res_y = computeResolution_local(yGrid, cut_y, thresh_3dB);
res_z = computeResolution_local(zGrid, cut_z, thresh_3dB);
res_volume = res_x * res_y * res_z;  % Resolution cell volume (m^3)

% -6 dB widths (wider than -3 dB, characterizes the extended PSF)
res_x_6dB = computeResolution_local(xGrid, cut_x, thresh_6dB);
res_y_6dB = computeResolution_local(yGrid, cut_y, thresh_6dB);
res_z_6dB = computeResolution_local(zGrid, cut_z, thresh_6dB);

% PSF (Point Spread Function) asymmetry analysis:
% Asymmetry ratio = max(res_i, res_j) / min(res_i, res_j) for each axis pair.
% A value of 1.0 indicates perfectly isotropic resolution; higher = more elongated.
asymmetry_xy = max(res_x, res_y) / min(res_x, res_y);
asymmetry_xz = max(res_x, res_z) / min(res_x, res_z);
asymmetry_yz = max(res_y, res_z) / min(res_y, res_z);
asymmetry_max = max([asymmetry_xy, asymmetry_xz, asymmetry_yz]);

% Equivalent sphere diameter: diameter of a sphere with the same volume as
% the resolution cell (useful for comparing anisotropic resolution cells)
equiv_diameter = 2 * (3 * res_volume / (4 * pi))^(1/3);

%% ======================== CONTRAST / SIDELOBE METRICS ===================
% Define spatial regions relative to the peak for contrast and sidelobe analysis.
% Three concentric spherical regions are used:
%   - Mainlobe: within half the largest resolution width of the peak
%   - Signal: within 2x the mainlobe radius (includes near sidelobes)
%   - Background: beyond 3x the mainlobe radius (clutter/noise region)
[Xgrid, Ygrid, Zgrid] = ndgrid(xGrid, yGrid, zGrid);
dist_from_peak = sqrt((Xgrid - x_peak).^2 + (Ygrid - y_peak).^2 + (Zgrid - z_peak).^2);

mainlobe_radius = max([res_x, res_y, res_z]) / 2;     % Half-width of the largest PSF dimension
mainlobe_mask = dist_from_peak <= mainlobe_radius;      % Core peak region
background_radius = 3 * mainlobe_radius;                % 3x mainlobe = far-field background
background_mask = dist_from_peak > background_radius;    % Background/clutter region
signal_mask = dist_from_peak <= 2 * mainlobe_radius;     % Extended signal region (for gradient analysis)

if any(mainlobe_mask(:)) && any(background_mask(:))
    mainlobe_mean = mean(M(mainlobe_mask));
    background_mean = mean(M(background_mask));
    background_std = std(M(background_mask));

    % Target-to-Background Ratio (TBR): average mainlobe / average background
    TBR = mainlobe_mean / background_mean;
    TBR_dB = 20 * log10(TBR);

    % Signal-to-Clutter Ratio (SCR): peak value / average background
    SCR = peakVal / background_mean;
    SCR_dB = 20 * log10(SCR);

    % Contrast-to-Noise Ratio (CNR): contrast normalized by background variability
    CNR = (mainlobe_mean - background_mean) / background_std;
else
    TBR = NaN; TBR_dB = NaN; SCR = NaN; SCR_dB = NaN; CNR = NaN;
end

% --- Sidelobe metrics ---
% Zero out the mainlobe region to isolate sidelobe contributions
M_outside_mainlobe = M;
M_outside_mainlobe(mainlobe_mask) = 0;

% Peak Sidelobe Level Ratio (PSLR): main peak / highest sidelobe peak
% Higher PSLR = better suppression of sidelobes
peak_sidelobe = max(M_outside_mainlobe(:));
if peak_sidelobe > 0
    PSLR = peakVal / peak_sidelobe;
    PSLR_dB = 20 * log10(PSLR);
else
    PSLR = inf; PSLR_dB = inf;
end

% Integrated Sidelobe Level Ratio (ISLR): mainlobe energy / total sidelobe energy
% Higher ISLR = more energy concentrated in the mainlobe
mainlobe_energy = sum(M(mainlobe_mask).^2);
sidelobe_energy = sum(M_outside_mainlobe(:).^2);
if sidelobe_energy > 0
    ISLR = mainlobe_energy / sidelobe_energy;
    ISLR_dB = 10 * log10(ISLR);
else
    ISLR = inf; ISLR_dB = inf;
end

%% ======================== STATISTICAL METRICS ===========================
% Higher-order statistics characterize the shape of the amplitude distribution.
M_centered = M_nonzero - mean(M_nonzero);

% Kurtosis: measures "peakedness" of the distribution. Values > 3 indicate
% a sharper peak than a Gaussian (leptokurtic), which is desirable.
kurtosis_val = mean(M_centered.^4) / (mean(M_centered.^2)^2);

% Skewness: measures asymmetry. Positive skewness = long right tail (few
% strong scatterers dominate, which is typical for well-focused SAR images).
skewness_val = mean(M_centered.^3) / (mean(M_centered.^2)^1.5);

% Coefficient of Variation: std/mean. Higher CV = more variability in
% amplitude, indicating stronger contrast between scatterers and background.
CV = std(M_nonzero) / mean(M_nonzero);

% Gini coefficient: measures inequality/concentration of amplitudes.
% Values close to 1 indicate that most energy is concentrated in very few voxels.
M_sorted = sort(M_nonzero);
gini = (2 * sum((1:n_vox)' .* M_sorted) / (n_vox * sum(M_sorted))) - (n_vox + 1) / n_vox;

%% ======================== ENERGY CONCENTRATION ==========================
% Determine what fraction of voxels contains a given percentage of total energy.
% Lower percentages = better energy concentration (tighter focus).
M2_sorted = sort(M_nonzero.^2, 'descend');  % Sort squared magnitudes (energy) descending
cumulative_energy = cumsum(M2_sorted) / sum(M2_sorted);  % Cumulative energy fraction

% Find the minimum number of voxels needed to capture 50%, 90%, and 99% of energy
idx_50 = find(cumulative_energy >= 0.50, 1, 'first');
idx_90 = find(cumulative_energy >= 0.90, 1, 'first');
idx_99 = find(cumulative_energy >= 0.99, 1, 'first');

pct_voxels_50 = 100 * idx_50 / n_vox;  % % of voxels for 50% energy
pct_voxels_90 = 100 * idx_90 / n_vox;  % % of voxels for 90% energy
pct_voxels_99 = 100 * idx_99 / n_vox;  % % of voxels for 99% energy

%% ======================== GRADIENT ======================================
% Image gradient magnitude measures edge sharpness. High gradients near the
% peak indicate a sharp transition from target to background (well-focused).
[Gx, Gy, Gz] = gradient(M, dx, dy, dz);  % 3D numerical gradient
gradient_magnitude = sqrt(Gx.^2 + Gy.^2 + Gz.^2);
mean_gradient = mean(gradient_magnitude(:));        % Global mean gradient
max_gradient = max(gradient_magnitude(:));          % Maximum gradient anywhere
gradient_at_peak_region = gradient_magnitude(signal_mask);
mean_gradient_peak = mean(gradient_at_peak_region(:));  % Mean gradient near the peak

%% ======================== DYNAMIC RANGE =================================
% Dynamic range: ratio between the peak and the minimum non-noise amplitude.
% Voxels below 1% of peak are considered noise floor and excluded.
M_above_noise = M(M > 0.01 * peakVal);
if ~isempty(M_above_noise)
    dynamic_range = peakVal / min(M_above_noise);
    dynamic_range_dB = 20 * log10(dynamic_range);
else
    dynamic_range = NaN; dynamic_range_dB = NaN; %#ok<NASGU>
end

%% ======================== EFFECTIVE SCATTERERS ==========================
% Effective number of scatterers based on the "participation ratio" concept
% from statistical physics. A value of N means the energy is spread over
% approximately N voxels. Lower values = more concentrated energy.
M4_sum = sum(M_nonzero.^4);
M2_sum = sum(M_nonzero.^2);
if M4_sum > 0
    effective_scatterers = M2_sum^2 / M4_sum;         % Participation ratio
    effective_scatterers_pct = 100 * effective_scatterers / n_vox;  % As % of total voxels
else
    effective_scatterers = NaN; effective_scatterers_pct = NaN;
end

%% ======================== STORE RESULTS =================================
results.n2 = n2;
results.n3 = n3;
results.depth1 = depth1;

% Peak
results.peak.value = peakVal;
results.peak.location = [x_peak, y_peak, z_peak];

% Entropy
results.entropy.shannon = entropy_shannon;
results.entropy.normalized = entropy_normalized;
results.entropy.intensity = entropy_intensity;
results.entropy.intensity_normalized = entropy_intensity_normalized;

% Sharpness
results.sharpness.peak_mean = sharpness_peak_mean;
results.sharpness.peak_median = sharpness_peak_median;
results.sharpness.peak_rms = sharpness_peak_rms;

% Resolution
results.resolution_3dB.x = res_x;
results.resolution_3dB.y = res_y;
results.resolution_3dB.z = res_z;
results.resolution_3dB.volume = res_volume;
results.resolution_3dB.equiv_diameter = equiv_diameter;
results.resolution_6dB.x = res_x_6dB;
results.resolution_6dB.y = res_y_6dB;
results.resolution_6dB.z = res_z_6dB;

% Contrast
results.contrast.TBR = TBR;
results.contrast.TBR_dB = TBR_dB;
results.contrast.SCR = SCR;
results.contrast.SCR_dB = SCR_dB;
results.contrast.CNR = CNR;

% Sidelobes
results.sidelobes.PSLR = PSLR;
results.sidelobes.PSLR_dB = PSLR_dB;
results.sidelobes.ISLR = ISLR;
results.sidelobes.ISLR_dB = ISLR_dB;

% Statistical
results.statistical.kurtosis = kurtosis_val;
results.statistical.skewness = skewness_val;
results.statistical.CV = CV;
results.statistical.gini = gini;

% Energy concentration
results.energy.pct_voxels_50 = pct_voxels_50;
results.energy.pct_voxels_90 = pct_voxels_90;
results.energy.pct_voxels_99 = pct_voxels_99;
results.energy.effective_scatterers = effective_scatterers;
results.energy.effective_scatterers_pct = effective_scatterers_pct;

% Gradient
results.gradient.mean = mean_gradient;
results.gradient.mean_peak = mean_gradient_peak;
results.gradient.max = max_gradient;

% PSF
results.psf.asymmetry_max = asymmetry_max;
results.psf.asymmetry = [asymmetry_xy, asymmetry_xz, asymmetry_yz];

% Dynamic range
results.dynamic_range_dB = dynamic_range_dB;

%% ======================== SAVE ANALYSIS .mat ============================
matDir = fullfile(outputDir, 'mat');
if ~exist(matDir, 'dir'), mkdir(matDir); end

% Save the compact results struct (scalar metric values only).
% This is the file loaded by run_parameter_sweep.m to populate the summary table.
save(fullfile(matDir, 'bp_analysis_results.mat'), 'results');

% Save ALL intermediate variables needed to reproduce every figure and
% every metric from scratch. This enables post-hoc re-analysis and custom
% visualization without re-running the expensive backprojection.
% Loading this file + bp_hybrid_results.mat gives full reproducibility.

av = struct();  % analysis_vars

% ---- Parameters & grid ----
av.n2     = n2;
av.n3     = n3;
av.depth1 = depth1;
av.xGrid  = xGrid;
av.yGrid  = yGrid;
av.zGrid  = zGrid;
av.dx     = dx;
av.dy     = dy;
av.dz     = dz;
av.Nx     = Nx;
av.Ny     = Ny;
av.Nz     = Nz;

% ---- Magnitude volumes ----
av.M          = M;              % absolute magnitude [Nx x Ny x Nz]
av.M_norm     = M_norm;         % peak-normalized magnitude [Nx x Ny x Nz]

% ---- Peak ----
av.peakVal      = peakVal;
av.peak_idx     = [ix_peak, iy_peak, iz_peak];
av.peak_location = [x_peak, y_peak, z_peak];

% ---- 1D cuts through peak (for resolution & Figure 1 row 2) ----
av.cut_x = cut_x;
av.cut_y = cut_y;
av.cut_z = cut_z;

% ---- 2D slices through peak (for Figure 1 row 1 & Figure 2 subplots) ----
av.slice_xy = squeeze(M_norm(:, :, iz_peak)).';   % [Ny x Nx]
av.slice_xz = squeeze(M_norm(:, iy_peak, :)).';   % [Nz x Nx]
av.slice_yz = squeeze(M_norm(ix_peak, :, :)).';    % [Nz x Ny]

% ---- Thresholds ----
av.thresh_3dB = thresh_3dB;
av.thresh_6dB = thresh_6dB;

% ---- Resolution -3 dB ----
av.res_x     = res_x;
av.res_y     = res_y;
av.res_z     = res_z;
av.res_volume = res_volume;
av.equiv_diameter = equiv_diameter;

% ---- Resolution -6 dB ----
av.res_x_6dB = res_x_6dB;
av.res_y_6dB = res_y_6dB;
av.res_z_6dB = res_z_6dB;

% ---- PSF asymmetry ----
av.asymmetry_xy  = asymmetry_xy;
av.asymmetry_xz  = asymmetry_xz;
av.asymmetry_yz  = asymmetry_yz;
av.asymmetry_max = asymmetry_max;

% ---- Masks (logical 3D volumes) ----
av.mainlobe_mask    = mainlobe_mask;     % [Nx x Ny x Nz]
av.background_mask  = background_mask;   % [Nx x Ny x Nz]
av.signal_mask      = signal_mask;       % [Nx x Ny x Nz]
av.mainlobe_radius  = mainlobe_radius;
av.background_radius = background_radius;
av.dist_from_peak   = dist_from_peak;    % [Nx x Ny x Nz]

% ---- Contrast & sidelobe intermediates ----
av.M_outside_mainlobe = M_outside_mainlobe;  % M with mainlobe zeroed [Nx x Ny x Nz]
av.peak_sidelobe      = peak_sidelobe;
av.mainlobe_energy    = mainlobe_energy;
av.sidelobe_energy    = sidelobe_energy;

% ---- Entropy distributions ----
av.M_nonzero = M_nonzero;    % non-zero magnitude voxels (vector)
av.p         = p;            % amplitude probability distribution
av.p2        = p2;           % intensity (power) probability distribution
av.n_vox     = n_vox;        % number of non-zero voxels

% ---- Statistical intermediates ----
av.M_centered = M_centered;  % mean-subtracted non-zero magnitudes
av.M_sorted   = M_sorted;    % sorted non-zero magnitudes (ascending, for Gini)

% ---- Energy concentration ----
av.M2_sorted          = M2_sorted;          % sorted squared magnitudes (descending)
av.cumulative_energy  = cumulative_energy;  % cumulative fraction of total energy

% ---- Gradient (3D volumes + components) ----
av.Gx                 = Gx;                  % gradient x-component [Nx x Ny x Nz]
av.Gy                 = Gy;                  % gradient y-component [Nx x Ny x Nz]
av.Gz                 = Gz;                  % gradient z-component [Nx x Ny x Nz]
av.gradient_magnitude = gradient_magnitude;  % |grad| [Nx x Ny x Nz]

% ---- Dynamic range ----
av.M_above_noise = M_above_noise;  % voxels above 1% of peak

% ---- DEM data (for Figure 2) ----
av.xDEM  = xDEM;
av.yDEM  = yDEM;
av.zDEM1 = zDEM1;
av.zDEM2 = zDEM2;

% DEM slices at peak Y and peak X (for Figure 2 subplots 4 & 5)
[~, iy_dem] = min(abs(yDEM - y_peak));
[~, ix_dem] = min(abs(xDEM - x_peak));
av.iy_dem       = iy_dem;
av.ix_dem       = ix_dem;
av.dem1_slice_x = zDEM1(iy_dem, :);       % DEM1 along X at peak Y
av.dem2_slice_x = zDEM2(iy_dem, :);       % DEM2 along X at peak Y
av.dem1_slice_y = zDEM1(:, ix_dem);       % DEM1 along Y at peak X
av.dem2_slice_y = zDEM2(:, ix_dem);       % DEM2 along Y at peak X

% ---- Histogram data (for Figure 1 subplot 8) ----
av.hist_data = M_norm(M_norm > 0);  % all non-zero normalized amplitudes

save(fullfile(matDir, 'bp_analysis_variables.mat'), 'av', '-v7.3');

%% ========================================================================
% FIGURE 1: IMAGE QUALITY ANALYSIS (LINEAR SCALE)
% =========================================================================
fig1 = figure('Color', 'w', 'Position', [50 50 1600 900], 'Visible', 'off');

% Row 1: 2D Image Slices
subplot(2, 4, 1);
slice_xy = squeeze(M_norm(:, :, iz_peak)).';
imagesc(xGrid, yGrid, slice_xy);
axis xy image; colorbar; colormap(gca, 'jet');
hold on;
plot(x_peak, y_peak, 'k+', 'MarkerSize', 15, 'LineWidth', 2);
theta = linspace(0, 2*pi, 100);
plot(x_peak + res_x/2*cos(theta), y_peak + res_y/2*sin(theta), 'k--', 'LineWidth', 1.5);
hold off;
xlabel('x [m]'); ylabel('y [m]');
title(sprintf('XY Slice at Z = %.2f m', z_peak));

subplot(2, 4, 2);
slice_xz = squeeze(M_norm(:, iy_peak, :)).';
imagesc(xGrid, zGrid, slice_xz);
axis xy image; colorbar; colormap(gca, 'jet');
hold on; plot(x_peak, z_peak, 'k+', 'MarkerSize', 15, 'LineWidth', 2); hold off;
xlabel('x [m]'); ylabel('z [m]');
title(sprintf('XZ Slice at Y = %.2f m', y_peak));

subplot(2, 4, 3);
slice_yz = squeeze(M_norm(ix_peak, :, :)).';
imagesc(yGrid, zGrid, slice_yz);
axis xy image; colorbar; colormap(gca, 'jet');
hold on; plot(y_peak, z_peak, 'k+', 'MarkerSize', 15, 'LineWidth', 2); hold off;
xlabel('y [m]'); ylabel('z [m]');
title(sprintf('YZ Slice at X = %.2f m', x_peak));

% Metrics text box
subplot(2, 4, 4);
axis off;
text(0.05, 0.98, 'KEY METRICS', 'FontSize', 12, 'FontWeight', 'bold');
ypos = 0.88;
text(0.05, ypos, sprintf('n_2=%.3f, n_3=%.3f, d_1=%.2fm', n2, n3, depth1), 'FontSize', 9);
ypos = ypos - 0.08;
text(0.05, ypos, 'Entropy (norm):', 'FontSize', 9, 'FontWeight', 'bold'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('  Ampl: %.4f  Int: %.4f', entropy_normalized, entropy_intensity_normalized), 'FontSize', 8, 'Color', 'b'); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('Sharpness: %.2f', sharpness_peak_mean), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('Gini: %.4f', gini), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.08;
text(0.05, ypos, 'Resolution -3dB:', 'FontSize', 9, 'FontWeight', 'bold'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('  X:%.3f Y:%.3f Z:%.3fm', res_x, res_y, res_z), 'FontSize', 8); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('SCR: %.1f dB', SCR_dB), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('PSLR: %.1f dB', PSLR_dB), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.06;
text(0.05, ypos, sprintf('ISLR: %.1f dB', ISLR_dB), 'FontSize', 9, 'Color', 'b'); ypos = ypos - 0.07;
text(0.05, ypos, sprintf('90%% energy in %.1f%% voxels', pct_voxels_90), 'FontSize', 8);

% Row 2: 1D Cuts
subplot(2, 4, 5);
plot(xGrid, cut_x, 'b-', 'LineWidth', 1.5);
hold on;
yline(thresh_3dB, 'r--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(thresh_6dB, 'm--', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
xline(x_peak - res_x/2, 'g:', 'LineWidth', 1.5);
xline(x_peak + res_x/2, 'g:', 'LineWidth', 1.5);
hold off;
xlim([xGrid(1) xGrid(end)]); ylim([0 1.05]);
xlabel('x [m]'); ylabel('Normalized Amplitude');
title(sprintf('X Cut (Res = %.4f m)', res_x)); grid on;

subplot(2, 4, 6);
plot(yGrid, cut_y, 'b-', 'LineWidth', 1.5);
hold on;
yline(thresh_3dB, 'r--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(thresh_6dB, 'm--', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
xline(y_peak - res_y/2, 'g:', 'LineWidth', 1.5);
xline(y_peak + res_y/2, 'g:', 'LineWidth', 1.5);
hold off;
xlim([yGrid(1) yGrid(end)]); ylim([0 1.05]);
xlabel('y [m]'); ylabel('Normalized Amplitude');
title(sprintf('Y Cut (Res = %.4f m)', res_y)); grid on;

subplot(2, 4, 7);
plot(zGrid, cut_z, 'b-', 'LineWidth', 1.5);
hold on;
yline(thresh_3dB, 'r--', '-3dB', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(thresh_6dB, 'm--', '-6dB', 'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
xline(z_peak - res_z/2, 'g:', 'LineWidth', 1.5);
xline(z_peak + res_z/2, 'g:', 'LineWidth', 1.5);
hold off;
xlim([zGrid(1) zGrid(end)]); ylim([0 1.05]);
xlabel('z [m]'); ylabel('Normalized Amplitude');
title(sprintf('Z Cut (Res = %.4f m)', res_z)); grid on;

subplot(2, 4, 8);
histogram(M_norm(M_norm > 0), 100, 'FaceColor', 'b', 'EdgeColor', 'none', 'Normalization', 'probability');
xlabel('Normalized Amplitude'); ylabel('Probability');
title('Amplitude Distribution'); xlim([0 1]); grid on;

sgtitle(sprintf('Image Analysis: n_2 = %.3f, n_3 = %.3f, d_1 = %.2f m', n2, n3, depth1), 'FontSize', 14, 'FontWeight', 'bold');

saveas(fig1, fullfile(plotDir, 'quality_analysis.png'));
saveas(fig1, fullfile(plotDir, 'quality_analysis.fig'));
close(fig1);

%% ========================================================================
% FIGURE 2: DEM INTERFACES
% =========================================================================
fig2 = figure('Color', 'w', 'Position', [100 100 1500 800], 'Visible', 'off');

% --- Subplot 1: 3D view of both DEM surfaces ---
subplot(2, 3, 1);
[X_dem, Y_dem] = meshgrid(xDEM, yDEM);
h1 = surf(X_dem, Y_dem, zDEM1, 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'FaceColor', [0.6 0.4 0.2]);
hold on;
h2 = surf(X_dem, Y_dem, zDEM2, 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'FaceColor', [0.4 0.3 0.5]);
h3 = plot3(x_peak, y_peak, z_peak, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot3([x_peak x_peak], [y_peak y_peak], [z_peak zGrid(end)], 'r:', 'LineWidth', 1);
hold off;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('3D View: DEM Interfaces');
legend([h1 h2 h3], {'DEM1 (Air-Soil1)', 'DEM2 (Soil1-Soil2)', 'Peak'}, 'Location', 'northeast');
view(45, 30); axis tight; grid on;

% --- Subplot 2: DEM1 surface (top view) ---
subplot(2, 3, 2);
imagesc(xDEM, yDEM, zDEM1); axis xy image;
cb = colorbar; ylabel(cb, 'z [m]');
hold on;
plot(x_peak, y_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
contour(xDEM, yDEM, zDEM1, 10, 'k', 'LineWidth', 0.5);
hold off;
xlabel('x [m]'); ylabel('y [m]');
title('DEM1: Air-Soil1 Interface'); colormap(gca, 'copper');

% --- Subplot 3: DEM2 surface (top view) ---
subplot(2, 3, 3);
imagesc(xDEM, yDEM, zDEM2); axis xy image;
cb = colorbar; ylabel(cb, 'z [m]');
hold on;
plot(x_peak, y_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
contour(xDEM, yDEM, zDEM2, 10, 'k', 'LineWidth', 0.5);
hold off;
xlabel('x [m]'); ylabel('y [m]');
title(sprintf('DEM2: Soil1-Soil2 (depth_1 = %.2f m)', depth1)); colormap(gca, 'copper');

% --- Subplot 4: XZ slice with DEM interfaces overlaid ---
subplot(2, 3, 4);
slice_xz = squeeze(M_norm(:, iy_peak, :)).';
imagesc(xGrid, zGrid, slice_xz); axis xy; colorbar;
hold on;
[~, iy_dem] = min(abs(yDEM - y_peak));
dem1_slice = zDEM1(iy_dem, :);
dem2_slice = zDEM2(iy_dem, :);
plot(xDEM, dem1_slice, 'w-', 'LineWidth', 3);
plot(xDEM, dem1_slice, 'k-', 'LineWidth', 1.5);
plot(xDEM, dem2_slice, 'w-', 'LineWidth', 3);
plot(xDEM, dem2_slice, 'm-', 'LineWidth', 1.5);
plot(x_peak, z_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
text(xGrid(end)-0.1, mean(dem1_slice)+0.05, 'DEM1', 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
text(xGrid(end)-0.1, mean(dem2_slice)+0.05, 'DEM2', 'Color', 'm', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
hold off;
xlabel('x [m]'); ylabel('z [m]');
title(sprintf('XZ Slice at Y = %.2f m', y_peak));
colormap(gca, 'jet');
ylim([zGrid(1) max(zGrid(end), max(dem1_slice(:))+0.1)]);

% --- Subplot 5: YZ slice with DEM interfaces overlaid ---
subplot(2, 3, 5);
slice_yz = squeeze(M_norm(ix_peak, :, :)).';
imagesc(yGrid, zGrid, slice_yz); axis xy; colorbar;
hold on;
[~, ix_dem] = min(abs(xDEM - x_peak));
dem1_slice_y = zDEM1(:, ix_dem);
dem2_slice_y = zDEM2(:, ix_dem);
plot(yDEM, dem1_slice_y, 'w-', 'LineWidth', 3);
plot(yDEM, dem1_slice_y, 'k-', 'LineWidth', 1.5);
plot(yDEM, dem2_slice_y, 'w-', 'LineWidth', 3);
plot(yDEM, dem2_slice_y, 'm-', 'LineWidth', 1.5);
plot(y_peak, z_peak, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
text(yGrid(end)-0.1, mean(dem1_slice_y)+0.05, 'DEM1', 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
text(yGrid(end)-0.1, mean(dem2_slice_y)+0.05, 'DEM2', 'Color', 'm', 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
hold off;
xlabel('y [m]'); ylabel('z [m]');
title(sprintf('YZ Slice at X = %.2f m', x_peak));
colormap(gca, 'jet');
ylim([zGrid(1) max(zGrid(end), max(dem1_slice_y(:))+0.1)]);

% --- Subplot 6: Cross-section schematic ---
subplot(2, 3, 6);
hold on;
xsch = [0 1];
fill([xsch fliplr(xsch)], [1 1 0.85 0.85], [0.7 0.9 1.0], 'EdgeColor', 'none');
fill([xsch fliplr(xsch)], [0.85 0.85 0.5 0.5], [0.76 0.60 0.42], 'EdgeColor', 'none');
fill([xsch fliplr(xsch)], [0.5 0.5 0.05 0.05], [0.55 0.45 0.35], 'EdgeColor', 'none');
plot(xsch, [0.85 0.85], 'k-', 'LineWidth', 2);
plot(xsch, [0.5 0.5], 'm-', 'LineWidth', 2);
text(0.5, 0.925, 'AIR', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0 0 0.5]);
text(0.5, 0.675, sprintf('SOIL 1 (n_2 = %.2f)', n2), 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', 'w');
text(0.5, 0.275, sprintf('SOIL 2 (n_3 = %.2f)', n3), 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', 'w');
text(1.05, 0.85, 'DEM1', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
text(1.05, 0.5, 'DEM2', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'm');
plot(0.5, 0.35, 'rp', 'MarkerSize', 20, 'MarkerFaceColor', 'r');
text(0.58, 0.35, 'Target', 'FontSize', 11, 'Color', 'r', 'FontWeight', 'bold');
text(1.12, 0.925, 'n_1 = 1.0', 'FontSize', 10);
text(1.12, 0.675, sprintf('n_2 = %.2f', n2), 'FontSize', 10);
text(1.12, 0.275, sprintf('n_3 = %.2f', n3), 'FontSize', 10);
hold off;
title('Layer Model', 'FontSize', 12, 'FontWeight', 'bold');
xlim([-0.05 1.35]); ylim([0 1.05]); axis off; box off;

sgtitle(sprintf('DEM Interfaces: n_2 = %.2f, n_3 = %.2f, depth_1 = %.2f m', n2, n3, depth1), ...
    'FontSize', 14, 'FontWeight', 'bold');

saveas(fig2, fullfile(plotDir, 'dem_interfaces.png'));
saveas(fig2, fullfile(plotDir, 'dem_interfaces.fig'));
close(fig2);

fprintf('  Analysis complete: %d metrics saved.\n', 42);
end

%% ========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================

function width = computeResolution_local(axis_vec, cut, threshold)
% COMPUTERESOLUTION_LOCAL  Measure the width of a 1D profile at a given threshold.
%
% Finds the two points where the profile crosses the threshold level on
% either side of the peak, using linear interpolation between samples.
% Returns the distance between these crossing points (the "width" at that level).
%
% Inputs:
%   axis_vec  - Coordinate vector (e.g., xGrid)
%   cut       - 1D amplitude profile through the peak
%   threshold - Normalized threshold level (e.g., 0.7079 for -3 dB)
%
% Output:
%   width     - Width of the profile at the given threshold (in same units as axis_vec)
    cut = cut(:);
    axis_vec = axis_vec(:);
    [~, ipk] = max(cut);

    left_cut = cut(1:ipk);
    left_axis = axis_vec(1:ipk);
    idx_left = find(left_cut < threshold, 1, 'last');
    if isempty(idx_left) || idx_left >= length(left_cut)
        x_left = left_axis(1);
    else
        x_left = interp1(left_cut(idx_left:idx_left+1), left_axis(idx_left:idx_left+1), threshold, 'linear', 'extrap');
    end

    right_cut = cut(ipk:end);
    right_axis = axis_vec(ipk:end);
    idx_right = find(right_cut < threshold, 1, 'first');
    if isempty(idx_right) || idx_right <= 1
        x_right = right_axis(end);
    else
        x_right = interp1(right_cut(idx_right-1:idx_right), right_axis(idx_right-1:idx_right), threshold, 'linear', 'extrap');
    end

    width = abs(x_right - x_left);
end

