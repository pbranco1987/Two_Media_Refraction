function [Img, tTotal] = bp_hybrid(n2_in, n3_in, depth1_in, outputDir)
% =========================================================================
% HYBRID MULTI-LAYER SAR BACKPROJECTION
% =========================================================================
% Core idea:
%   For each voxel (3D point) and each radar position, we coherently sum
%   the radar trace contribution at the time instant corresponding to the
%   round-trip travel time of the pulse through refracting media.
%
% Propagation scenario (3 media):
%   - Medium 1: Air (refractive index n_air = 1.0)
%   - Medium 2: Subsurface "soil layer 1" (refractive index n2)
%   - Medium 3: Deeper layer "soil layer 2" (refractive index n3),
%               located below an interface at depth 'depth1' under DEM1.
%
% Ray path for a voxel V:
%   - If V is above the 2nd interface (in layer 2): path is R -> P1 -> V
%   - If V is below the 2nd interface (in layer 3): path is R -> P1 -> P2 -> V
% Where:
%   R  = radar position
%   P1 = refraction point on the ground surface (DEM1), air -> n2
%   P2 = refraction point on the subsurface interface (DEM2), n2 -> n3
%
% Hybrid optimization strategy:
%   PHASE 1) DEM-IVS (Iterative Voxel Search): A coarse global search
%            using few iterations and few candidate points to find
%            approximate refraction points P1 and P2.
%   PHASE 2) Newton-Raphson: Local refinement with quadratic convergence,
%            fast when properly initialized by Phase 1.
%
% Computational cost comparison (number of geometric cost evaluations):
%   Pure DEM-IVS (example): 6 iterations x 25 candidates = 150 evaluations
%   Hybrid:                 2x9 + 4x1 = 22 evaluations (~7x faster)
%
% Optional inputs:
%   n2_in     - refractive index for soil layer 1 (default: 3.8)
%   n3_in     - refractive index for soil layer 2 (default: 3.4)
%   depth1_in - depth (m) of interface between layers 2 and 3 below DEM (default: 0.85)
%   outputDir - directory for saving .mat and plots (default: current dir)
%
% Outputs:
%   Img    - Complex 3D backprojection image [Nx x Ny x Nz] (single precision)
%   tTotal - Total computation time in seconds
% =========================================================================

clc; close all;

%% ======================== SETUP =========================================
% Initialize and reset the GPU to clear fragmentation and ensure maximum
% available memory for the computation.
g = gpuDevice; reset(g);
fprintf('GPU: %s (%.1f GB free)\n', g.Name, g.AvailableMemory/1e9);

% -------------------------- Physical parameters --------------------------
dataFile = 'IMOC_Inputs.mat';   % MAT file containing radarData, positions, DEM, etc.

% Use input arguments if provided; otherwise fall back to defaults.
% This allows the function to be called standalone (no arguments) or
% programmatically from the parameter sweep script.
if nargin < 1 || isempty(n2_in),     n2_in = 3.8;  end
if nargin < 2 || isempty(n3_in),     n3_in = 3.4;  end
if nargin < 3 || isempty(depth1_in), depth1_in = 0.85; end
if nargin < 4 || isempty(outputDir), outputDir = '.'; end

n2 = single(n2_in);        % Effective refractive index of soil layer 1 (medium 2)
n3 = single(n3_in);        % Effective refractive index of soil layer 2 (medium 3)
depth1 = single(depth1_in); % Depth (m) of DEM2 below DEM1 (soil1-soil2 interface)
n_air = single(1.0);   % Refractive index of air (medium 1)
c = single(299792458); % Speed of light in vacuum (m/s)
fc = single(425e6);    % Center frequency (Hz) -- used for coherent phase correction

% -------------------------- 3D imaging grid ------------------------------
% Define the voxel grid: coordinates where the 3D image will be reconstructed.
% Each voxel represents a point in 3D space that might contain a scatterer.
step = 0.05;                             % Grid spacing (m) in all three axes
xGrid = single(3.2:step:5.4);
yGrid = single(3.2:step:5.4);
zGrid = single(-2.7:step:-0.1);         % Negative z = depth below reference level

% -------------------------- Signal oversampling ----------------------------
oversampFactor = 4;  % Upsample radar traces by this factor via FFT zero-padding.
                     % Higher values give more accurate time-delay interpolation
                     % at the cost of proportionally more memory. Factor of 4
                     % is a good trade-off for sub-sample delay accuracy.

% -------------------------- Hybrid model parameters ----------------------
nCPIVSIter = 2;      % Number of DEM-IVS iterations for global initialization
NgridXY   = 3;       % Candidate grid size per iteration: NgridXY x NgridXY = 3x3 = 9 candidates
shrink    = single(0.35); % Search radius shrink factor per iteration (coarse -> fine)
radFac    = single(0.5);  % Scale factor for initial search radius (based on horizontal distance)
radMin    = single(0.02); % Minimum search radius (m) to prevent degenerate searches
radMax    = single(2.0);  % Maximum search radius (m) to prevent excessively wide searches

nNewtonIter = 4;     % Number of Newton-Raphson iterations for local refinement (3-4 is usually sufficient)

%% ======================== LOAD DATA =====================================
fprintf('  HYBRID BACKPROJECTION (DEM-IVS init + Newton refine)\n');

% Load the input data structure robustly (handles nested structs; see loadData below)
S = loadData(dataFile);

% radarData: matrix [Nt x Nrad] where Nt = number of time samples,
% Nrad = number of radar positions (each column is one A-scan / trace)
radarData = single(S.radarData);

% Temporal sampling rate of the radar signal (Hz)
fs = single(S.samplingRate);

% Radar positions: matrix [Nrad x 3] where each row is (x, y, z) in meters
radarPos = single(S.radarPositions);
if size(radarPos,2) ~= 3, radarPos = radarPos.'; end
Nrad = size(radarPos, 1);

% Ensure radarData columns correspond to radar positions (transpose if needed)
if size(radarData,2) ~= Nrad, radarData = radarData.'; end
Nt = size(radarData, 1);

% --------------------- Time axis adjustment ------------------------------
% Some input files include a TimeAxis with a non-zero start offset (t0).
% We save t0 here; the actual idxOff is computed after oversampling
% (which modifies fs), so that the mapping tau -> index is correct for the
% upsampled trace:  idx = tau * fs_upsampled + idxOff
t0 = single(0);
if isfield(S,'TimeAxis') && numel(S.TimeAxis) == Nt
    t0 = single(S.TimeAxis(1));
    % If t0 is in nanoseconds (large value), convert to seconds
    if t0 > 1e-3, t0 = t0 * 1e-9; end
end

% Angular frequency for coherent phase correction (used when data is complex I/Q)
omega_c = single(2 * pi * fc);

% If the original radar data is complex (I/Q demodulated), we apply a phase
% rotation exp(j * omega_c * tau) during backprojection to compensate for
% the carrier frequency removal. For real-valued data, this is not needed.
applyPhase = ~isreal(S.radarData);

fprintf('Data: Nt=%d, Nrad=%d, fs=%.1f MHz\n', Nt, Nrad, double(fs)/1e6);
fprintf('Params: n2=%.1f, n3=%.1f, d1=%.2fm\n', n2, n3, depth1);
fprintf('Hybrid: %d DEM-IVS (%dx%d) + %d Newton\n', nCPIVSIter, NgridXY, NgridXY, nNewtonIter);

% --------------------- Analytic signal + oversampling --------------------
% If the signal is real-valued, convert it to an analytic signal via FFT.
% The filter H doubles positive frequencies and zeros negative ones.
% This eliminates phase ambiguity and enables coherent backprojection.
if isreal(radarData)
    H = [1; 2*ones(floor((Nt-1)/2),1,'single'); ...
         1+mod(Nt+1,2); zeros(floor((Nt-1)/2),1,'single')];
    radarData = ifft(fft(radarData,[],1).*H,[],1);
end

% ----- FFT-based oversampling (zero-padding in frequency domain) -----
% Upsample the radar traces by 'oversampFactor' using spectral zero-padding.
% This is equivalent to ideal sinc interpolation and provides sub-sample
% accuracy for the time-delay lookup during backprojection.
%
% Method: take the FFT of each trace, pad the spectrum symmetrically with
% zeros (inserting zeros at the Nyquist boundary), then take the IFFT of
% the longer sequence. The result has oversampFactor * Nt samples with the
% same physical time extent, and amplitudes are scaled to preserve the
% signal level.
if oversampFactor > 1
    Nt_orig = Nt;                        % Original number of samples
    Nt_up   = Nt_orig * oversampFactor;  % Upsampled length

    F = fft(radarData, [], 1);           % [Nt_orig x Nrad] complex spectrum

    % Split the spectrum at the Nyquist frequency and insert zeros between
    % the positive and negative frequency halves.
    halfN = floor(Nt_orig / 2);
    nZeros = Nt_up - Nt_orig;            % Number of zero-frequency bins to insert

    if mod(Nt_orig, 2) == 0
        % Even-length: split the Nyquist bin equally between pos & neg halves
        F_up = [F(1:halfN, :); ...
                F(halfN+1, :) / 2; ...
                zeros(nZeros - 1, Nrad, 'single'); ...
                F(halfN+1, :) / 2; ...
                F(halfN+2:end, :)];
    else
        % Odd-length: no Nyquist bin ambiguity
        F_up = [F(1:halfN+1, :); ...
                zeros(nZeros, Nrad, 'single'); ...
                F(halfN+2:end, :)];
    end

    radarData = ifft(F_up, [], 1) * single(oversampFactor);  % Scale to preserve amplitude
    Nt = Nt_up;                          % Update Nt to reflect oversampled length
    fs = fs * single(oversampFactor);    % Update sampling rate accordingly

    fprintf('Oversampling: %dx (Nt: %d -> %d, fs: %.1f -> %.1f MHz)\n', ...
        oversampFactor, Nt_orig, Nt, double(fs)/1e6/oversampFactor, double(fs)/1e6);
end

% Append one zero sample at the end for safe linear interpolation.
% This allows us to always access index iFl+1 without bounds checking.
radarData = [radarData; zeros(1,Nrad,'single')];
Nt_pad = Nt + 1;

% Now compute idxOff using the (possibly upsampled) fs:
%   idx = tau * fs + idxOff  maps physical time tau to a 1-based index
idxOff = single(1 - t0 * fs);

%% ======================== DEM + VOXEL GRID ===============================
% Extract the DEM (Digital Elevation Model): x/y axis vectors and z elevation
% matrix. The extractDEM function standardizes orientation (x increasing,
% y increasing, Z as [Ny x Nx]).
[xD, yD, zD] = extractDEM(S.DEM);

% Store the DEM in a struct with vectorized elevation and grid metadata
% for fast linear-index access during bilinear interpolation on GPU/CPU.
dem.Zv = single(zD(:));           % Elevation values as a column vector

dem.Nx = int32(size(zD,2));       % Number of DEM columns (x direction)
dem.Ny = int32(size(zD,1));       % Number of DEM rows (y direction)

dem.x0 = single(xD(1));          % Minimum x coordinate
dem.y0 = single(yD(1));          % Minimum y coordinate

dem.xMax = single(xD(end));      % Maximum x coordinate
dem.yMax = single(yD(end));      % Maximum y coordinate

dem.dx = single(xD(2)-xD(1));    % DEM grid spacing in x
dem.dy = single(yD(2)-yD(1));    % DEM grid spacing in y

dem.invDx = single(1/dem.dx);    % Precomputed reciprocal for fast division
dem.invDy = single(1/dem.dy);    % Precomputed reciprocal for fast division

% Build the full 3D voxel grid (Nx * Ny * Nz points)
Nx = numel(xGrid); Ny = numel(yGrid); Nz = numel(zGrid);
[Xg, Yg, Zg] = ndgrid(xGrid, yGrid, zGrid);
Xg = Xg(:);   % Flatten to column vectors for vectorized processing
Yg = Yg(:);
Zg = Zg(:);

% Compute the ground surface elevation at each voxel's (x,y) position.
% We use CPU interpolation here because we need to filter out invalid
% voxels (above the surface) before transferring data to the GPU.
zS1 = demCPU(dem, Xg, Yg);

% A voxel is valid only if it lies below the ground surface (inside the soil).
% Voxels above the surface are excluded from backprojection.
valid = Zg < zS1;
vidx = find(valid);
Nv = numel(vidx);

% Extract only the valid (subsurface) voxel coordinates
Xv = Xg(vidx); Yv = Yg(vidx); Zv = Zg(vidx);

% The second interface (DEM2, between soil layers 1 and 2) is located
% at a constant depth 'depth1' below DEM1 at each (x,y) position.
zS1v = zS1(vidx);
zS2v = zS1v - depth1;

% Classify each valid voxel: isL3 = true means the voxel is in layer 3
% (below DEM2), requiring refraction at both interfaces.
isL3 = Zv < zS2v;

fprintf('Grid: %dx%dx%d, Valid: %d voxels (L2=%d, L3=%d)\n', ...
    Nx, Ny, Nz, Nv, sum(~isL3), sum(isL3));

%% ======================== GPU SETUP ======================================
% Transfer DEM parameters to GPU memory (struct "demG") so that all DEM
% interpolation during the main loop runs entirely on the GPU.
demG.Zv = gpuArray(dem.Zv);
demG.Nx = dem.Nx; demG.Ny = dem.Ny;
demG.x0 = dem.x0; demG.y0 = dem.y0;
demG.xMax = dem.xMax; demG.yMax = dem.yMax;
demG.dx = dem.dx; demG.dy = dem.dy;
demG.invDx = dem.invDx; demG.invDy = dem.invDy;

% --------------------- DEM-IVS candidate grid ----------------------------
% Build normalized offsets for the candidate search grid used in DEM-IVS.
% dxOff, dyOff have shape [1 x 1 x Nc] where Nc = NgridXY^2 = 9.
% Each candidate displaces the current refraction point (x1,y1) or (x2,y2)
% by rad * dxOff and rad * dyOff, creating a 2D grid of test positions.
[gx, gy] = meshgrid(linspace(-1, 1, NgridXY));
dxOff = gpuArray(reshape(single(gx(:)), 1, 1, []));
dyOff = gpuArray(reshape(single(gy(:)), 1, 1, []));
Nc = numel(dxOff);

% Spatial bounds of the DEM domain (used to clamp candidate positions so
% they remain within the valid interpolation region)
xMin = demG.x0;
xMax_ = demG.xMax;
yMin = demG.y0;
yMax_ = demG.yMax;

%% ======================== BATCH SIZE ESTIMATION ==========================
% The backprojection is processed in blocks (batches) to avoid exceeding
% GPU memory. We estimate the maximum number of voxels and radar positions
% that can be processed simultaneously based on available VRAM.
availMem = 0.45 * g.AvailableMemory; % Use ~45% of free VRAM as safety margin
bytesPerArray = 4;                   % single precision = 4 bytes per element
numArrays = 15;                      % Approximate number of large intermediate arrays

maxElements = availMem / (bytesPerArray * numArrays);
maxNvB = floor(maxElements / Nc);    % Account for DEM-IVS expansion (Nc candidates per point)

% Radar batch size (number of radar positions per inner loop iteration).
% 384 balances GPU occupancy vs. kernel launch overhead.
radarBatchSize = 384;

% Voxel chunk size: how many voxels are processed per radar batch.
% Clamped to [1000, 8000] to avoid extreme cases.
voxelChunkSize = floor(maxNvB / radarBatchSize);
voxelChunkSize = max(1000, min(8000, voxelChunkSize));

% Total number of batch iterations (outer voxel chunks x inner radar batches)
nVoxelChunks = ceil(Nv / voxelChunkSize);
nRadarBatches = ceil(Nrad / radarBatchSize);
totalBatches = nVoxelChunks * nRadarBatches;

fprintf('Batching: %d x %d = %d batches\n', nVoxelChunks, nRadarBatches, totalBatches);

%% ======================== DATA TRANSFER TO GPU ===========================
% Transfer the main data arrays to GPU memory once (they remain there for
% the entire computation, accessed by each batch iteration).
radarData_g = gpuArray(radarData);
radarPos_g  = gpuArray(radarPos);

%% ======================== CONSTANTS =====================================
EPS = single(1e-9); % Small epsilon to prevent division by zero

%% ======================== MAIN PROCESSING LOOP ===========================
fprintf('\n--- Processing ---\n');
tStart = tic;

% ImgAccum: accumulates the coherent backprojection result (averaged over
% all Nrad radar positions) for valid voxels only. Final shape: [Nv x 1].
ImgAccum = zeros(Nv, 1, 'single');

batchCount = 0;
lastPrintTime = 0;

% OUTER LOOP: iterate over chunks of voxels (to fit within GPU memory)
for vc = 1:nVoxelChunks
    v1 = (vc-1)*voxelChunkSize + 1;
    v2 = min(vc*voxelChunkSize, Nv);
    nv = v2 - v1 + 1;

    % Transfer the current voxel chunk to GPU memory
    Xc = gpuArray(Xv(v1:v2));       % x-coordinates [nv x 1]
    Yc = gpuArray(Yv(v1:v2));       % y-coordinates [nv x 1]
    Zc = gpuArray(Zv(v1:v2));       % z-coordinates [nv x 1]
    isL3c = gpuArray(isL3(v1:v2));  % Logical mask: true if voxel is in layer 3

    % Accumulator for this chunk (complex because coherent BP sums phase and amplitude)
    chunkAccum = gpuArray(complex(zeros(nv, 1, 'single')));

    % INNER LOOP: iterate over batches of radar positions
    for rb = 1:nRadarBatches
        r1 = (rb-1)*radarBatchSize + 1;
        r2 = min(rb*radarBatchSize, Nrad);
        B  = r2 - r1 + 1; % Number of radar positions in this batch

        batchCount = batchCount + 1;

        % Reshape radar positions to [1 x B] for implicit broadcasting with [nv x 1] voxels
        rx = reshape(radarPos_g(r1:r2, 1), 1, B);
        ry = reshape(radarPos_g(r1:r2, 2), 1, B);
        rz = reshape(radarPos_g(r1:r2, 3), 1, B);

        % Extract the radar traces for this batch: [Nt_pad x B] matrix
        traces = radarData_g(:, r1:r2);

        % ==============================================================
        % PHASE 1: DEM-IVS (GLOBAL INITIALIZATION) FOR P1/P2
        % ==============================================================
        % Goal: find a good initial estimate for the refraction points
        % (P1 on DEM1 and P2 on DEM2) without getting trapped in poor
        % local minima.
        %
        % Algorithm:
        %   1. Initialize P1 and P2 at the voxel's (x,y) position
        %   2. For each iteration, test Nc candidate points around the
        %      current estimate within a search radius "rad"
        %   3. Alternate optimization between P1 and P2 (P1, P2, P1, P2...)
        %   4. Shrink the search radius each step for progressive refinement
        %
        % Cost function = optical path length (sum of distances x refractive indices):
        %   Layer 2 voxels: cost = n_air * |R-P1| + n2 * |P1-V|
        %   Layer 3 voxels: cost = n_air * |R-P1| + n2 * |P1-P2| + n3 * |P2-V|
        % ==============================================================

        % Initialize P1 and P2 at the voxel position, replicated for all B radar positions
        x1 = repmat(Xc, 1, B);
        y1 = repmat(Yc, 1, B);
        x2 = x1;
        y2 = y1;

        % Initial search radius: proportional to horizontal distance between
        % voxel and radar. Farther radars require larger initial search radii
        % because the refraction point is displaced further from the voxel.
        Lh  = sqrt((Xc - rx).^2 + (Yc - ry).^2) + EPS;       % [nv x B]
        rad = min(radMax, max(radMin, radFac * Lh));            % [nv x B]

        % bestCost: tracks the lowest optical path cost found so far for
        % each (voxel, radar) pair. Initialized to infinity.
        bestCost = gpuArray.inf(nv, B, 'single');

        % bX1/bY1/bX2/bY2: store the best P1 and P2 coordinates found so far
        bX1 = x1; bY1 = y1;
        bX2 = x2; bY2 = y2;

        % Index grids for converting (voxel, radar, candidate) triplets to linear indices
        [ii, jj] = ndgrid(1:nv, 1:B);

        % Total iterations = 2 * nCPIVSIter because we alternate P1/P2:
        % e.g., nCPIVSIter=2 => 4 steps: optimize P1, P2, P1, P2
        for iter = 1:(2*nCPIVSIter)
            doP1 = (mod(iter, 2) == 1); % Odd iterations optimize P1; even optimize P2

            if doP1
                % ===================== Optimize P1 =====================
                % Generate Nc candidate positions around the current P1:
                %   xc = x1 + rad * dxOff,  yc = y1 + rad * dyOff
                % Result shape: [nv x B x Nc] (voxels x radars x candidates)
                xc = x1 + rad .* dxOff;
                yc = y1 + rad .* dyOff;

                % Clamp candidate coordinates to stay within the DEM domain
                xc = min(max(xc, xMin), xMax_);
                yc = min(max(yc, yMin), yMax_);

                % Evaluate DEM elevation at each candidate (P1 lies on the surface)
                zc = demGPU_batch(demG, xc, yc);

                % Distance from radar to candidate P1 (air segment): [nv x B x Nc]
                Ra = sqrt((xc - rx).^2 + (yc - ry).^2 + (zc - rz).^2);

                % ---------- Cost for layer 2 voxels ----------
                % Distance from candidate P1 to voxel V (soil layer 1 segment)
                Rs1_l2  = sqrt((Xc - xc).^2 + (Yc - yc).^2 + (Zc - zc).^2);
                cost_l2 = n_air * Ra + n2 * Rs1_l2;

                % ---------- Cost for layer 3 voxels ----------
                % Current P2 position on DEM2 (DEM1 elevation minus depth1)
                z2_cur   = demGPU(demG, x2, y2) - depth1; % [nv x B]
                % Distance from current P2 to voxel V (soil layer 2 segment)
                Rs2_fixed = sqrt((Xc - x2).^2 + (Yc - y2).^2 + (Zc - z2_cur).^2); % [nv x B]

                % Expand P2 to candidate dimension Nc for P1(candidate)->P2(fixed) distance
                x2_e = reshape(x2, nv, B, 1);
                y2_e = reshape(y2, nv, B, 1);
                z2_e = reshape(z2_cur, nv, B, 1);

                % Distance from candidate P1 to fixed P2 (soil layer 1 segment for L3)
                Rs1_l3 = sqrt((x2_e - xc).^2 + (y2_e - yc).^2 + (z2_e - zc).^2);

                % Total cost for layer 3: air(R->P1) + soil1(P1->P2) + soil2(P2->V)
                cost_l3 = n_air * Ra + n2 * Rs1_l3 + n3 * reshape(Rs2_fixed, nv, B, 1);

                % Select cost based on voxel layer membership (L2 or L3)
                isL3_e = reshape(isL3c, nv, 1, 1); % [nv x 1 x 1]
                cost = cost_l2 .* single(~isL3_e) + cost_l3 .* single(isL3_e);

                % Enforce geometric validity: the voxel must be below the surface at P1
                cost(Zc >= zc) = inf;

                % For each (voxel, radar) pair, pick the candidate with minimum cost
                [minCost, bestIdx] = min(cost, [], 3);

                % Update the best solution only where improvement was found
                improved = minCost < bestCost;
                bestCost(improved) = minCost(improved);

                % Convert 3D candidate index to linear index for extracting xc/yc values
                linIdx = ii + (jj-1)*nv + (bestIdx-1)*nv*B;
                bX1(improved) = xc(linIdx(improved));
                bY1(improved) = yc(linIdx(improved));

                % Advance the current P1 estimate to the best found so far
                x1 = bX1; y1 = bY1;

            else
                % ===================== Optimize P2 =====================
                % Only relevant for voxels in layer 3 (below DEM2)
                if any(isL3c)
                    % Generate candidates around the current P2 on the DEM2 interface
                    xc = x2 + rad .* dxOff;
                    yc = y2 + rad .* dyOff;
                    xc = min(max(xc, xMin), xMax_);
                    yc = min(max(yc, yMin), yMax_);

                    % P2 elevation = DEM1 minus depth1 at candidate position
                    zc = demGPU_batch(demG, xc, yc) - depth1;

                    % Current P1 elevation on DEM1 surface (held fixed during P2 optimization)
                    z1_cur = demGPU(demG, x1, y1);

                    % Air segment distance: R -> P1 (fixed during P2 optimization)
                    Ra_fixed = sqrt((x1 - rx).^2 + (y1 - ry).^2 + (z1_cur - rz).^2);

                    % Expand fixed P1 to candidate dimension Nc for P1->P2(candidate) distance
                    x1_e = reshape(x1, nv, B, 1);
                    y1_e = reshape(y1, nv, B, 1);
                    z1_e = reshape(z1_cur, nv, B, 1);

                    % Distance from fixed P1 to candidate P2 (soil layer 1 segment)
                    Rs1 = sqrt((xc - x1_e).^2 + (yc - y1_e).^2 + (zc - z1_e).^2);

                    % Distance from candidate P2 to voxel V (soil layer 2 segment)
                    Rs2 = sqrt((Xc - xc).^2 + (Yc - yc).^2 + (Zc - zc).^2);

                    % Total layer 3 cost: air(R->P1) + soil1(P1->P2) + soil2(P2->V)
                    cost = n_air * reshape(Ra_fixed, nv, B, 1) + n2 * Rs1 + n3 * Rs2;

                    % Enforce geometric validity:
                    %   - Voxel must be below P2 (Zc < zc)
                    %   - P2 must be below P1 (zc < z1_e)
                    %   - Voxel must actually be in layer 3 (isL3)
                    isL3_e = reshape(isL3c, nv, 1, 1);
                    validP2 = (Zc < zc) & (zc < z1_e) & isL3_e;
                    cost(~validP2) = inf;

                    % Select best candidate per (voxel, radar) pair
                    [minCost, bestIdx] = min(cost, [], 3);
                    improved = minCost < bestCost;
                    bestCost(improved) = minCost(improved);

                    % Update P2 coordinates where improvement was found
                    linIdx = ii + (jj-1)*nv + (bestIdx-1)*nv*B;
                    bX2(improved) = xc(linIdx(improved));
                    bY2(improved) = yc(linIdx(improved));

                    x2 = bX2; y2 = bY2;
                end
            end

            % Shrink the search radius for the next iteration (progressive refinement)
            rad = rad * shrink;
        end

        % After DEM-IVS: use the best refraction points found as the
        % starting point for Newton-Raphson local refinement.
        x1 = bX1; y1 = bY1;
        x2 = bX2; y2 = bY2;

        % ==============================================================
        % PHASE 2: NEWTON-RAPHSON LOCAL REFINEMENT
        % ==============================================================
        % Now that DEM-IVS has placed us near the true solution, Newton-
        % Raphson efficiently solves Snell's law conditions with quadratic
        % convergence.
        %
        % The method alternates between refining P1 and P2, solving a
        % 2x2 nonlinear system at each step: F1(x,y) = 0, F2(x,y) = 0,
        % where F1 and F2 are the x- and y-projected Snell's law residuals
        % that incorporate the DEM surface gradient (hx = dz/dx, hy = dz/dy).
        %
        % Stability measures:
        %   - Add small epsilon to denominators (prevent division by zero)
        %   - Clamp step magnitude to max_step (prevent divergence)
        %   - Clamp updated positions within the DEM domain
        % ==============================================================

        % Expand voxel and radar coordinates to full [nv x B] matrices
        % to avoid complex broadcasting inside the Newton loop.
        Vx = repmat(Xc, 1, B);
        Vy = repmat(Yc, 1, B);
        Vz = repmat(Zc, 1, B);
        Rx = repmat(rx, nv, 1);
        Ry = repmat(ry, nv, 1);
        Rz = repmat(rz, nv, 1);

        isL3_e = repmat(isL3c, 1, B); % Layer 3 mask, expanded to [nv x B]
        isL2_e = ~isL3_e;             % Layer 2 mask (complement)

        for iter = 1:nNewtonIter
            optimizeP1 = (mod(iter, 2) == 1); % Alternate: odd=P1, even=P2

            if optimizeP1
                % ===================== Newton for P1 ===================
                % Evaluate DEM elevation and gradient at current P1:
                % z1 = elevation, hx1 = dz/dx (x-slope), hy1 = dz/dy (y-slope)
                [z1, hx1, hy1] = demGPU_with_grad(demG, x1, y1);

                % P2 elevation on the second interface (for layer 3 voxels)
                z2 = demGPU(demG, x2, y2) - depth1;

                % Vector from radar R to P1 and its Euclidean distance
                dx_r = x1 - Rx; dy_r = y1 - Ry; dz_r = z1 - Rz;
                r1 = sqrt(dx_r.^2 + dy_r.^2 + dz_r.^2) + EPS;

                % Vector from P1 to the "target" point:
                %   Layer 2 voxels: target = voxel V
                %   Layer 3 voxels: target = P2 (next refraction point)
                dx_2 = (x1 - Vx) .* single(isL2_e) + (x1 - x2) .* single(isL3_e);
                dy_2 = (y1 - Vy) .* single(isL2_e) + (y1 - y2) .* single(isL3_e);
                dz_2 = (z1 - Vz) .* single(isL2_e) + (z1 - z2) .* single(isL3_e);
                r2 = sqrt(dx_2.^2 + dy_2.^2 + dz_2.^2) + EPS;

                % Snell's law residuals (x and y components), accounting for the
                % tilted surface via DEM gradient. A1/B1 are the air-side terms;
                % A2/B2 are the soil-side terms, both projected onto the surface.
                A1 = dx_r + dz_r .* hx1;
                B1 = dy_r + dz_r .* hy1;
                A2 = dx_2 + dz_2 .* hx1;
                B2 = dy_2 + dz_2 .* hy1;

                % F = 0 is the vectorial Snell's law condition (projected form)
                F1 = n_air * A1 ./ r1 + n2 * A2 ./ r2;
                F2 = n_air * B1 ./ r1 + n2 * B2 ./ r2;

                % Jacobian matrix (partial derivatives of F1, F2 w.r.t. x, y)
                hx2 = hx1.^2; hy2 = hy1.^2; hxy = hx1 .* hy1;
                r1_sq = r1.^2; r2_sq = r2.^2;

                J11 = n_air*(1+hx2-A1.^2./r1_sq)./r1 + n2*(1+hx2-A2.^2./r2_sq)./r2;
                J22 = n_air*(1+hy2-B1.^2./r1_sq)./r1 + n2*(1+hy2-B2.^2./r2_sq)./r2;
                J12 = n_air*(hxy-A1.*B1./r1_sq)./r1 + n2*(hxy-A2.*B2./r2_sq)./r2;

                % Solve 2x2 linear system: J * [dx; dy] = -[F1; F2]
                % Using Cramer's rule (symmetric Jacobian: J21 = J12)
                det_J  = J11.*J22 - J12.^2 + EPS;
                delta_x = (J12.*F2 - J22.*F1) ./ det_J;
                delta_y = (J12.*F1 - J11.*F2) ./ det_J;

                % Clamp step magnitude to prevent divergence in regions with
                % complex surface geometry (sharp DEM gradients or ill-conditioned Jacobian)
                step_mag = sqrt(delta_x.^2 + delta_y.^2);
                max_step = single(0.2);
                scale = min(single(1), max_step ./ (step_mag + EPS));

                % Apply the update and clamp P1 within the DEM domain
                x1 = max(xMin, min(xMax_, x1 + scale .* delta_x));
                y1 = max(yMin, min(yMax_, y1 + scale .* delta_y));

            else
                % ===================== Newton for P2 ===================
                % Only affects voxels in layer 3 (below DEM2)
                z1 = demGPU(demG, x1, y1);
                [z2_s1, hx2, hy2] = demGPU_with_grad(demG, x2, y2);
                z2 = z2_s1 - depth1;

                % Vector and distance from P1 to P2 (soil layer 1 segment)
                dx_12 = x2 - x1; dy_12 = y2 - y1; dz_12 = z2 - z1;
                r12 = sqrt(dx_12.^2 + dy_12.^2 + dz_12.^2) + EPS;

                % Vector and distance from P2 to voxel V (soil layer 2 segment)
                dx_2v = x2 - Vx; dy_2v = y2 - Vy; dz_2v = z2 - Vz;
                r2v = sqrt(dx_2v.^2 + dy_2v.^2 + dz_2v.^2) + EPS;

                % Snell's law terms projected onto the tilted DEM2 surface at P2
                A1 = dx_12 + dz_12 .* hx2;
                B1 = dy_12 + dz_12 .* hy2;
                A2 = dx_2v + dz_2v .* hx2;
                B2 = dy_2v + dz_2v .* hy2;

                % Snell's law residuals at the n2->n3 interface (only for L3 voxels)
                F1 = (n2 * A1 ./ r12 + n3 * A2 ./ r2v) .* single(isL3_e);
                F2 = (n2 * B1 ./ r12 + n3 * B2 ./ r2v) .* single(isL3_e);

                % Jacobian at the DEM2 interface
                hx2_sq = hx2.^2; hy2_sq = hy2.^2; hxy2 = hx2 .* hy2;
                r12_sq = r12.^2; r2v_sq = r2v.^2;

                % For layer 2 voxels, add +isL2_e to the diagonal to keep the
                % Jacobian well-conditioned (effectively: "don't move P2" for L2 voxels)
                J11 = n2*(1+hx2_sq-A1.^2./r12_sq)./r12 + n3*(1+hx2_sq-A2.^2./r2v_sq)./r2v + single(isL2_e);
                J22 = n2*(1+hy2_sq-B1.^2./r12_sq)./r12 + n3*(1+hy2_sq-B2.^2./r2v_sq)./r2v + single(isL2_e);
                J12 = n2*(hxy2-A1.*B1./r12_sq)./r12 + n3*(hxy2-A2.*B2./r2v_sq)./r2v;

                det_J  = J11.*J22 - J12.^2 + EPS;
                delta_x = (J12.*F2 - J22.*F1) ./ det_J;
                delta_y = (J12.*F1 - J11.*F2) ./ det_J;

                % Clamp step magnitude
                step_mag = sqrt(delta_x.^2 + delta_y.^2);
                max_step = single(0.2);
                scale = min(single(1), max_step ./ (step_mag + EPS));

                % Apply the update and clamp P2 within the DEM domain
                x2 = max(xMin, min(xMax_, x2 + scale .* delta_x));
                y2 = max(yMin, min(yMax_, y2 + scale .* delta_y));
            end
        end

        % ==============================================================
        % FINAL OPTICAL PATH LENGTH COMPUTATION
        % ==============================================================
        % After Newton-Raphson refinement of P1 and P2, compute the final
        % optical path length for each (voxel, radar) pair:
        %   pathLen = sum_i { n_i * geometric_distance_in_medium_i }
        %
        % The optical path length determines the round-trip travel time
        % (tau = 2 * pathLen / c) used to index into the radar trace.
        z1 = demGPU(demG, x1, y1);
        z2 = demGPU(demG, x2, y2) - depth1;

        % Geometric segment distances along the ray path:
        Ra     = sqrt((x1 - Rx).^2 + (y1 - Ry).^2 + (z1 - Rz).^2);           % Air segment: Radar R -> P1 on DEM1
        Rs1_l2 = sqrt((Vx - x1).^2 + (Vy - y1).^2 + (Vz - z1).^2);           % Soil1 segment (layer 2 voxels): P1 -> Voxel V
        Rs1_l3 = sqrt((x2 - x1).^2 + (y2 - y1).^2 + (z2 - z1).^2);           % Soil1 segment (layer 3 voxels): P1 -> P2 on DEM2
        Rs2_l3 = sqrt((Vx - x2).^2 + (Vy - y2).^2 + (Vz - z2).^2);           % Soil2 segment (layer 3 voxels): P2 -> Voxel V

        % Total optical path length (weighted by refractive indices):
        %   Layer 2 voxels: n_air * |R-P1| + n2 * |P1-V|
        %   Layer 3 voxels: n_air * |R-P1| + n2 * |P1-P2| + n3 * |P2-V|
        pathLen = (n_air*Ra + n2*Rs1_l2) .* single(isL2_e) + ...
                  (n_air*Ra + n2*Rs1_l3 + n3*Rs2_l3) .* single(isL3_e);

        % ==============================================================
        % SIGNAL INTERPOLATION (BACKPROJECTION CORE)
        % ==============================================================
        % Convert optical path length to round-trip travel time:
        %   tau = 2 * pathLen / c  (factor 2 for two-way propagation)
        % Then map tau to a fractional sample index in the radar trace:
        %   idxF = tau * fs + idxOff  (continuous index, 1-based)
        tau  = 2 * pathLen / c;
        idxF = tau * fs + idxOff;

        % Determine which indices fall within the valid trace range [1, Nt]
        validI = isfinite(idxF) & (idxF >= 1) & (idxF <= Nt);

        % Linear interpolation between adjacent samples:
        %   iFl = floor index (integer), iCe = ceil index = iFl + 1
        %   wCe = fractional weight for the ceiling sample
        iFl = int32(max(1, min(Nt, floor(idxF))));
        iCe = int32(min(Nt_pad, iFl + 1));
        wCe = idxF - single(iFl); % Fractional weight for the ceiling (upper) sample

        % Compute linear index offsets for accessing columns of the trace matrix.
        % 'traces' is stored as [Nt_pad x B], so column j starts at (j-1)*Nt_pad.
        base = int32(0:B-1) * int32(Nt_pad);

        % Interpolated (complex) sample value at time tau for each (voxel, radar) pair
        val = traces(iFl + base) .* (1 - wCe) + traces(iCe + base) .* wCe;

        % If the input data is I/Q demodulated (complex baseband), apply coherent
        % phase correction: multiply by exp(j * omega_c * tau) to restore the
        % carrier phase that was removed during demodulation.
        if applyPhase
            val = val .* exp(1j * omega_c * tau);
        end

        % Zero out contributions from out-of-range samples (invalid indices)
        val(~validI) = 0;

        % Coherent summation: accumulate contributions from all radar positions
        % in this batch (sum across the B radar dimension, column 2)
        chunkAccum = chunkAccum + sum(val, 2);

        % --------------------- Progress display -----------------------
        elapsed = toc(tStart);
        if elapsed - lastPrintTime >= 5 || batchCount == totalBatches
            pct = 100 * batchCount / totalBatches;
            eta = elapsed / batchCount * (totalBatches - batchCount);
            if eta > 60
                etaStr = sprintf('%dm %02.0fs', floor(eta/60), mod(eta,60));
            else
                etaStr = sprintf('%.0fs', eta);
            end
            fprintf('[%5.1f%%] Batch %d/%d | %.1fs | ETA: %s\n', ...
                pct, batchCount, totalBatches, elapsed, etaStr);
            lastPrintTime = elapsed;
        end
    end

    % Transfer the chunk result from GPU back to CPU and normalize by the
    % total number of radar positions (coherent average over all aperture positions)
    ImgAccum(v1:v2) = gather(chunkAccum) / Nrad;
end

% Record total computation time
tTotal = toc(tStart);
if tTotal > 60
    fprintf('  COMPLETED: %dm %.1fs\n', floor(tTotal/60), mod(tTotal,60));
else
    fprintf('  COMPLETED: %.2fs\n', tTotal);
end

%% ======================== OUTPUT ========================================
% Reconstruct the full 3D complex image volume [Nx x Ny x Nz].
% Only the valid (subsurface) voxels contain data; the rest remain zero.
Img = complex(zeros(Nx, Ny, Nz, 'single'));
Img(vidx) = ImgAccum;

% Quick diagnostic metrics: entropy (energy dispersion) and sharpness (peak/mean ratio)
M = abs(ImgAccum); M = M(M > 0);
if ~isempty(M)
    Mn = M / sum(M);
    fprintf('Entropy: %.3f, Sharpness: %.1f\n', -sum(Mn.*log(Mn+eps)), max(M)/mean(M));
end

% Locate the peak (maximum magnitude) in the 3D volume and print its coordinates
[~, mi] = max(abs(Img(:)));
[ix, iy, iz] = ind2sub([Nx, Ny, Nz], mi);
fprintf('Peak: (%.3f, %.3f, %.3f) m\n', xGrid(ix), yGrid(iy), zGrid(iz));

% Build and save both DEM surfaces for use by the quality analysis script:
%   DEM1 = air-soil interface (original DEM elevation)
%   DEM2 = soil1-soil2 interface (DEM1 shifted down by depth1)
xDEM = xD;
yDEM = yD;
zDEM1 = zD;
zDEM2 = zD - double(depth1);  % DEM2 is depth1 below DEM1

% Save all results to a MAT file in the output directory
matDir = fullfile(outputDir, 'mat');
if ~exist(matDir, 'dir'), mkdir(matDir); end
matFile = fullfile(matDir, 'bp_hybrid_results.mat');
save(matFile, 'Img', 'xGrid', 'yGrid', 'zGrid', 'n2', 'n3', 'depth1', 'tTotal', ...
    'xDEM', 'yDEM', 'zDEM1', 'zDEM2', 'oversampFactor', '-v7.3');
fprintf('Saved: %s\n', matFile);

% Generate a quick visualization: three orthogonal slices (XY, XZ, YZ)
% through the peak location, displayed in dB scale with 40 dB dynamic range
plotDir = fullfile(outputDir, 'plots');
if ~exist(plotDir, 'dir'), mkdir(plotDir); end
visualize(Img, xGrid, yGrid, zGrid, 40, n2, n3, depth1);
saveas(gcf, fullfile(plotDir, 'bp_hybrid_slices.png'));
saveas(gcf, fullfile(plotDir, 'bp_hybrid_slices.fig'));
fprintf('Saved plots to: %s\n', plotDir);
end

%% ========================================================================
% DEM INTERPOLATION FUNCTIONS (BILINEAR + GRADIENT)
% ========================================================================

function [z, hx, hy] = demGPU_with_grad(d, xq, yq)
% DEMGPU_WITH_GRAD  Bilinear DEM interpolation with surface gradient (GPU).
%
% Returns:
%   z  - Interpolated DEM elevation at query points (xq, yq)
%   hx - Partial derivative dz/dx (surface slope in x-direction)
%   hy - Partial derivative dz/dy (surface slope in y-direction)
%
% These gradients are essential for Snell's law on tilted surfaces: the
% refraction direction depends on the local surface normal, which is
% determined by (hx, hy).
%
% The DEM is stored as a regular grid with spacing (dx, dy). Bilinear
% interpolation uses the 4 corner values of the grid cell containing
% each query point.

% Convert query coordinates to fractional grid indices (0-based)
tx = (xq - d.x0) * d.invDx;
ty = (yq - d.y0) * d.invDy;

% Integer indices of the top-left corner of the containing cell,
% clamped to valid range [0, N-2] to stay within the grid
j = max(int32(0), min(d.Nx-2, int32(floor(tx))));
i = max(int32(0), min(d.Ny-2, int32(floor(ty))));

% Fractional weights within the cell: (0,0) = top-left corner, (1,1) = bottom-right
ax = tx - single(j);
ay = ty - single(i);

% Convert to 1-based MATLAB indexing
j = j + 1; i = i + 1;
Ny = d.Ny;

% Linear index in column-major order: idx = row + (col-1)*numRows
idx = i + (j-1)*Ny;

% Sample the 4 corner elevations of the grid cell:
%   z00 = bottom-left, z10 = bottom-right, z01 = top-left, z11 = top-right
z00 = d.Zv(idx);
z10 = d.Zv(idx + Ny);
z01 = d.Zv(idx + 1);
z11 = d.Zv(idx + Ny + 1);

% Bilinear interpolation for elevation z
z = (1-ax).*(1-ay).*z00 + ax.*(1-ay).*z10 + (1-ax).*ay.*z01 + ax.*ay.*z11;

% Analytical partial derivatives of the bilinear interpolant:
%   hx = dz/dx = d(bilinear)/dax * (1/dx)
%   hy = dz/dy = d(bilinear)/day * (1/dy)
hx = ((z10 - z00).*(1-ay) + (z11 - z01).*ay) * d.invDx;
hy = ((z01 - z00).*(1-ax) + (z11 - z10).*ax) * d.invDy;
end

function z = demGPU(d, xq, yq)
% DEMGPU  Lightweight bilinear DEM interpolation on GPU (elevation only).
%
% Same algorithm as demGPU_with_grad but without computing the gradient.
% Used when only the surface elevation is needed (e.g., computing segment
% distances), avoiding the overhead of gradient computation.
tx = (xq - d.x0) * d.invDx;
ty = (yq - d.y0) * d.invDy;
j = max(int32(0), min(d.Nx-2, int32(floor(tx))));
i = max(int32(0), min(d.Ny-2, int32(floor(ty))));
ax = tx - single(j);
ay = ty - single(i);
j = j + 1; i = i + 1;
Ny = d.Ny;
idx = i + (j-1)*Ny;

z = (1-ax).*(1-ay).*d.Zv(idx) + ax.*(1-ay).*d.Zv(idx+Ny) + ...
    (1-ax).*ay.*d.Zv(idx+1) + ax.*ay.*d.Zv(idx+Ny+1);
end

function z = demGPU_batch(d, xq, yq)
% DEMGPU_BATCH  Bilinear DEM interpolation for multi-dimensional arrays.
%
% Handles inputs with extra dimensions (e.g., [nv x B x Nc] during the
% DEM-IVS candidate evaluation). Flattens to a 1D vector, interpolates
% using the same bilinear scheme, then reshapes back to the original size.
origSize = size(xq);
xq = xq(:); yq = yq(:);

tx = (xq - d.x0) * d.invDx;
ty = (yq - d.y0) * d.invDy;
j = max(int32(0), min(d.Nx-2, int32(floor(tx))));
i = max(int32(0), min(d.Ny-2, int32(floor(ty))));
ax = tx - single(j);
ay = ty - single(i);
j = j + 1; i = i + 1;
Ny = d.Ny;
idx = i + (j-1)*Ny;

z = (1-ax).*(1-ay).*d.Zv(idx) + ax.*(1-ay).*d.Zv(idx+Ny) + ...
    (1-ax).*ay.*d.Zv(idx+1) + ax.*ay.*d.Zv(idx+Ny+1);

z = reshape(z, origSize);
end

function z = demCPU(d, xq, yq)
% DEMCPU  Bilinear DEM interpolation on CPU using double precision.
%
% Used during the pre-processing stage (before GPU transfer) to determine
% which voxels lie below the ground surface. Double precision ensures
% robust boundary handling when classifying voxels near the DEM surface.
tx = double((xq - d.x0) * d.invDx);
ty = double((yq - d.y0) * d.invDy);
j = max(0, min(double(d.Nx-2), floor(tx)));
i = max(0, min(double(d.Ny-2), floor(ty)));
ax = tx - j; ay = ty - i;
j = int32(j)+1; i = int32(i)+1;
Ny = double(d.Ny); Z = double(d.Zv);

z = single((1-ax).*(1-ay).*Z(i+(j-1)*Ny) + ax.*(1-ay).*Z(i+j*Ny) + ...
           (1-ax).*ay.*Z(i+1+(j-1)*Ny) + ax.*ay.*Z(i+1+j*Ny));
end

function S = loadData(f)
% LOADDATA  Load a MAT file and handle nested struct containers.
%
% Some MAT files wrap the actual data inside one or more levels of a
% struct field named 'S'. This function unwraps such nesting automatically,
% returning the innermost data structure regardless of how many levels
% of wrapping are present.
r = load(f);
if isfield(r,'S'), S = r.S;
    while isstruct(S) && isfield(S,'S'), S = S.S; end
else, S = r;
end
end

function [x, y, Z] = extractDEM(D)
% EXTRACTDEM  Extract DEM data and ensure consistent orientation.
%
% Handles two common DEM formats:
%   1) D.x and D.y are vectors, D.z is a 2D matrix
%   2) D.x and D.y are meshgrid-style 2D matrices
%
% Guarantees the output satisfies:
%   - x is a row vector with strictly increasing values (left to right)
%   - y is a column vector with strictly increasing values (top to bottom)
%   - Z has size [numel(y) x numel(x)]  (rows = y, columns = x)
if isvector(D.x) && isvector(D.y)
    x = double(D.x(:)).'; y = double(D.y(:)); Z = double(D.z);
    if size(Z,1)==numel(x) && size(Z,2)==numel(y), Z = Z.'; end
else
    x = double(D.x(1,:)); y = double(D.y(:,1)); Z = double(D.z);
    if ~isequal(size(Z), [numel(y), numel(x)]), Z = Z.'; end
end
% Ensure x is monotonically increasing (flip if descending)
if any(diff(x)<0), x = fliplr(x); Z = Z(:, end:-1:1); end
% Ensure y is monotonically increasing (flip if descending)
if any(diff(y)<0), y = flipud(y); Z = Z(end:-1:1, :); end
end

function visualize(Img, xG, yG, zG, dyn, n2, n3, d1)
% VISUALIZE  Quick 3-slice visualization of the backprojection result.
%
% Displays three orthogonal slices (XY, XZ, YZ) through the peak location
% of the 3D image volume. The image is normalized to the peak value and
% displayed in dB scale with a dynamic range of 'dyn' dB.
%
% Inputs:
%   Img  - Complex 3D image volume [Nx x Ny x Nz]
%   xG, yG, zG - Axis vectors for each dimension
%   dyn  - Dynamic range in dB (e.g., 40 means display from -40 dB to 0 dB)
%   n2, n3, d1 - Parameters shown in the figure title
M = abs(Img); pk = max(M(:));
if pk <= 0, return; end
MdB = 20*log10(M/pk + eps); MdB(MdB < -dyn) = -dyn;

[~, mi] = max(M(:));
[ix, iy, iz] = ind2sub(size(M), mi);

figure('Color', 'w', 'Position', [100 100 1400 400]); colormap('jet');

subplot(1,3,1);
imagesc(xG, yG, MdB(:,:,iz).'); axis xy image; colorbar; clim([-dyn 0]);
title(sprintf('XY (Z=%.2f)', zG(iz))); xlabel('x'); ylabel('y');

subplot(1,3,2);
imagesc(xG, zG, squeeze(MdB(:,iy,:)).'); axis xy image; colorbar; clim([-dyn 0]);
title(sprintf('XZ (Y=%.2f)', yG(iy))); xlabel('x'); ylabel('z');

subplot(1,3,3);
imagesc(yG, zG, squeeze(MdB(ix,:,:)).'); axis xy image; colorbar; clim([-dyn 0]);
title(sprintf('YZ (X=%.2f)', xG(ix))); xlabel('y'); ylabel('z');

sgtitle(sprintf('Hybrid BP: n2=%.2f, n3=%.2f, d1=%.2fm', n2, n3, d1));
end
