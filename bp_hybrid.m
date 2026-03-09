function [Img, tTotal] = bp_hybrid(n2_in, n3_in, depth1_in, outputDir)
% =========================================================================
% HYBRID MULTI-LAYER SAR BACKPROJECTION (OPTIMIZED)
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
% Performance optimizations vs. original:
%   - Newton section uses implicit broadcasting instead of repmat,
%     eliminating 7 large [nv x B] temporary arrays per iteration.
%   - Pre-computed single-precision masks avoid repeated logical-to-single
%     conversions inside hot loops.
%   - anyL3 guard skips Newton P2 iterations for L2-only chunks.
%   - ndgrid index arrays replaced with broadcasting-based linear indexing.
%   - Larger batch sizes from reduced memory footprint.
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
if isreal(radarData)
    H = [1; 2*ones(floor((Nt-1)/2),1,'single'); ...
         1+mod(Nt+1,2); zeros(floor((Nt-1)/2),1,'single')];
    radarData = ifft(fft(radarData,[],1).*H,[],1);
end

% ----- FFT-based 4x oversampling (zero-padding in frequency domain) -----
if oversampFactor > 1
    Nt_orig = Nt;                        % Original number of samples
    Nt_up   = Nt_orig * oversampFactor;  % Upsampled length

    F = fft(radarData, [], 1);           % [Nt_orig x Nrad] complex spectrum

    halfN = floor(Nt_orig / 2);
    nZeros = Nt_up - Nt_orig;            % Number of zero-frequency bins to insert

    if mod(Nt_orig, 2) == 0
        F_up = [F(1:halfN, :); ...
                F(halfN+1, :) / 2; ...
                zeros(nZeros - 1, Nrad, 'single'); ...
                F(halfN+1, :) / 2; ...
                F(halfN+2:end, :)];
    else
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
radarData = [radarData; zeros(1,Nrad,'single')];
Nt_pad = Nt + 1;

% Now compute idxOff using the (possibly upsampled) fs:
%   idx = tau * fs + idxOff  maps physical time tau to a 1-based index
idxOff = single(1 - t0 * fs);

%% ======================== DEM + VOXEL GRID ===============================
[xD, yD, zD] = extractDEM(S.DEM);

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
Xg = Xg(:); Yg = Yg(:); Zg = Zg(:);

% Compute ground surface elevation at each voxel's (x,y) position
zS1 = demCPU(dem, Xg, Yg);

% A voxel is valid only if it lies below the ground surface
valid = Zg < zS1;
vidx = find(valid);
Nv = numel(vidx);

% Extract only the valid (subsurface) voxel coordinates
Xv = Xg(vidx); Yv = Yg(vidx); Zv = Zg(vidx);

% The second interface (DEM2) is at constant depth 'depth1' below DEM1
zS1v = zS1(vidx);
zS2v = zS1v - depth1;

% Classify each valid voxel: isL3 = true means the voxel is in layer 3
isL3 = Zv < zS2v;

fprintf('Grid: %dx%dx%d, Valid: %d voxels (L2=%d, L3=%d)\n', ...
    Nx, Ny, Nz, Nv, sum(~isL3), sum(isL3));

%% ======================== GPU SETUP ======================================
demG.Zv = gpuArray(dem.Zv);
demG.Nx = dem.Nx; demG.Ny = dem.Ny;
demG.x0 = dem.x0; demG.y0 = dem.y0;
demG.xMax = dem.xMax; demG.yMax = dem.yMax;
demG.dx = dem.dx; demG.dy = dem.dy;
demG.invDx = dem.invDx; demG.invDy = dem.invDy;

% --------------------- DEM-IVS candidate grid ----------------------------
% dxOff, dyOff have shape [1 x 1 x Nc] where Nc = NgridXY^2 = 9.
[gx, gy] = meshgrid(linspace(-1, 1, NgridXY));
dxOff = gpuArray(reshape(single(gx(:)), 1, 1, []));
dyOff = gpuArray(reshape(single(gy(:)), 1, 1, []));
Nc = numel(dxOff);

% Spatial bounds of the DEM domain
xMin = demG.x0;
xMax_ = demG.xMax;
yMin = demG.y0;
yMax_ = demG.yMax;

%% ======================== BATCH SIZE ESTIMATION ==========================
availMem = 0.50 * g.AvailableMemory; % Use ~50% of free VRAM as safety margin
bytesPerArray = 4;                   % single precision = 4 bytes per element
numArrays = 14;                      % Approximate number of large intermediate arrays

maxElements = availMem / (bytesPerArray * numArrays);
maxNvB = floor(maxElements / Nc);    % Account for DEM-IVS expansion (Nc candidates per point)

% Radar batch size (number of radar positions per inner loop iteration).
radarBatchSize = 512;

% Voxel chunk size: how many voxels are processed per radar batch.
voxelChunkSize = floor(maxNvB / radarBatchSize);
voxelChunkSize = max(1000, min(8000, voxelChunkSize));

% Total number of batch iterations (outer voxel chunks x inner radar batches)
nVoxelChunks = ceil(Nv / voxelChunkSize);
nRadarBatches = ceil(Nrad / radarBatchSize);
totalBatches = nVoxelChunks * nRadarBatches;

fprintf('Batching: %d x %d = %d batches (vChunk=%d, rBatch=%d)\n', ...
    nVoxelChunks, nRadarBatches, totalBatches, voxelChunkSize, radarBatchSize);

%% ======================== DATA TRANSFER TO GPU ===========================
radarData_g = gpuArray(radarData);
radarPos_g  = gpuArray(radarPos);

%% ======================== CONSTANTS =====================================
EPS = single(1e-9); % Small epsilon to prevent division by zero

%% ======================== MAIN PROCESSING LOOP ===========================
fprintf('\n--- Processing ---\n');
tStart = tic;

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

    % Pre-compute single-precision masks (avoid repeated conversion in hot loops)
    isL3_s = single(isL3c);   % [nv x 1] broadcasts against [nv x B] and [nv x B x Nc]
    isL2_s = single(~isL3c);  % [nv x 1] broadcasts against [nv x B] and [nv x B x Nc]
    anyL3  = any(isL3c);      % Scalar flag to skip L3 code paths when all voxels are L2

    % Accumulator for this chunk (complex because coherent BP sums phase and amplitude)
    chunkAccum = complex(zeros(nv, 1, 'single', 'gpuArray'));

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
        % Vectorized 3D [nv x B x Nc] evaluation of Nc candidates per
        % (voxel, radar) pair. Broadcasting eliminates reshapes: x2 [nv x B]
        % broadcasts against xc [nv x B x Nc], isL3_s [nv x 1] broadcasts
        % against [nv x B x Nc].
        % ==============================================================

        % Initialize P1 and P2 at the voxel position, replicated for all B radar positions
        x1 = repmat(Xc, 1, B);
        y1 = repmat(Yc, 1, B);
        x2 = x1;
        y2 = y1;

        % Initial search radius: proportional to horizontal distance
        Lh  = sqrt((Xc - rx).^2 + (Yc - ry).^2) + EPS;       % [nv x B]
        rad = min(radMax, max(radMin, radFac * Lh));            % [nv x B]

        % bestCost: lowest optical path cost found so far per (voxel, radar) pair
        bestCost = gpuArray(inf(nv, B, 'single'));

        % bX1/bY1/bX2/bY2: store the best P1 and P2 coordinates found so far
        bX1 = x1; bY1 = y1;
        bX2 = x2; bY2 = y2;

        % Broadcasting-based index vectors replace ndgrid(1:nv, 1:B)
        % These combine to produce linear indices into [nv x B x Nc] arrays:
        %   linIdx = ii_col + jj_base + (bestIdx - 1) * nvB
        ii_col = gpuArray((1:nv)');        % [nv x 1]
        jj_base = gpuArray((0:B-1) * nv); % [1 x B], pre-multiplied by nv
        nvB = nv * B;

        % Total iterations = 2 * nCPIVSIter because we alternate P1/P2
        for iter = 1:(2*nCPIVSIter)
            doP1 = (mod(iter, 2) == 1); % Odd iterations optimize P1; even optimize P2

            if doP1
                % ===================== Optimize P1 =====================
                % Generate Nc candidate positions around current P1:
                %   xc = x1 + rad * dxOff  [nv x B x Nc]
                xc = x1 + rad .* dxOff;
                yc = y1 + rad .* dyOff;

                % Clamp candidate coordinates to DEM domain
                xc = min(max(xc, xMin), xMax_);
                yc = min(max(yc, yMin), yMax_);

                % Evaluate DEM elevation at each candidate P1
                zc = demGPU(demG, xc, yc);

                % Distance from radar to candidate P1 (air segment): [nv x B x Nc]
                Ra = sqrt((xc - rx).^2 + (yc - ry).^2 + (zc - rz).^2);

                % L2 cost: air(R->P1) + soil1(P1->V)
                Rs1_l2  = sqrt((Xc - xc).^2 + (Yc - yc).^2 + (Zc - zc).^2);
                cost_l2 = n_air * Ra + n2 * Rs1_l2;

                if anyL3
                    % L3 cost: air(R->P1) + soil1(P1->P2) + soil2(P2->V)
                    z2_cur    = demGPU(demG, x2, y2) - depth1;          % [nv x B]
                    Rs2_fixed = sqrt((Xc - x2).^2 + (Yc - y2).^2 + (Zc - z2_cur).^2); % [nv x B]

                    % Broadcasting: x2 [nv x B] auto-expands to [nv x B x Nc]
                    Rs1_l3  = sqrt((x2 - xc).^2 + (y2 - yc).^2 + (z2_cur - zc).^2);
                    cost_l3 = n_air * Ra + n2 * Rs1_l3 + n3 * Rs2_fixed;

                    % Select cost based on layer (isL2_s/isL3_s [nv x 1] broadcast to [nv x B x Nc])
                    cost = cost_l2 .* isL2_s + cost_l3 .* isL3_s;
                else
                    cost = cost_l2;
                end

                % Enforce geometric validity: voxel must be below the surface at P1
                cost(Zc >= zc) = inf;

                % For each (voxel, radar) pair, pick the candidate with minimum cost
                [minCost, bestIdx] = min(cost, [], 3);

                % Update the best solution only where improvement was found
                improved = minCost < bestCost;
                bestCost(improved) = minCost(improved);

                % Linear index into [nv x B x Nc] using broadcasting vectors
                linIdx = ii_col + jj_base + (bestIdx - 1) * nvB;
                bX1(improved) = xc(linIdx(improved));
                bY1(improved) = yc(linIdx(improved));

                % Advance the current P1 estimate to the best found so far
                x1 = bX1; y1 = bY1;

            else
                % ===================== Optimize P2 =====================
                % Only relevant for voxels in layer 3 (below DEM2)
                if anyL3
                    % Generate candidates around current P2 on DEM2 interface
                    xc = x2 + rad .* dxOff;
                    yc = y2 + rad .* dyOff;
                    xc = min(max(xc, xMin), xMax_);
                    yc = min(max(yc, yMin), yMax_);

                    % P2 elevation = DEM1 minus depth1
                    zc = demGPU(demG, xc, yc) - depth1;

                    % Current P1 elevation on DEM1 (held fixed during P2 optimization)
                    z1_cur = demGPU(demG, x1, y1);

                    % Air segment distance: R -> P1 (fixed)
                    Ra_fixed = sqrt((x1 - rx).^2 + (y1 - ry).^2 + (z1_cur - rz).^2);

                    % Broadcasting: x1 [nv x B] auto-expands to [nv x B x Nc]
                    Rs1 = sqrt((xc - x1).^2 + (yc - y1).^2 + (zc - z1_cur).^2);

                    % Distance from candidate P2 to voxel V
                    Rs2 = sqrt((Xc - xc).^2 + (Yc - yc).^2 + (Zc - zc).^2);

                    % Total L3 cost
                    cost = n_air * Ra_fixed + n2 * Rs1 + n3 * Rs2;

                    % Enforce geometric validity
                    validP2 = (Zc < zc) & (zc < z1_cur) & isL3c;
                    cost(~validP2) = inf;

                    % Select best candidate per (voxel, radar) pair
                    [minCost, bestIdx] = min(cost, [], 3);
                    improved = minCost < bestCost;
                    bestCost(improved) = minCost(improved);

                    linIdx = ii_col + jj_base + (bestIdx - 1) * nvB;
                    bX2(improved) = xc(linIdx(improved));
                    bY2(improved) = yc(linIdx(improved));

                    x2 = bX2; y2 = bY2;
                end
            end

            % Shrink the search radius for the next iteration
            rad = rad * shrink;
        end

        % After DEM-IVS: use the best refraction points found
        x1 = bX1; y1 = bY1;
        x2 = bX2; y2 = bY2;

        % ==============================================================
        % PHASE 2: NEWTON-RAPHSON LOCAL REFINEMENT
        % ==============================================================
        % Uses implicit broadcasting instead of repmat for Vx/Vy/Vz/
        % Rx/Ry/Rz/isL3_e, eliminating 7 large [nv x B] arrays:
        %   Xc/Yc/Zc [nv x 1] broadcast against [nv x B]
        %   rx/ry/rz  [1 x B]  broadcast against [nv x B]
        %   isL3_s/isL2_s [nv x 1] broadcast against [nv x B]
        % ==============================================================

        for iter = 1:nNewtonIter
            optimizeP1 = (mod(iter, 2) == 1); % Alternate: odd=P1, even=P2

            if optimizeP1
                % ===================== Newton for P1 ===================
                [z1, hx1, hy1] = demGPU_with_grad(demG, x1, y1);
                z2 = demGPU(demG, x2, y2) - depth1;

                % Vector from radar R to P1 (rx [1 x B] broadcasts to [nv x B])
                dx_r = x1 - rx; dy_r = y1 - ry; dz_r = z1 - rz;
                r1 = sqrt(dx_r.^2 + dy_r.^2 + dz_r.^2) + EPS;

                % Vector from P1 to target (V for L2, P2 for L3)
                % Xc/Yc/Zc [nv x 1] broadcast to [nv x B]; isL2_s/isL3_s [nv x 1] broadcast
                dx_2 = (x1 - Xc) .* isL2_s + (x1 - x2) .* isL3_s;
                dy_2 = (y1 - Yc) .* isL2_s + (y1 - y2) .* isL3_s;
                dz_2 = (z1 - Zc) .* isL2_s + (z1 - z2) .* isL3_s;
                r2 = sqrt(dx_2.^2 + dy_2.^2 + dz_2.^2) + EPS;

                % Snell's law residuals accounting for tilted DEM surface
                A1 = dx_r + dz_r .* hx1;
                B1 = dy_r + dz_r .* hy1;
                A2 = dx_2 + dz_2 .* hx1;
                B2 = dy_2 + dz_2 .* hy1;

                F1 = n_air * A1 ./ r1 + n2 * A2 ./ r2;
                F2 = n_air * B1 ./ r1 + n2 * B2 ./ r2;

                % Jacobian matrix
                hx2 = hx1.^2; hy2 = hy1.^2; hxy = hx1 .* hy1;
                r1_sq = r1.^2; r2_sq = r2.^2;

                J11 = n_air*(1+hx2-A1.^2./r1_sq)./r1 + n2*(1+hx2-A2.^2./r2_sq)./r2;
                J22 = n_air*(1+hy2-B1.^2./r1_sq)./r1 + n2*(1+hy2-B2.^2./r2_sq)./r2;
                J12 = n_air*(hxy-A1.*B1./r1_sq)./r1 + n2*(hxy-A2.*B2./r2_sq)./r2;

                % Solve 2x2 system via Cramer's rule
                det_J  = J11.*J22 - J12.^2 + EPS;
                delta_x = (J12.*F2 - J22.*F1) ./ det_J;
                delta_y = (J12.*F1 - J11.*F2) ./ det_J;

                % Clamp step magnitude to prevent divergence
                step_mag = sqrt(delta_x.^2 + delta_y.^2);
                max_step = single(0.2);
                scale = min(single(1), max_step ./ (step_mag + EPS));

                x1 = max(xMin, min(xMax_, x1 + scale .* delta_x));
                y1 = max(yMin, min(yMax_, y1 + scale .* delta_y));

            else
                % ===================== Newton for P2 ===================
                % Skip entirely when no L3 voxels exist (saves 2 full iterations)
                if anyL3
                    z1 = demGPU(demG, x1, y1);
                    [z2_s1, hx2, hy2] = demGPU_with_grad(demG, x2, y2);
                    z2 = z2_s1 - depth1;

                    % P1 -> P2 vector and distance
                    dx_12 = x2 - x1; dy_12 = y2 - y1; dz_12 = z2 - z1;
                    r12 = sqrt(dx_12.^2 + dy_12.^2 + dz_12.^2) + EPS;

                    % P2 -> V vector and distance (Xc/Yc/Zc [nv x 1] broadcast)
                    dx_2v = x2 - Xc; dy_2v = y2 - Yc; dz_2v = z2 - Zc;
                    r2v = sqrt(dx_2v.^2 + dy_2v.^2 + dz_2v.^2) + EPS;

                    % Snell's law at the n2->n3 interface
                    A1 = dx_12 + dz_12 .* hx2;
                    B1 = dy_12 + dz_12 .* hy2;
                    A2 = dx_2v + dz_2v .* hx2;
                    B2 = dy_2v + dz_2v .* hy2;

                    F1 = (n2 * A1 ./ r12 + n3 * A2 ./ r2v) .* isL3_s;
                    F2 = (n2 * B1 ./ r12 + n3 * B2 ./ r2v) .* isL3_s;

                    % Jacobian at the DEM2 interface (isL2_s [nv x 1] broadcasts)
                    hx2_sq = hx2.^2; hy2_sq = hy2.^2; hxy2 = hx2 .* hy2;
                    r12_sq = r12.^2; r2v_sq = r2v.^2;

                    J11 = n2*(1+hx2_sq-A1.^2./r12_sq)./r12 + n3*(1+hx2_sq-A2.^2./r2v_sq)./r2v + isL2_s;
                    J22 = n2*(1+hy2_sq-B1.^2./r12_sq)./r12 + n3*(1+hy2_sq-B2.^2./r2v_sq)./r2v + isL2_s;
                    J12 = n2*(hxy2-A1.*B1./r12_sq)./r12 + n3*(hxy2-A2.*B2./r2v_sq)./r2v;

                    det_J  = J11.*J22 - J12.^2 + EPS;
                    delta_x = (J12.*F2 - J22.*F1) ./ det_J;
                    delta_y = (J12.*F1 - J11.*F2) ./ det_J;

                    step_mag = sqrt(delta_x.^2 + delta_y.^2);
                    max_step = single(0.2);
                    scale = min(single(1), max_step ./ (step_mag + EPS));

                    x2 = max(xMin, min(xMax_, x2 + scale .* delta_x));
                    y2 = max(yMin, min(yMax_, y2 + scale .* delta_y));
                end
            end
        end

        % ==============================================================
        % FINAL OPTICAL PATH LENGTH COMPUTATION
        % ==============================================================
        z1 = demGPU(demG, x1, y1);
        z2 = demGPU(demG, x2, y2) - depth1;

        % Geometric segment distances (rx [1xB], Xc [nvx1] broadcast to [nv x B])
        Ra     = sqrt((x1 - rx).^2 + (y1 - ry).^2 + (z1 - rz).^2);
        Rs1_l2 = sqrt((Xc - x1).^2 + (Yc - y1).^2 + (Zc - z1).^2);
        Rs1_l3 = sqrt((x2 - x1).^2 + (y2 - y1).^2 + (z2 - z1).^2);
        Rs2_l3 = sqrt((Xc - x2).^2 + (Yc - y2).^2 + (Zc - z2).^2);

        % Total optical path length (isL2_s/isL3_s [nv x 1] broadcast)
        pathLen = (n_air*Ra + n2*Rs1_l2) .* isL2_s + ...
                  (n_air*Ra + n2*Rs1_l3 + n3*Rs2_l3) .* isL3_s;

        % ==============================================================
        % SIGNAL INTERPOLATION (BACKPROJECTION CORE)
        % ==============================================================
        tau  = 2 * pathLen / c;
        idxF = tau * fs + idxOff;

        validI = isfinite(idxF) & (idxF >= 1) & (idxF <= Nt);

        iFl = int32(max(1, min(Nt, floor(idxF))));
        iCe = int32(min(Nt_pad, iFl + 1));
        wCe = idxF - single(iFl);

        base = int32(0:B-1) * int32(Nt_pad);
        val = traces(iFl + base) .* (1 - wCe) + traces(iCe + base) .* wCe;

        if applyPhase
            val = val .* exp(1j * omega_c * tau);
        end

        val(~validI) = 0;

        % Coherent summation: accumulate contributions from this radar batch
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

    % Transfer chunk result to CPU and normalize by total radar positions
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
Img = complex(zeros(Nx, Ny, Nz, 'single'));
Img(vidx) = ImgAccum;

% Quick diagnostic metrics
M = abs(ImgAccum); M = M(M > 0);
if ~isempty(M)
    Mn = M / sum(M);
    fprintf('Entropy: %.3f, Sharpness: %.1f\n', -sum(Mn.*log(Mn+eps)), max(M)/mean(M));
end

% Locate the peak in the 3D volume
[~, mi] = max(abs(Img(:)));
[ix, iy, iz] = ind2sub([Nx, Ny, Nz], mi);
fprintf('Peak: (%.3f, %.3f, %.3f) m\n', xGrid(ix), yGrid(iy), zGrid(iz));

% Build and save both DEM surfaces
xDEM = xD;
yDEM = yD;
zDEM1 = zD;
zDEM2 = zD - double(depth1);

% Save all results
matDir = fullfile(outputDir, 'mat');
if ~exist(matDir, 'dir'), mkdir(matDir); end
matFile = fullfile(matDir, 'bp_hybrid_results.mat');
save(matFile, 'Img', 'xGrid', 'yGrid', 'zGrid', 'n2', 'n3', 'depth1', 'tTotal', ...
    'xDEM', 'yDEM', 'zDEM1', 'zDEM2', 'oversampFactor', '-v7.3');
fprintf('Saved: %s\n', matFile);

% Generate a quick visualization
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
tx = (xq - d.x0) * d.invDx;
ty = (yq - d.y0) * d.invDy;
j = max(int32(0), min(d.Nx-2, int32(floor(tx))));
i = max(int32(0), min(d.Ny-2, int32(floor(ty))));
ax = tx - single(j);
ay = ty - single(i);
j = j + 1; i = i + 1;
Ny = d.Ny;
idx = i + (j-1)*Ny;

z00 = d.Zv(idx);
z10 = d.Zv(idx + Ny);
z01 = d.Zv(idx + 1);
z11 = d.Zv(idx + Ny + 1);

z = (1-ax).*(1-ay).*z00 + ax.*(1-ay).*z10 + (1-ax).*ay.*z01 + ax.*ay.*z11;
hx = ((z10 - z00).*(1-ay) + (z11 - z01).*ay) * d.invDx;
hy = ((z01 - z00).*(1-ax) + (z11 - z10).*ax) * d.invDy;
end

function z = demGPU(d, xq, yq)
% DEMGPU  Bilinear DEM interpolation on GPU (elevation only).
%
% Handles inputs of any dimensionality ([nv x B], [nv x B x Nc], etc.)
% via element-wise operations and linear indexing into the 1D elevation
% vector d.Zv.
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

function z = demCPU(d, xq, yq)
% DEMCPU  Bilinear DEM interpolation on CPU using double precision.
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
r = load(f);
if isfield(r,'S'), S = r.S;
    while isstruct(S) && isfield(S,'S'), S = S.S; end
else, S = r;
end
end

function [x, y, Z] = extractDEM(D)
% EXTRACTDEM  Extract DEM data and ensure consistent orientation.
if isvector(D.x) && isvector(D.y)
    x = double(D.x(:)).'; y = double(D.y(:)); Z = double(D.z);
    if size(Z,1)==numel(x) && size(Z,2)==numel(y), Z = Z.'; end
else
    x = double(D.x(1,:)); y = double(D.y(:,1)); Z = double(D.z);
    if ~isequal(size(Z), [numel(y), numel(x)]), Z = Z.'; end
end
if any(diff(x)<0), x = fliplr(x); Z = Z(:, end:-1:1); end
if any(diff(y)<0), y = flipud(y); Z = Z(end:-1:1, :); end
end

function visualize(Img, xG, yG, zG, dyn, n2, n3, d1)
% VISUALIZE  Quick 3-slice visualization of the backprojection result.
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
