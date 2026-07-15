// app.js
// Interactive logic, real-time simulation, and browser-side CV image processing

// Simulation State
let isPlaying = false;
let mode = 'sim'; // 'sim' or 'upload'
let animationId = null;

// Canvas details
const canvas = document.getElementById('fieldCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const width = canvas.width;
const height = canvas.height;

// Parameters & Settings
let numNozzles = 5;
let speedKmh = 6.0;
let latencyMs = 80;
let weedDensity = 18;
let colorCutoff = 60;

// Simulation Assets / Coordinates
let scrollY = 0;
let plants = [];
const rowXS = [];
let scanY = 100; // Sensor/camera scanning line Y-position
let nozzleBoomY = 160; // Spray nozzles boom Y-position
let activeSprayCones = []; // Track active sprays for rendering

// Stats Counter
let cropsProtected = 0;
let weedsTargeted = 0;
let totalWeedsDetected = 0;
let totalCropsDetected = 0;
let sprayTicksSelective = 0;
let sprayTicksConventional = 0;

// Uploaded Image Cache
let uploadedImage = null;
let analyzedData = null;

// Initialize the system
function init() {
    updateParameters();
    setupListeners();
    generateInitialField();
    draw();
    updateUIIndicators();
}

// Read parameters from UI controls
function updateParameters() {
    numNozzles = parseInt(document.getElementById('nozzlesSlider').value);
    speedKmh = parseFloat(document.getElementById('speedSlider').value);
    latencyMs = parseInt(document.getElementById('latencySlider').value);
    
    document.getElementById('nozzlesVal').textContent = numNozzles;
    document.getElementById('speedVal').textContent = speedKmh.toFixed(1);
    document.getElementById('latencyVal').textContent = latencyMs;
    
    if (document.getElementById('cutoffSlider')) {
        colorCutoff = parseInt(document.getElementById('cutoffSlider').value);
        document.getElementById('cutoffVal').textContent = colorCutoff;
    }
    
    // Recalculate crop row centers for 3 rows
    rowXS.length = 0;
    const margin = width / 4;
    for (let i = 0; i < 3; i++) {
        rowXS.push((i + 1) * margin);
    }
    
    updateNozzleIndicators();
}

// Setup all button and slider listeners
function setupListeners() {
    // Sliders
    document.getElementById('nozzlesSlider').addEventListener('input', () => {
        updateParameters();
        if (mode === 'upload' && uploadedImage) {
            analyzeUploadedImage();
        }
    });
    document.getElementById('speedSlider').addEventListener('input', updateParameters);
    document.getElementById('latencySlider').addEventListener('input', updateParameters);
    
    const cutoffSlider = document.getElementById('cutoffSlider');
    if (cutoffSlider) {
        cutoffSlider.addEventListener('input', () => {
            updateParameters();
            if (mode === 'upload' && uploadedImage) {
                analyzeUploadedImage();
            }
        });
    }

    // Selects
    document.getElementById('densitySelect').addEventListener('change', (e) => {
        weedDensity = parseInt(e.target.value);
        const labels = { "5": "Low", "18": "Medium", "35": "High" };
        document.getElementById('densityLabel').textContent = labels[weedDensity] || "Medium";
        resetSimulation();
    });

    // Buttons
    const playBtn = document.getElementById('playBtn');
    playBtn.addEventListener('click', () => {
        if (isPlaying) {
            pauseSimulation();
        } else {
            startSimulation();
        }
    });

    document.getElementById('resetBtn').addEventListener('click', () => {
        resetSimulation();
    });

    // Upload & Drag/Drop
    const imageUpload = document.getElementById('imageUpload');
    const dropZone = document.getElementById('dropZone');

    imageUpload.addEventListener('change', handleFileSelect);
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--color-crop)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        if (e.dataTransfer.files.length > 0) {
            imageUpload.files = e.dataTransfer.files;
            handleFileSelect({ target: imageUpload });
        }
    });

    document.getElementById('analyzeBtn').addEventListener('click', () => {
        if (uploadedImage) {
            analyzeUploadedImage();
        } else {
            // If no image is uploaded, load a preset demonstration image
            loadDemoImage();
        }
    });
}

// Setup LED elements corresponding to nozzle count
function updateNozzleIndicators() {
    const grid = document.getElementById('indicatorGrid');
    grid.innerHTML = '';
    for (let i = 0; i < numNozzles; i++) {
        const led = document.createElement('div');
        led.className = 'nozzle-indicator';
        led.id = `nozzle-led-${i}`;
        led.title = `Nozzle ${i + 1}`;
        grid.appendChild(led);
    }
}

// Switch view tabs
function switchMode(newMode) {
    mode = newMode;
    document.getElementById('tabSim').classList.toggle('active', mode === 'sim');
    document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
    document.getElementById('tabPrep').classList.toggle('active', mode === 'prep');
    
    document.getElementById('sprayerSettingsCard').style.display = (mode === 'sim' || mode === 'upload') ? 'block' : 'none';
    document.getElementById('simulationControllerCard').style.display = mode === 'sim' ? 'block' : 'none';
    document.getElementById('uploadCard').style.display = mode === 'upload' ? 'block' : 'none';
    document.getElementById('prepCard').style.display = mode === 'prep' ? 'block' : 'none';
    
    document.getElementById('simWorkspace').style.display = mode === 'prep' ? 'none' : 'flex';
    document.getElementById('prepWorkspace').style.display = mode === 'prep' ? 'flex' : 'none';
    
    document.getElementById('simTelemetryPanel').style.display = mode === 'prep' ? 'none' : 'flex';
    document.getElementById('prepTelemetryPanel').style.display = mode === 'prep' ? 'flex' : 'none';
    
    const centerTitle = document.getElementById('centerPanelTitle');
    if (mode === 'prep') {
        centerTitle.innerHTML = '<i class="fa-solid fa-database"></i> Dataset Prep Walkthrough';
    } else {
        centerTitle.innerHTML = '<i class="fa-solid fa-eye"></i> Field Visualizer';
    }

    const laser = document.getElementById('laserLine');
    if (laser) {
        laser.style.display = mode === 'sim' ? 'block' : 'none';
    }

    pauseSimulation();
    resetSimulation();
    
    if (mode === 'upload') {
        loadDemoImage();
    } else if (mode === 'prep') {
        initDatasetPrepView();
    }
}

// ----------------------------------------------------
// LIVE SIMULATOR LOGIC
// ----------------------------------------------------

// Create initial crops in rows and scattered weeds
function generateInitialField() {
    plants = [];
    cropsProtected = 0;
    weedsTargeted = 0;
    totalWeedsDetected = 0;
    totalCropsDetected = 0;
    sprayTicksSelective = 0;
    sprayTicksConventional = 0;
    activeSprayCones = new Array(numNozzles).fill(false);

    // Fill the starting canvas with some plants
    for (let y = 50; y < height; y += 70) {
        // Generate crops along the rows
        rowXS.forEach(rx => {
            plants.push(createPlant('crop', rx + randomRange(-8, 8), y + randomRange(-10, 10)));
        });
        
        // Scattered weeds
        if (Math.random() < (weedDensity / 30)) {
            plants.push(createPlant('weed', randomRange(20, width - 20), y + randomRange(-20, 20)));
        }
    }
}

// Helper to instantiate plant objects
function createPlant(type, x, y) {
    const isCrop = type === 'crop';
    const size = isCrop ? randomRange(25, 32) : randomRange(10, 18);
    const aspect = isCrop ? randomRange(0.8, 1.2) : randomRange(0.4, 1.8);
    
    // Bounding box dimensions
    const w = size * aspect;
    const h = size / aspect;

    // Drawing shapes: crops have 4-5 leaves, weeds are spiky/round clumps
    const leaves = [];
    const numLeaves = isCrop ? randomRange(4, 6) : randomRange(3, 5);
    for (let i = 0; i < numLeaves; i++) {
        leaves.push({
            angle: (i * 360 / numLeaves) + randomRange(-15, 15),
            length: isCrop ? size * 0.9 : size,
            breadth: isCrop ? size * 0.5 : size * 0.25
        });
    }

    return {
        type: type,
        x: x,
        y: y,
        w: w,
        h: h,
        leaves: leaves,
        size: size,
        detected: false,
        sprayed: false,
        detectionTime: 0,
        color: isCrop ? 'rgba(16, 185, 129, 0.9)' : 'rgba(20, 120, 140, 0.9)',
        shadowColor: isCrop ? 'rgba(25, 120, 50, 0.9)' : 'rgba(10, 70, 80, 0.9)'
    };
}

// Start simulation rendering loop
function startSimulation() {
    if (isPlaying) return;
    isPlaying = true;
    document.getElementById('playBtn').innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
    document.getElementById('playBtn').className = 'btn btn-danger';
    document.getElementById('laserLine').style.display = 'block';
    
    document.getElementById('statusText').textContent = 'Scanning...';
    document.getElementById('valveStatusBadge').querySelector('.status-dot').style.backgroundColor = 'var(--color-crop)';
    
    tick();
}

// Pause simulation
function pauseSimulation() {
    isPlaying = false;
    document.getElementById('playBtn').innerHTML = '<i class="fa-solid fa-play"></i> Start';
    document.getElementById('playBtn').className = 'btn';
    document.getElementById('laserLine').style.display = 'none';
    document.getElementById('statusText').textContent = 'System Paused';
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
}

// Reset stats and regenerate
function resetSimulation() {
    pauseSimulation();
    if (mode === 'sim') {
        generateInitialField();
        draw();
    } else {
        if (uploadedImage) {
            analyzeUploadedImage();
        } else {
            loadDemoImage();
        }
    }
    updateUIIndicators();
}

// Main physics loop (called 60 times a second)
function tick() {
    if (!isPlaying) return;

    updatePhysics();
    draw();
    updateUIIndicators();

    animationId = requestAnimationFrame(tick);
}

// Move plants down, detect crossings, actuate nozzles
function updatePhysics() {
    // Tractor speed to canvas pixel velocity:
    // Say 1 km/h = 1000m / 3600s = 0.27 m/s.
    // Let's scale: 1 km/h = 0.6 pixels per frame.
    const speedPixels = speedKmh * 0.7;
    
    // Update plant positions
    plants.forEach(p => {
        p.y += speedPixels;
        
        // 1. Camera Detection Line logic (at y = scanY)
        if (!p.detected && (p.y + p.h/2) >= scanY) {
            p.detected = true;
            p.detectionTime = Date.now();
            if (p.type === 'crop') {
                totalCropsDetected++;
                cropsProtected++; // Crops are spared by default
            } else {
                totalWeedsDetected++;
            }
        }
    });

    // Remove plants that fall off-screen
    plants = plants.filter(p => p.y < height + 40);

    // Generate new plants at the top
    scrollY += speedPixels;
    if (scrollY >= 70) {
        scrollY = 0;
        // Spawn crop row plants
        rowXS.forEach(rx => {
            plants.push(createPlant('crop', rx + randomRange(-8, 8), -30));
        });
        
        // Spawn weeds randomly
        if (Math.random() < (weedDensity / 25)) {
            plants.push(createPlant('weed', randomRange(20, width - 20), -50));
        }
    }

    // 2. Spray Boom Actuation Logic (at y = nozzleBoomY)
    // We map boom width into N slices (nozzles)
    const nozzleWidth = width / numNozzles;
    activeSprayCones = new Array(numNozzles).fill(false);
    
    // Conventional spraying is active 100% of the time for all nozzles
    sprayTicksConventional += numNozzles;

    // Selective spraying evaluates if any weed bounding box lies under the nozzle boom
    plants.forEach(p => {
        if (p.type === 'weed') {
            // Check if weed is physically passing the boom
            const plantTop = p.y - p.h/2;
            const plantBottom = p.y + p.h/2;
            
            // Speed latency adjustment: delay when the nozzle opens
            // latencyMs represents electrical/solenoid delay
            const latencyOffset = (speedPixels * (latencyMs / 16.67));
            const activeZoneStart = nozzleBoomY - 15 + latencyOffset;
            const activeZoneEnd = nozzleBoomY + p.h + 15 + latencyOffset;
            
            if (plantBottom >= activeZoneStart && plantTop <= activeZoneEnd) {
                // Find which nozzle slices overlap this weed horizontally
                const weedLeft = p.x - p.w/2;
                const weedRight = p.x + p.w/2;
                
                for (let j = 0; j < numNozzles; j++) {
                    const nozzleLeft = j * nozzleWidth;
                    const nozzleRight = (j + 1) * nozzleWidth;
                    
                    // Horizontal intersection check
                    if (weedRight >= nozzleLeft - 10 && weedLeft <= nozzleRight + 10) {
                        activeSprayCones[j] = true;
                        
                        if (!p.sprayed) {
                            p.sprayed = true;
                            weedsTargeted++;
                        }
                    }
                }
            }
        }
    });

    // Update active nozzle spray count
    activeSprayCones.forEach(active => {
        if (active) sprayTicksSelective++;
    });
}

// ----------------------------------------------------
// CANVAS DRAWING METHODS
// ----------------------------------------------------

function draw() {
    // 1. Draw soil background
    ctx.fillStyle = '#233018'; // Base soil dark olive/brown
    ctx.fillRect(0, 0, width, height);

    // Draw soil noise/specks
    ctx.fillStyle = '#1d2714';
    for (let i = 0; i < 60; i++) {
        // Use a deterministic layout if static mode
        const rx = (i * 137.5) % width;
        const ry = (i * 265.4) % height;
        ctx.fillRect(rx, ry, 6, 6);
    }
    
    // 2. Draw Plants
    plants.forEach(p => {
        ctx.save();
        ctx.translate(p.x, p.y);

        // Draw shadow/outline
        ctx.fillStyle = p.shadowColor;
        p.leaves.forEach(leaf => {
            ctx.save();
            ctx.rotate(leaf.angle * Math.PI / 180);
            drawEllipse(0, 0, leaf.length, leaf.breadth);
            ctx.restore();
        });

        // Draw main leaf bodies
        ctx.fillStyle = p.color;
        p.leaves.forEach(leaf => {
            ctx.save();
            ctx.rotate(leaf.angle * Math.PI / 180);
            drawEllipse(0, 0, leaf.length * 0.92, leaf.breadth * 0.92);
            ctx.restore();
        });

        // Draw leaf veins (for crops)
        if (p.type === 'crop') {
            ctx.strokeStyle = 'rgba(70, 210, 110, 0.8)';
            ctx.lineWidth = 1;
            p.leaves.forEach(leaf => {
                ctx.save();
                ctx.rotate(leaf.angle * Math.PI / 180);
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(leaf.length * 0.7, 0);
                ctx.stroke();
                ctx.restore();
            });
        }
        
        ctx.restore();

        // 3. Draw bounding boxes around scanned plants
        if (p.detected) {
            ctx.lineWidth = 2;
            if (p.type === 'crop') {
                ctx.strokeStyle = '#10b981'; // Green box
                ctx.strokeRect(p.x - p.w/2 - 2, p.y - p.h/2 - 2, p.w + 4, p.h + 4);
                
                ctx.fillStyle = '#10b981';
                ctx.font = '10px Inter';
                ctx.fillText('Crop', p.x - p.w/2, p.y - p.h/2 - 6);
            } else {
                ctx.strokeStyle = '#f43f5e'; // Red box
                ctx.strokeRect(p.x - p.w/2 - 2, p.y - p.h/2 - 2, p.w + 4, p.h + 4);
                
                ctx.fillStyle = '#f43f5e';
                ctx.font = '10px Inter';
                ctx.fillText(`Weed (${Math.round(p.size*p.size)}px)`, p.x - p.w/2, p.y - p.h/2 - 6);
            }
        }
    });

    // 4. Draw Nozzle Spray Cones
    const nozzleWidth = width / numNozzles;
    activeSprayCones.forEach((active, index) => {
        const nx = (index + 0.5) * nozzleWidth;
        
        if (active) {
            // Draw glowing blue spray cone
            const gradient = ctx.createLinearGradient(nx, nozzleBoomY, nx, nozzleBoomY + 80);
            gradient.addColorStop(0, 'rgba(59, 130, 246, 0.7)');
            gradient.addColorStop(0.5, 'rgba(59, 130, 246, 0.35)');
            gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.moveTo(nx, nozzleBoomY);
            ctx.lineTo(nx - nozzleWidth/2 + 5, nozzleBoomY + 80);
            ctx.lineTo(nx + nozzleWidth/2 - 5, nozzleBoomY + 80);
            ctx.closePath();
            ctx.fill();

            // Draw nozzle water droplets
            ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
            for (let i = 0; i < 8; i++) {
                const rx = nx + randomRange(-nozzleWidth/4, nozzleWidth/4);
                const ry = nozzleBoomY + randomRange(10, 75);
                ctx.beginPath();
                ctx.arc(rx, ry, randomRange(1, 3), 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    });

    // 5. Draw Hardware overlay: Camera Scanner Line & Sprayer Boom line
    // Scanning camera laser
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.6)'; // Cyan
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, scanY);
    ctx.lineTo(width, scanY);
    ctx.stroke();
    
    // Sprayer Boom Bar
    ctx.fillStyle = 'rgba(30, 41, 59, 0.9)'; // Slate bar
    ctx.fillRect(0, nozzleBoomY - 8, width, 12);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.strokeRect(0, nozzleBoomY - 8, width, 12);

    // Individual nozzles on the boom bar
    for (let i = 0; i < numNozzles; i++) {
        const nx = (i + 0.5) * nozzleWidth;
        ctx.fillStyle = activeSprayCones[i] ? '#60a5fa' : '#475569'; // Active vs idle nozzle body
        ctx.fillRect(nx - 6, nozzleBoomY - 4, 12, 8);
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(nx - 3, nozzleBoomY + 4, 6, 3);
        
        // Small spray LED indicator on canvas
        ctx.beginPath();
        ctx.arc(nx, nozzleBoomY - 10, 3, 0, 2 * Math.PI);
        ctx.fillStyle = activeSprayCones[i] ? '#3b82f6' : '#334155';
        ctx.fill();
    }
}

// Canvas helper to draw leaf shapes
function drawEllipse(cx, cy, rx, ry) {
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
    ctx.fill();
}

// Update telemetry dashboard statistics
function updateUIIndicators() {
    // 1. Update stats card
    document.getElementById('statCrops').textContent = cropsProtected;
    document.getElementById('statWeeds').textContent = totalWeedsDetected > 0 ? `${weedsTargeted}/${totalWeedsDetected}` : '0';
    
    // Mock spray pesticide savings metric
    let savingsPct = 100;
    if (sprayTicksConventional > 0) {
        savingsPct = (1.0 - (sprayTicksSelective / sprayTicksConventional)) * 100;
        if (savingsPct < 0) savingsPct = 0;
        if (savingsPct > 96) savingsPct = 96; // Cap at realistic maximum savings
    } else {
        savingsPct = 85.4; // Default starting mockup
    }

    // Liter savings estimator
    const conventionalLitersPerMin = 1.2 * numNozzles; // 1.2 L/min per nozzle
    const timeElapsedSeconds = (sprayTicksConventional / numNozzles) / 60; // Approximate frames elapsed as seconds
    const conventionalLiters = (conventionalLitersPerMin * timeElapsedSeconds) / 60;
    const selectiveLiters = (conventionalLiters * (sprayTicksSelective / sprayTicksConventional)) || 0;
    const litersSaved = Math.max(0, conventionalLiters - selectiveLiters);

    document.getElementById('statVolume').textContent = `${litersSaved.toFixed(2)} L Saved`;

    // 2. Update Gauge circle
    const savingsCircle = document.getElementById('savingsCircle');
    const savingsValue = document.getElementById('savingsValue');
    savingsCircle.style.setProperty('--savings-percent', `${savingsPct}%`);
    savingsValue.textContent = `${savingsPct.toFixed(1)}%`;

    // Nozzles count text
    const activeNozzlesCount = activeSprayCones.filter(Boolean).length;
    document.getElementById('activeNozzlesText').textContent = `${activeNozzlesCount} Nozzles Open`;
    
    // LED classes
    for (let i = 0; i < numNozzles; i++) {
        const led = document.getElementById(`nozzle-led-${i}`);
        if (led) {
            led.className = activeSprayCones[i] ? 'nozzle-indicator active' : 'nozzle-indicator';
        }
    }
}

// ----------------------------------------------------
// STATIC PHOTO CV SEGMENTATION ANALYZER
// ----------------------------------------------------

// Handle uploaded image files
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImage = new Image();
        uploadedImage.onload = function() {
            // Draw uploaded image onto canvas fitting dimensions
            renderUploadedImageToCanvas();
            analyzeUploadedImage();
        };
        uploadedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Render the user's image to canvas conserving aspect ratio
function renderUploadedImageToCanvas() {
    if (!uploadedImage) return;
    
    ctx.clearRect(0, 0, width, height);
    
    // Scale image to fit canvas
    const imgRatio = uploadedImage.width / uploadedImage.height;
    const canvasRatio = width / height;
    
    let drawWidth, drawHeight;
    if (imgRatio > canvasRatio) {
        drawWidth = width;
        drawHeight = width / imgRatio;
    } else {
        drawHeight = height;
        drawWidth = height * imgRatio;
    }
    
    const x = (width - drawWidth) / 2;
    const y = (height - drawHeight) / 2;
    
    ctx.drawImage(uploadedImage, x, y, drawWidth, drawHeight);
}

// Javascript pixel-level segmentation: Excess Green Index clustering
function analyzeUploadedImage() {
    if (!uploadedImage) return;

    // Reset counts
    cropsProtected = 0;
    weedsTargeted = 0;
    totalWeedsDetected = 0;
    totalCropsDetected = 0;
    
    // 1. Re-render base image
    renderUploadedImageToCanvas();
    
    // 2. Fetch pixel map from canvas
    const imgData = ctx.getImageData(0, 0, width, height);
    const pixels = imgData.data;
    
    // 3. Grid-based connected component scan
    // To scan rapidly, we check grid blocks (e.g. 5x5 pixels).
    const step = 5;
    const gridRows = Math.floor(height / step);
    const gridCols = Math.floor(width / step);
    const greenGrid = Array.from({ length: gridRows }, () => new Array(gridCols).fill(false));
    
    // Segments green pixels using Excess Green (ExG) formula: 2G - R - B
    // Threshold is adjusted by colorCutoff slider
    for (let r = 0; r < gridRows; r++) {
        for (let c = 0; c < gridCols; c++) {
            // Sample pixel color at grid block center
            const pxIndex = ((r * step + Math.floor(step/2)) * width + (c * step + Math.floor(step/2))) * 4;
            if (pxIndex >= pixels.length) continue;
            
            const red = pixels[pxIndex];
            const green = pixels[pxIndex + 1];
            const blue = pixels[pxIndex + 2];
            
            const exg = 2.0 * green - red - blue;
            if (exg > colorCutoff) {
                greenGrid[r][c] = true;
            }
        }
    }
    
    // Connected components search (BFS) on grid cells to cluster plants
    const visited = Array.from({ length: gridRows }, () => new Array(gridCols).fill(false));
    const plantClusters = [];
    
    for (let r = 0; r < gridRows; r++) {
        for (let c = 0; c < gridCols; c++) {
            if (greenGrid[r][c] && !visited[r][c]) {
                // Perform BFS to gather cluster points
                const cluster = [];
                const queue = [[r, c]];
                visited[r][c] = true;
                
                while (queue.length > 0) {
                    const [currR, currC] = queue.shift();
                    cluster.push({ r: currR, c: currC });
                    
                    // Check 4-way neighbors
                    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
                    for (const [dr, dc] of dirs) {
                        const nr = currR + dr;
                        const nc = currC + dc;
                        
                        if (nr >= 0 && nr < gridRows && nc >= 0 && nc < gridCols) {
                            if (greenGrid[nr][nc] && !visited[nr][nc]) {
                                visited[nr][nc] = true;
                                queue.push([nr, nc]);
                            }
                        }
                    }
                }
                
                // Keep clusters that are large enough (filter out single grid noise)
                if (cluster.length >= 4) {
                    plantClusters.push(cluster);
                }
            }
        }
    }
    
    // Process plant clusters into bounding box objects
    const detectedPlants = [];
    plantClusters.forEach(cluster => {
        let minR = gridRows, maxR = 0, minC = gridCols, maxC = 0;
        cluster.forEach(cell => {
            if (cell.r < minR) minR = cell.r;
            if (cell.r > maxR) maxR = cell.r;
            if (cell.c < minC) minC = cell.c;
            if (cell.c > maxC) maxC = cell.c;
        });
        
        // Convert back to canvas pixels
        const bbox = {
            x: minC * step,
            y: minR * step,
            w: (maxC - minC + 1) * step,
            h: (maxR - minR + 1) * step
        };
        bbox.cx = bbox.x + bbox.w / 2;
        bbox.cy = bbox.y + bbox.h / 2;
        bbox.area = bbox.w * bbox.h;
        
        detectedPlants.push(bbox);
    });
    
    // Classify: We determine crop row lines based on cluster centers
    // We group plant center X's to find vertical crop lines
    const plantCentersX = detectedPlants.map(p => p.cx);
    let cropRowCenters = [];
    
    if (plantCentersX.length > 0) {
        const sortedX = plantCentersX.sort((a, b) => a - b);
        const groups = [[sortedX[0]]];
        for (let i = 1; i < sortedX.length; i++) {
            if (sortedX[i] - groups[groups.length - 1][groups.length - 1] < 100) {
                groups[groups.length - 1].push(sortedX[i]);
            } else {
                groups.push([sortedX[i]]);
            }
        }
        cropRowCenters = groups.map(g => g.reduce((sum, val) => sum + val, 0) / g.length);
    } else {
        cropRowCenters = [width/4, width/2, 3*width/4];
    }
    
    // Nozzle layout
    const nozzleWidth = width / numNozzles;
    activeSprayCones = new Array(numNozzles).fill(false);
    
    // Standard spray counts: conventional sprays the whole image (100%)
    sprayTicksConventional = 1000; 
    let activeSprayArea = 0;
    
    // Draw grid overlay for crop rows (translucent green)
    ctx.save();
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.15)';
    ctx.lineWidth = 12;
    cropRowCenters.forEach(rx => {
        ctx.beginPath();
        ctx.moveTo(rx, 0);
        ctx.lineTo(rx, height);
        ctx.stroke();
    });
    ctx.restore();
    
    // Classify and annotate
    detectedPlants.forEach(plant => {
        // Is it aligned vertically with an expected crop row?
        const isNearRow = cropRowCenters.some(rx => Math.abs(plant.cx - rx) < 60);
        
        // Size threshold: weeds are smaller, crops are larger
        const isCrop = isNearRow && (plant.area > 500);
        
        if (isCrop) {
            totalCropsDetected++;
            cropsProtected++;
            
            // Draw Crop Green Box
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            ctx.strokeRect(plant.x, plant.y, plant.w, plant.h);
            
            ctx.fillStyle = '#10b981';
            ctx.font = '11px Outfit';
            ctx.fillText("CROP", plant.x + 3, plant.y - 4);
        } else {
            totalWeedsDetected++;
            weedsTargeted++; // Immediately targeted since static
            
            // Draw Weed Red Box
            ctx.strokeStyle = '#f43f5e';
            ctx.lineWidth = 2;
            ctx.strokeRect(plant.x, plant.y, plant.w, plant.h);
            
            ctx.fillStyle = '#f43f5e';
            ctx.font = '11px Outfit';
            ctx.fillText("WEED", plant.x + 3, plant.y - 4);
            
            // Activate specific spray nozzle zone overlay
            const nozzleIndex = Math.floor(plant.cx / nozzleWidth);
            if (nozzleIndex >= 0 && nozzleIndex < numNozzles) {
                activeSprayCones[nozzleIndex] = true;
                
                // Add spray target visualization (cyan transparent mask)
                ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 1;
                
                // Spray range includes weed y limits with safety margins
                const sprayY = Math.max(0, plant.y - 10);
                const sprayH = Math.min(height, plant.h + 20);
                
                ctx.fillRect(nozzleIndex * nozzleWidth, sprayY, nozzleWidth, sprayH);
                ctx.strokeRect(nozzleIndex * nozzleWidth, sprayY, nozzleWidth, sprayH);
                
                activeSprayArea += (nozzleWidth * sprayH);
            }
        }
    });
    
    // Draw horizontal nozzle grid guides
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let i = 1; i < numNozzles; i++) {
        ctx.beginPath();
        ctx.moveTo(i * nozzleWidth, 0);
        ctx.lineTo(i * nozzleWidth, height);
        ctx.stroke();
    }
    
    // Calculate simulated savings percentage:
    // Selective Spray Area vs Total Field Area
    const totalFieldArea = width * height;
    const savingsPct = totalFieldArea > 0 ? (1.0 - (activeSprayArea / totalFieldArea)) * 100 : 85;
    
    // Set conventional/selective ratios to match calculations for gauge updates
    sprayTicksConventional = 1000;
    sprayTicksSelective = Math.round((activeSprayArea / totalFieldArea) * 1000);
    
    updateUIIndicators();
    
    // Override savings circle display value to represent image analytics
    document.getElementById('savingsCircle').style.setProperty('--savings-percent', `${savingsPct.toFixed(1)}%`);
    document.getElementById('savingsValue').textContent = `${savingsPct.toFixed(1)}%`;
    document.getElementById('statVolume').textContent = `Selective Spraying Active`;
}

// Generate a synthetic image and analyze it if no image was uploaded
function loadDemoImage() {
    // We create a mock field layout drawing directly to canvas
    uploadedImage = null;
    
    // Clear & draw synthetic elements
    ctx.fillStyle = '#2d1f10'; // Brown soil background
    ctx.fillRect(0, 0, width, height);

    // Draw some soil lines
    ctx.fillStyle = '#221609';
    for (let i = 0; i < 40; i++) {
        ctx.fillRect(Math.random() * width, Math.random() * height, Math.random() * 20 + 5, 8);
    }
    
    // Draw 3 crop rows with green leaves
    const margin = width / 4;
    const cropXS = [margin, margin*2, margin*3];
    ctx.fillStyle = 'rgb(40, 185, 80)';
    
    // Crops
    cropXS.forEach(cx => {
        for (let cy = 80; cy < height; cy += 120) {
            // Draw a plant shape
            ctx.beginPath();
            ctx.arc(cx, cy, 25, 0, 2 * Math.PI);
            ctx.arc(cx - 15, cy - 10, 18, 0, 2 * Math.PI);
            ctx.arc(cx + 15, cy + 10, 20, 0, 2 * Math.PI);
            ctx.arc(cx - 10, cy + 15, 17, 0, 2 * Math.PI);
            ctx.fill();
        }
    });

    // Draw 6-8 scattered small weeds (light/yellow green circles)
    ctx.fillStyle = 'rgb(120, 160, 40)';
    const weedPositions = [
        {x: 100, y: 180},
        {x: 150, y: 350},
        {x: 350, y: 120},
        {x: 500, y: 280},
        {x: 520, y: 480},
        {x: 700, y: 220},
        {x: 310, y: 440}
    ];

    weedPositions.forEach(wp => {
        ctx.beginPath();
        ctx.arc(wp.x, wp.y, 8, 0, 2 * Math.PI);
        ctx.arc(wp.x - 5, wp.y + 4, 6, 0, 2 * Math.PI);
        ctx.arc(wp.x + 4, wp.y - 3, 7, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // Create an Image object from this canvas state so it caches for processing
    uploadedImage = new Image();
    uploadedImage.onload = function() {
        analyzeUploadedImage();
    };
    uploadedImage.src = canvas.toDataURL();
}

// ----------------------------------------------------
// UTILITY FUNCTIONS
// ----------------------------------------------------

function randomRange(min, max) {
    return Math.random() * (max - min) + min;
}

// ----------------------------------------------------
// DATA PREPARATION WIZARD LOGIC
// ----------------------------------------------------

let prepStep = 1;
let annotationClass = 0; // 0 for Crop, 1 for Weed
let annotations = [];
let annotatorCanvas = null;
let annotatorCtx = null;
let isDrawingAnnotation = false;
let startX, startY;
let currentX, currentY;
let augCanvas = null;
let augCtx = null;

// Initialize Dataset Prep View
function initDatasetPrepView() {
    prepStep = 1;
    switchPrepStep(1);
    
    // Set up step 3 resize controls
    const sizeSlider = document.getElementById('prepSizeSlider');
    if (sizeSlider) {
        sizeSlider.addEventListener('input', updateResizeSimulation);
        updateResizeSimulation();
    }
    
    // Set up step 4 augmentation controls
    const rotSlider = document.getElementById('augRotSlider');
    const zoomSlider = document.getElementById('augZoomSlider');
    const shiftSlider = document.getElementById('augShiftXSlider');
    const flipCheck = document.getElementById('augFlipH');
    
    if (rotSlider) rotSlider.addEventListener('input', updateAugmentationPreview);
    if (zoomSlider) zoomSlider.addEventListener('input', updateAugmentationPreview);
    if (shiftSlider) shiftSlider.addEventListener('input', updateAugmentationPreview);
    if (flipCheck) flipCheck.addEventListener('change', updateAugmentationPreview);
    
    const regenBtn = document.getElementById('regenAugBtn');
    if (regenBtn) {
        // Simple helper to force redraw
        regenBtn.onclick = function() {
            // Apply randomized variations relative to slider baselines
            const rotVal = Math.round(randomRange(-rotSlider.value, rotSlider.value));
            const zoomVal = randomRange(1.0, parseFloat(zoomSlider.value));
            const shiftVal = Math.round(randomRange(-shiftSlider.value, shiftSlider.value));
            const flipVal = flipCheck.checked ? Math.random() < 0.5 : false;
            
            drawAugmentationFrame(rotVal, zoomVal, shiftVal, flipVal);
        };
    }

    // Set up step 5 annotator canvas
    annotatorCanvas = document.getElementById('annotatorCanvas');
    if (annotatorCanvas) {
        annotatorCtx = annotatorCanvas.getContext('2d');
        setupAnnotationCanvas();
        drawAnnotationScene();
    }
    
    // Step 5 buttons
    const clearBtn = document.getElementById('clearLabelsBtn');
    if (clearBtn) clearBtn.onclick = clearAnnotations;
    
    const autoBtn = document.getElementById('autoLabelBtn');
    if (autoBtn) autoBtn.onclick = autoLabelAnnotations;
}

// Switch between prep steps (1, 3, 4, 5)
function switchPrepStep(step) {
    prepStep = step;
    
    // Update active state of step buttons
    document.querySelectorAll('.prep-step-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`prepStep${step}`);
    if (activeBtn) activeBtn.classList.add('active');
    
    // Hide all step view containers, show current
    document.getElementById('prepStepView1').style.display = step === 1 ? 'flex' : 'none';
    document.getElementById('prepStepView3').style.display = step === 3 ? 'flex' : 'none';
    document.getElementById('prepStepView4').style.display = step === 4 ? 'flex' : 'none';
    document.getElementById('prepStepView5').style.display = step === 5 ? 'flex' : 'none';
    
    // Trigger step specific redraws
    if (step === 4) {
        setTimeout(initAugmentationCanvas, 50); // Small timeout to ensure DOM layout is ready
    } else if (step === 5) {
        setTimeout(() => {
            setupAnnotationCanvas();
            drawAnnotationScene();
        }, 50);
    }
}

// Step 3: Resize Simulation
function updateResizeSimulation() {
    const size = parseInt(document.getElementById('prepSizeSlider').value);
    document.getElementById('targetSizeLabel').textContent = `${size} x ${size}`;
    document.getElementById('resizedResVal').textContent = `${size} x ${size} PX`;
    document.getElementById('rightPanelRes').textContent = `${size} x ${size}`;
    
    // Resize preview container scale styling
    const previewField = document.getElementById('resizedPreviewField');
    if (previewField) {
        // Adjust visual display scaling
        previewField.style.transform = `scale(${320 / size})`;
    }
    
    // Calculate estimated files stats
    const bytes = size * size * 3;
    const kb = (bytes / 1024).toFixed(0);
    document.getElementById('calcFileSize').textContent = `${kb} KB`;
    
    let trainTime = "1h 15m";
    if (size === 128) trainTime = "0h 10m";
    else if (size === 256) trainTime = "0h 25m";
    else if (size === 384) trainTime = "0h 48m";
    else if (size === 512) trainTime = "1h 15m";
    else if (size === 640) trainTime = "2h 05m";
    else if (size === 768) trainTime = "3h 10m";
    else if (size === 896) trainTime = "4h 30m";
    else if (size === 1024) trainTime = "6h 15m";
    
    document.getElementById('calcTrainTime').textContent = trainTime;
}

// Step 4: Data Augmentation Canvas Setup
function initAugmentationCanvas() {
    augCanvas = document.getElementById('augCanvas');
    if (!augCanvas) return;
    augCtx = augCanvas.getContext('2d');
    updateAugmentationPreview();
}

// Read sliders and update preview
function updateAugmentationPreview() {
    if (!augCanvas || !augCtx) return;
    
    const rot = parseInt(document.getElementById('augRotSlider').value);
    const zoom = parseFloat(document.getElementById('augZoomSlider').value);
    const shiftX = parseInt(document.getElementById('augShiftXSlider').value);
    const flipH = document.getElementById('augFlipH').checked;
    
    // Update labels
    document.getElementById('augRotLabel').textContent = `${rot}°`;
    document.getElementById('augZoomLabel').textContent = `${zoom.toFixed(2)}x`;
    document.getElementById('augShiftXLabel').textContent = `${shiftX}%`;
    
    drawAugmentationFrame(rot, zoom, shiftX, flipH);
}

// Draw frame helper
function drawAugmentationFrame(rot, zoom, shiftX, flipH) {
    if (!augCanvas || !augCtx) return;
    
    // Clear canvas
    augCtx.fillStyle = '#233018'; // Soil background
    augCtx.fillRect(0, 0, augCanvas.width, augCanvas.height);
    
    // Draw background texture specks
    augCtx.fillStyle = '#1d2714';
    for (let i = 0; i < 20; i++) {
        const rx = (i * 37) % augCanvas.width;
        const ry = (i * 73) % augCanvas.height;
        augCtx.fillRect(rx, ry, 4, 4);
    }
    
    // Apply transformations
    augCtx.save();
    augCtx.translate(augCanvas.width / 2, augCanvas.height / 2);
    
    // Apply shift
    const pxShiftX = (shiftX / 100) * augCanvas.width;
    augCtx.translate(pxShiftX, 0);
    
    // Apply zoom
    augCtx.scale(zoom, zoom);
    
    // Apply rotation
    augCtx.rotate(rot * Math.PI / 180);
    
    // Apply flip
    if (flipH) {
        augCtx.scale(-1, 1);
    }
    
    // Draw shadow
    augCtx.fillStyle = 'rgba(25, 120, 50, 0.7)';
    const size = 50;
    const numLeaves = 5;
    for (let i = 0; i < numLeaves; i++) {
        augCtx.save();
        augCtx.rotate(i * 2 * Math.PI / numLeaves);
        augCtx.beginPath();
        augCtx.ellipse(0, 0, size, size * 0.55, 0, 0, 2 * Math.PI);
        augCtx.fill();
        augCtx.restore();
    }
    
    // Leaves body
    augCtx.fillStyle = '#10b981';
    for (let i = 0; i < numLeaves; i++) {
        augCtx.save();
        augCtx.rotate(i * 2 * Math.PI / numLeaves);
        augCtx.beginPath();
        augCtx.ellipse(0, 0, size * 0.92, size * 0.5, 0, 0, 2 * Math.PI);
        augCtx.fill();
        
        // Veins
        augCtx.strokeStyle = 'rgba(70, 210, 110, 0.8)';
        augCtx.lineWidth = 1;
        augCtx.beginPath();
        augCtx.moveTo(0, 0);
        augCtx.lineTo(size * 0.7, 0);
        augCtx.stroke();
        augCtx.restore();
    }
    
    augCtx.restore();
    
    // Update Python/Keras code snippet and right panel stats
    const rotVal = rot;
    const shiftVal = (shiftX / 100).toFixed(2);
    const zoomVal = (zoom - 1.0).toFixed(2);
    
    const code = `from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=${rotVal},
    width_shift_range=${shiftVal},
    height_shift_range=${shiftVal},
    zoom_range=${zoomVal},
    horizontal_flip=${flipH ? 'True' : 'False'},
    fill_mode='nearest'
)`;
    
    document.getElementById('kerasCodeSnippet').textContent = code;
    
    // Calculate total training samples
    const factor = (1.5 + (Math.abs(rot) / 100) + (zoom - 1.0) + (Math.abs(shiftX) / 15)).toFixed(1);
    const totalSamples = Math.round(546 * parseFloat(factor));
    
    document.getElementById('rightPanelAugFactor').textContent = `${factor}x`;
    document.getElementById('rightPanelTotalSamples').textContent = totalSamples.toLocaleString();
}

// Step 5: Bounding Box Manual Annotation Canvas
function setupAnnotationCanvas() {
    if (!annotatorCanvas) return;
    
    // Clean event listeners to prevent duplicate triggers
    annotatorCanvas.onmousedown = onMouseDownAnnotation;
    annotatorCanvas.onmousemove = onMouseMoveAnnotation;
    annotatorCanvas.onmouseup = onMouseUpAnnotation;
}

// Choose Crop or Weed label class
function selectAnnotationClass(classId) {
    annotationClass = classId;
    document.getElementById('classCropBtn').className = classId === 0 ? 'annotation-class active-crop' : 'annotation-class';
    document.getElementById('classWeedBtn').className = classId === 1 ? 'annotation-class active-weed' : 'annotation-class';
}

// Draw static field image on labeling canvas
function drawAnnotationScene() {
    if (!annotatorCanvas || !annotatorCtx) return;
    
    // Draw soil base
    annotatorCtx.fillStyle = '#233018';
    annotatorCtx.fillRect(0, 0, annotatorCanvas.width, annotatorCanvas.height);
    
    // Soil speckles
    annotatorCtx.fillStyle = '#1d2714';
    for (let i = 0; i < 15; i++) {
        const rx = (i * 29) % annotatorCanvas.width;
        const ry = (i * 47) % annotatorCanvas.height;
        annotatorCtx.fillRect(rx, ry, 4, 4);
    }
    
    // Draw plant objects in absolute positions
    drawPlantObject(110, 190, 'crop');
    drawPlantObject(290, 100, 'crop');
    drawPlantObject(300, 230, 'weed');
    drawPlantObject(150, 80, 'weed');
    
    // Draw already drawn bounding boxes
    annotations.forEach(box => {
        const isCrop = box.classId === 0;
        annotatorCtx.strokeStyle = isCrop ? '#10b981' : '#f43f5e';
        annotatorCtx.lineWidth = 2;
        annotatorCtx.strokeRect(box.x, box.y, box.w, box.h);
        
        // Label text background
        annotatorCtx.fillStyle = isCrop ? '#10b981' : '#f43f5e';
        annotatorCtx.font = '10px Inter';
        const labelText = isCrop ? 'Crop' : 'Weed';
        const textW = annotatorCtx.measureText(labelText).width + 6;
        annotatorCtx.fillRect(box.x, box.y - 14, textW, 14);
        
        annotatorCtx.fillStyle = '#ffffff';
        annotatorCtx.fillText(labelText, box.x + 3, box.y - 3);
    });
    
    // Draw active rectangle while dragging
    if (isDrawingAnnotation) {
        const labelColor = annotationClass === 0 ? '#10b981' : '#f43f5e';
        annotatorCtx.strokeStyle = labelColor;
        annotatorCtx.lineWidth = 1.5;
        annotatorCtx.setLineDash([4, 4]);
        annotatorCtx.strokeRect(startX, startY, currentX - startX, currentY - startY);
        annotatorCtx.setLineDash([]);
    }
}

// Canvas draw helper
function drawPlantObject(cx, cy, type) {
    annotatorCtx.save();
    annotatorCtx.translate(cx, cy);
    
    if (type === 'crop') {
        annotatorCtx.fillStyle = 'rgba(25, 120, 50, 0.7)';
        for (let i = 0; i < 4; i++) {
            annotatorCtx.rotate(Math.PI / 2);
            annotatorCtx.beginPath();
            annotatorCtx.ellipse(0, 0, 20, 10, 0, 0, 2*Math.PI);
            annotatorCtx.fill();
        }
        annotatorCtx.fillStyle = '#10b981';
        for (let i = 0; i < 4; i++) {
            annotatorCtx.rotate(Math.PI / 2);
            annotatorCtx.beginPath();
            annotatorCtx.ellipse(0, 0, 18, 8, 0, 0, 2*Math.PI);
            annotatorCtx.fill();
        }
    } else {
        annotatorCtx.fillStyle = 'rgba(20, 80, 50, 0.7)';
        for (let i = 0; i < 6; i++) {
            annotatorCtx.rotate(Math.PI / 3);
            annotatorCtx.beginPath();
            annotatorCtx.ellipse(0, 0, 12, 4, 0, 0, 2*Math.PI);
            annotatorCtx.fill();
        }
        annotatorCtx.fillStyle = '#14785a';
        for (let i = 0; i < 6; i++) {
            annotatorCtx.rotate(Math.PI / 3);
            annotatorCtx.beginPath();
            annotatorCtx.ellipse(0, 0, 10, 3, 0, 0, 2*Math.PI);
            annotatorCtx.fill();
        }
    }
    annotatorCtx.restore();
}

// Mouse actions for manual drawing
function onMouseDownAnnotation(e) {
    const rect = annotatorCanvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    isDrawingAnnotation = true;
    currentX = startX;
    currentY = startY;
}

function onMouseMoveAnnotation(e) {
    if (!isDrawingAnnotation) return;
    const rect = annotatorCanvas.getBoundingClientRect();
    currentX = e.clientX - rect.left;
    currentY = e.clientY - rect.top;
    drawAnnotationScene();
}

function onMouseUpAnnotation(e) {
    if (!isDrawingAnnotation) return;
    isDrawingAnnotation = false;
    
    const rect = annotatorCanvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);
    const w = Math.abs(startX - endX);
    const h = Math.abs(startY - endY);
    
    if (w > 10 && h > 10) {
        annotations.push({
            classId: annotationClass,
            x: x,
            y: y,
            w: w,
            h: h
        });
        updateYoloLogs();
    }
    
    drawAnnotationScene();
}

// Clear all coordinates drawn
function clearAnnotations() {
    annotations = [];
    updateYoloLogs();
    drawAnnotationScene();
}

// Simulate HSV Auto-Labeling
function autoLabelAnnotations() {
    annotations = [
        { classId: 0, x: 85, y: 170, w: 50, h: 42 },
        { classId: 0, x: 265, y: 80, w: 50, h: 42 },
        { classId: 1, x: 280, y: 215, w: 38, h: 32 },
        { classId: 1, x: 135, y: 68, w: 32, h: 26 }
    ];
    updateYoloLogs();
    drawAnnotationScene();
}

// Generate normalized YOLO logs coordinates output
function updateYoloLogs() {
    const logsContainer = document.getElementById('yoloLogs');
    if (!logsContainer) return;
    
    if (annotations.length === 0) {
        logsContainer.innerHTML = '<span style="color:var(--text-muted); font-style:italic;">No boxes drawn yet...</span>';
        return;
    }
    
    let logsHTML = '';
    annotations.forEach(box => {
        const imgW = annotatorCanvas.width;
        const imgH = annotatorCanvas.height;
        
        const xc = ((box.x + box.w / 2) / imgW).toFixed(6);
        const yc = ((box.y + box.h / 2) / imgH).toFixed(6);
        const nw = (box.w / imgW).toFixed(6);
        const nh = (box.h / imgH).toFixed(6);
        
        logsHTML += `${box.classId} ${xc} ${yc} ${nw} ${nh}\n`;
    });
    
    logsContainer.textContent = logsHTML.trim();
}

// Run initial configurations
window.onload = init;
