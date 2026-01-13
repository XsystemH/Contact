// Manual Annotation Tool - Three.js based face selection with brush support
const socket = io();

// ============================================
// State Management
// ============================================
let currentTask = null;
let currentTaskId = null;

// Metrics for manual annotation (clicks + time)
let manualMetrics = {
    task_id: null,
    start_ts_ms: null,
    end_ts_ms: null,
    duration_s: null,
    click_count: 0, // pure click events
    drag_count: 0,  // orbit/drag sessions
    zoom_count: 0,  // wheel/pinch zoom events (throttled)
    interaction_count: 0, // derived at save
};

function _nowMs() {
    return Date.now();
}

// Count ALL clicks on this page (face selections, brush interactions, UI buttons)
document.addEventListener(
    'click',
    () => {
        manualMetrics.click_count += 1;
    },
    true
);

// Drag/zoom tracking for Three.js (OrbitControls: drag rotate, wheel zoom)
let _dragActive = false;
let _dragMoved = false;
let _dragStart = { x: 0, y: 0 };
const _DRAG_THRESHOLD_PX = 6;
let _wheelLastMs = 0;
const _WHEEL_THROTTLE_MS = 200;
let _pinchActive = false;

function _isInThree(el) {
    try {
        return !!(threeContainer && el && threeContainer.contains(el));
    } catch (e) {
        return false;
    }
}

document.addEventListener(
    'pointerdown',
    (e) => {
        if (!_isInThree(e.target)) return;
        _dragActive = true;
        _dragMoved = false;
        _dragStart = { x: e.clientX, y: e.clientY };
    },
    true
);
document.addEventListener(
    'pointermove',
    (e) => {
        if (!_dragActive) return;
        const dx = e.clientX - _dragStart.x;
        const dy = e.clientY - _dragStart.y;
        if (!_dragMoved && (dx * dx + dy * dy >= _DRAG_THRESHOLD_PX * _DRAG_THRESHOLD_PX)) {
            _dragMoved = true;
        }
    },
    true
);
document.addEventListener(
    'pointerup',
    () => {
        if (_dragActive && _dragMoved) {
            manualMetrics.drag_count += 1;
        }
        _dragActive = false;
        _dragMoved = false;
    },
    true
);

document.addEventListener(
    'wheel',
    (e) => {
        if (!_isInThree(e.target)) return;
        const now = _nowMs();
        if (now - _wheelLastMs >= _WHEEL_THROTTLE_MS) {
            manualMetrics.zoom_count += 1;
            _wheelLastMs = now;
        }
    },
    true
);

document.addEventListener(
    'touchstart',
    (e) => {
        if (!_isInThree(e.target)) return;
        if (e.touches && e.touches.length >= 2) {
            _pinchActive = true;
        }
    },
    true
);
document.addEventListener(
    'touchend',
    (e) => {
        if (!_pinchActive) return;
        const touches = e.touches ? e.touches.length : 0;
        if (touches < 2) {
            manualMetrics.zoom_count += 1;
            _pinchActive = false;
        }
    },
    true
);

// Mode: 'navigate' | 'annotate' | 'brush' | 'erase'
let mode = 'navigate';

// Face selection
let selectedFaces = new Set();      // Permanently selected (red)
let hoveredFaceIndex = -1;
let strokePendingFaces = new Set();  // Faces during brush/erase stroke (orange)

// Brush state
let isBrushing = false;
let brushSize = 30;  // Brush radius in pixels

// Stroke behavior: 'add' (brush) or 'erase'
let strokeAction = 'add';

// Hand region mapping (loaded from mapping.json)
let regionFaceMapping = {};         // {region_id: array of face indices}

// Hand region checkbox selection
let handRegionSelection = {};       // {region_id: true/false}

// Undo/Redo stacks
let undoStack = [];  // Array of {type, data} actions
let redoStack = [];
const MAX_UNDO_STACK = 50;

// Three.js objects
let scene, camera, renderer, controls;
let humanMesh = null;
let humanGeometry = null;
let raycaster, mouse;
let originalFaces = null;  // Store original face data for vertex lookup

// ============================================
// Colors (as hex integers)
// ============================================
const COLORS = {
    default: 0x8899aa,      // Gray-blue
    hover: 0xffff00,        // Yellow
    selected: 0xff0000,     // Red
    brushPending: 0xffcc00, // Orange-yellow
    mapped: 0x2196F3        // Blue (for mapped regions)
};

// ============================================
// Elements
// ============================================
const threeContainer = document.getElementById('threeContainer');
const faceTooltip = document.getElementById('faceTooltip');
const refImage = document.getElementById('refImage');

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initEventListeners();
    initHandRegionUI();
    loadMappingFromServer();
    loadTaskData();
});

function initThreeJS() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    
    // Camera
    const aspect = threeContainer.clientWidth / threeContainer.clientHeight;
    camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    camera.position.set(0, 0, 3);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    threeContainer.appendChild(renderer.domElement);
    
    // Controls with extended rotation range
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    // Remove polar angle limits for full rotation
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI;
    // Allow full azimuth rotation
    controls.minAzimuthAngle = -Infinity;
    controls.maxAzimuthAngle = Infinity;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);
    
    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-5, -5, -5);
    scene.add(backLight);
    
    // Additional lights for better visibility from all angles
    const topLight = new THREE.DirectionalLight(0xffffff, 0.4);
    topLight.position.set(0, 10, 0);
    scene.add(topLight);
    
    const bottomLight = new THREE.DirectionalLight(0xffffff, 0.2);
    bottomLight.position.set(0, -10, 0);
    scene.add(bottomLight);
    
    // Raycaster for face picking
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    
    // Animation loop
    animate();
    
    // Resize handler
    window.addEventListener('resize', onWindowResize);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const width = threeContainer.clientWidth;
    const height = threeContainer.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// ============================================
// Event Listeners
// ============================================
function initEventListeners() {
    // Mode buttons
    document.getElementById('btnNavigate').addEventListener('click', () => setMode('navigate'));
    document.getElementById('btnAnnotate').addEventListener('click', () => setMode('annotate'));
    const btnBrush = document.getElementById('btnBrush');
    if (btnBrush) {
        btnBrush.addEventListener('click', () => setMode('brush'));
    }

    const btnErase = document.getElementById('btnErase');
    if (btnErase) {
        btnErase.addEventListener('click', () => setMode('erase'));
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === '1') setMode('navigate');
        if (e.key === '2') setMode('annotate');
        if (e.key === '3') setMode('brush');
        if (e.key === '4') setMode('erase');
        if (e.key === 'Escape') {
            cancelBrushStroke();
        }
        // Ctrl+Z for undo, Ctrl+Shift+Z or Ctrl+Y for redo
        if (e.ctrlKey && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            undo();
        }
        if (e.ctrlKey && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
            e.preventDefault();
            redo();
        }
    });
    
    // Mouse events for face selection and brushing
    threeContainer.addEventListener('mousemove', onMouseMove);
    threeContainer.addEventListener('mousedown', onMouseDown);
    threeContainer.addEventListener('mouseup', onMouseUp);
    threeContainer.addEventListener('click', onMouseClick);
    
    // Brush size slider
    const brushSlider = document.getElementById('brushSize');
    if (brushSlider) {
        brushSlider.addEventListener('input', (e) => {
            brushSize = parseInt(e.target.value);
            document.getElementById('brushSizeValue').textContent = brushSize;
        });
    }
    
    // Brush confirm/cancel buttons
    const btnConfirmBrush = document.getElementById('btnConfirmBrush');
    if (btnConfirmBrush) {
        btnConfirmBrush.addEventListener('click', confirmBrushStroke);
    }
    
    const btnCancelBrush = document.getElementById('btnCancelBrush');
    if (btnCancelBrush) {
        btnCancelBrush.addEventListener('click', cancelBrushStroke);
    }
    
    // Action buttons
    document.getElementById('btnSave').addEventListener('click', saveAnnotation);
    document.getElementById('btnClear').addEventListener('click', clearSelection);
    document.getElementById('btnUndo').addEventListener('click', undo);
    document.getElementById('btnRedo').addEventListener('click', redo);
    document.getElementById('btnBack').addEventListener('click', goBack);
}

// ============================================
// Mode Management
// ============================================
function setMode(newMode) {
    // Cancel any pending brush strokes when switching modes
    if (strokePendingFaces.size > 0) {
        cancelBrushStroke();
    }
    
    mode = newMode;
    
    // Update UI
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    
    const modeNames = {
        'navigate': 'Navigate',
        'annotate': 'Annotate',
        'brush': 'Brush',
        'erase': 'Erase'
    };
    document.getElementById('currentMode').textContent = modeNames[mode] || mode;
    
    const modeInfos = {
        'navigate': 'Drag to rotate, scroll to zoom',
        'annotate': 'Click faces to select/deselect',
        'brush': 'Hold mouse to paint, then confirm',
        'erase': 'Hold mouse to erase, then confirm'
    };
    document.getElementById('modeInfo').textContent = modeInfos[mode] || '';
    
    // Update container class
    threeContainer.className = '';
    threeContainer.classList.add(`${mode}-mode`);
    
    // Enable/disable orbit controls
    controls.enabled = (mode === 'navigate');
    
    // Show/hide brush controls
    const brushControls = document.getElementById('brushControls');
    if (brushControls) {
        brushControls.style.display = (mode === 'brush' || mode === 'erase') ? 'block' : 'none';
    }

    strokeAction = (mode === 'erase') ? 'erase' : 'add';
    
    // Reset hover state
    if (hoveredFaceIndex >= 0) {
        restoreFaceColor(hoveredFaceIndex);
        hoveredFaceIndex = -1;
    }
    faceTooltip.style.display = 'none';
}

// ============================================
// Mouse Event Handlers
// ============================================
function getMousePosition(event) {
    const rect = threeContainer.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function getFaceAtMouse() {
    if (!humanMesh) return -1;
    
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(humanMesh);
    
    if (intersects.length > 0) {
        return intersects[0].faceIndex;
    }
    return -1;
}

function onMouseMove(event) {
    if (mode === 'navigate') return;
    
    const pos = getMousePosition(event);
    const faceIndex = getFaceAtMouse();
    
    // Handle brush/erase mode with mouse held down
    if ((mode === 'brush' || mode === 'erase') && isBrushing) {
        const facesInBrush = getVisibleFacesInRadius(pos.x, pos.y, brushSize);
        facesInBrush.forEach(fi => {
            if (strokeAction === 'add') {
                if (!selectedFaces.has(fi) && !strokePendingFaces.has(fi)) {
                    strokePendingFaces.add(fi);
                    setFaceColor(fi, COLORS.brushPending);
                }
            } else {
                // erase action: only mark faces currently selected
                if (selectedFaces.has(fi) && !strokePendingFaces.has(fi)) {
                    strokePendingFaces.add(fi);
                    setFaceColor(fi, COLORS.brushPending);
                }
            }
        });
        return;
    }
    
    // Handle hover for annotate/brush/erase mode
    if (mode === 'annotate' || mode === 'brush' || mode === 'erase') {
        if (faceIndex !== hoveredFaceIndex) {
            // Restore previous hover
            if (hoveredFaceIndex >= 0) {
                restoreFaceColor(hoveredFaceIndex);
            }
            
            hoveredFaceIndex = faceIndex;
            
            // Highlight new hover if not already colored
            if (faceIndex >= 0 && !selectedFaces.has(faceIndex) && !strokePendingFaces.has(faceIndex)) {
                setFaceColor(faceIndex, COLORS.hover);
            }
        }
        
        // Show tooltip
        if (faceIndex >= 0) {
            faceTooltip.style.display = 'block';
            faceTooltip.style.left = (pos.x + 10) + 'px';
            faceTooltip.style.top = (pos.y + 10) + 'px';
            faceTooltip.textContent = `Face: ${faceIndex}`;
        } else {
            faceTooltip.style.display = 'none';
        }
    }
}

function onMouseDown(event) {
    if (event.button !== 0) return; // Only left mouse button
    
    if (mode === 'brush' || mode === 'erase') {
        isBrushing = true;
        const pos = getMousePosition(event);
        
        // Start painting - only visible front-facing faces
        const facesInBrush = getVisibleFacesInRadius(pos.x, pos.y, brushSize);
        facesInBrush.forEach(fi => {
            if (strokeAction === 'add') {
                if (!selectedFaces.has(fi) && !strokePendingFaces.has(fi)) {
                    strokePendingFaces.add(fi);
                    setFaceColor(fi, COLORS.brushPending);
                }
            } else {
                if (selectedFaces.has(fi) && !strokePendingFaces.has(fi)) {
                    strokePendingFaces.add(fi);
                    setFaceColor(fi, COLORS.brushPending);
                }
            }
        });
        
        // Show confirm/cancel buttons
        updateBrushControlsVisibility();
    }
}

function onMouseUp(event) {
    if (event.button !== 0) return;
    
    if ((mode === 'brush' || mode === 'erase') && isBrushing) {
        isBrushing = false;
        updateBrushControlsVisibility();
    }
}

function onMouseClick(event) {
    if (mode === 'navigate' || mode === 'brush' || mode === 'erase') return;
    
    const faceIndex = getFaceAtMouse();
    if (faceIndex < 0) return;
    
    if (mode === 'annotate') {
        toggleFaceSelection(faceIndex);
    }
}

// ============================================
// Brush Functions - With Visibility Check
// ============================================
function getVisibleFacesInRadius(screenX, screenY, radius) {
    const faces = [];
    if (!humanMesh || !humanGeometry) return faces;
    
    const positions = humanGeometry.attributes.position;
    const normals = humanGeometry.attributes.normal;
    const numFaces = positions.count / 3;
    
    // Get camera direction for back-face culling
    const cameraDir = new THREE.Vector3();
    camera.getWorldDirection(cameraDir);
    
    for (let i = 0; i < numFaces; i++) {
        // Get face center (average of 3 vertices)
        const baseIdx = i * 3;
        let cx = 0, cy = 0, cz = 0;
        let nx = 0, ny = 0, nz = 0;
        
        for (let v = 0; v < 3; v++) {
            cx += positions.getX(baseIdx + v);
            cy += positions.getY(baseIdx + v);
            cz += positions.getZ(baseIdx + v);
            nx += normals.getX(baseIdx + v);
            ny += normals.getY(baseIdx + v);
            nz += normals.getZ(baseIdx + v);
        }
        cx /= 3; cy /= 3; cz /= 3;
        nx /= 3; ny /= 3; nz /= 3;
        
        // Check if face is front-facing (normal points towards camera)
        const faceNormal = new THREE.Vector3(nx, ny, nz).normalize();
        const faceCenterWorld = new THREE.Vector3(cx, cy, cz);
        faceCenterWorld.applyMatrix4(humanMesh.matrixWorld);
        
        // Direction from face to camera
        const toCamera = new THREE.Vector3().subVectors(camera.position, faceCenterWorld).normalize();
        
        // Dot product: positive means facing camera
        const dot = faceNormal.dot(toCamera);
        if (dot <= 0) continue; // Back-facing, skip
        
        // Project to screen space
        const pos3D = new THREE.Vector3(cx, cy, cz);
        pos3D.applyMatrix4(humanMesh.matrixWorld);
        pos3D.project(camera);
        
        // Check if behind camera
        if (pos3D.z > 1) continue;
        
        const rect = threeContainer.getBoundingClientRect();
        const screenPosX = (pos3D.x + 1) / 2 * rect.width;
        const screenPosY = (-pos3D.y + 1) / 2 * rect.height;
        
        // Check if within brush radius
        const dx = screenPosX - screenX;
        const dy = screenPosY - screenY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist <= radius) {
            // Check occlusion using raycaster
            const rayOrigin = camera.position.clone();
            const rayDir = new THREE.Vector3().subVectors(faceCenterWorld, rayOrigin).normalize();
            
            raycaster.set(rayOrigin, rayDir);
            const intersects = raycaster.intersectObject(humanMesh);
            
            if (intersects.length > 0) {
                // Check if the first hit is this face (or very close to it)
                const hitFaceIndex = intersects[0].faceIndex;
                if (hitFaceIndex === i) {
                    faces.push(i);
                }
            }
        }
    }
    
    return faces;
}

function confirmBrushStroke() {
    if (strokePendingFaces.size === 0) return;

    const faces = Array.from(strokePendingFaces);
    pushUndoAction('stroke', { action: strokeAction, faces });

    if (strokeAction === 'add') {
        faces.forEach(fi => {
            selectedFaces.add(fi);
            setFaceColor(fi, COLORS.selected);
        });
    } else {
        // Erase: remove from selection first, then refresh colors.
        faces.forEach(fi => {
            selectedFaces.delete(fi);
        });
    }

    strokePendingFaces.clear();
    refreshAllFaceColors();
    updateStats();
    updateBrushControlsVisibility();
}

function cancelBrushStroke() {
    // Restore pending faces to original color
    strokePendingFaces.forEach(fi => {
        restoreFaceColor(fi);
    });
    strokePendingFaces.clear();
    
    // Force refresh all face colors to ensure clean state
    refreshAllFaceColors();
    
    updateBrushControlsVisibility();
}

function updateBrushControlsVisibility() {
    const confirmBtn = document.getElementById('btnConfirmBrush');
    const cancelBtn = document.getElementById('btnCancelBrush');
    
    if (confirmBtn && cancelBtn) {
        const showButtons = strokePendingFaces.size > 0;
        confirmBtn.style.display = showButtons ? 'inline-block' : 'none';
        cancelBtn.style.display = showButtons ? 'inline-block' : 'none';
    }
}

// ============================================
// Face Selection & Coloring
// ============================================
function toggleFaceSelection(faceIndex) {
    // Save for undo
    pushUndoAction('toggle', { faceIndex, wasSelected: selectedFaces.has(faceIndex) });
    
    if (selectedFaces.has(faceIndex)) {
        selectedFaces.delete(faceIndex);
        setFaceColor(faceIndex, COLORS.hover);
    } else {
        selectedFaces.add(faceIndex);
        setFaceColor(faceIndex, COLORS.selected);
    }
    updateStats();
}

function setFaceColor(faceIndex, color) {
    if (!humanGeometry) return;
    
    const colors = humanGeometry.attributes.color;
    if (!colors) return;
    
    const r = ((color >> 16) & 255) / 255;
    const g = ((color >> 8) & 255) / 255;
    const b = (color & 255) / 255;
    
    // Each face has 3 vertices in non-indexed geometry
    const i = faceIndex * 3;
    colors.setXYZ(i, r, g, b);
    colors.setXYZ(i + 1, r, g, b);
    colors.setXYZ(i + 2, r, g, b);
    colors.needsUpdate = true;
}

function restoreFaceColor(faceIndex) {
    // Pending stroke color should override selected color (important for erase mode)
    if (strokePendingFaces.has(faceIndex)) {
        setFaceColor(faceIndex, COLORS.brushPending);
    } else if (selectedFaces.has(faceIndex)) {
        setFaceColor(faceIndex, COLORS.selected);
    } else {
        setFaceColor(faceIndex, COLORS.default);
    }
}

function refreshAllFaceColors() {
    if (!humanGeometry) return;
    
    const numFaces = humanGeometry.attributes.position.count / 3;
    for (let i = 0; i < numFaces; i++) {
        // Pending stroke color should override selected color (important for erase mode)
        if (strokePendingFaces.has(i)) {
            setFaceColor(i, COLORS.brushPending);
        } else if (selectedFaces.has(i)) {
            setFaceColor(i, COLORS.selected);
        } else {
            setFaceColor(i, COLORS.default);
        }
    }
}

// ============================================
// Undo/Redo System
// ============================================
function pushUndoAction(type, data) {
    undoStack.push({ type, data, selectedFacesCopy: new Set(selectedFaces) });
    if (undoStack.length > MAX_UNDO_STACK) {
        undoStack.shift();
    }
    // Clear redo stack on new action
    redoStack = [];
    updateUndoRedoButtons();
}

function undo() {
    if (undoStack.length === 0) return;
    
    const action = undoStack.pop();
    
    // Save current state to redo stack
    redoStack.push({ 
        type: 'restore', 
        data: new Set(selectedFaces)
    });
    
    // Restore previous state
    selectedFaces = new Set(action.selectedFacesCopy);
    
    // Apply to selected faces from before the action
    if (action.type === 'toggle') {
        // For toggle, just restore
    } else if (action.type === 'stroke') {
        // Restore snapshot is enough; keep for compatibility
    } else if (action.type === 'region') {
        // For region, remove the region faces
        action.data.forEach(fi => selectedFaces.delete(fi));
    }
    
    refreshAllFaceColors();
    updateStats();
    updateUndoRedoButtons();
}

function redo() {
    if (redoStack.length === 0) return;
    
    const action = redoStack.pop();
    
    // Save current state to undo stack (without triggering clear of redo)
    undoStack.push({ 
        type: 'restore', 
        data: null, 
        selectedFacesCopy: new Set(selectedFaces) 
    });
    
    // Restore the redo state
    if (action.type === 'restore') {
        selectedFaces = new Set(action.data);
    }
    
    refreshAllFaceColors();
    updateStats();
    updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
    const btnUndo = document.getElementById('btnUndo');
    const btnRedo = document.getElementById('btnRedo');
    
    if (btnUndo) {
        btnUndo.disabled = undoStack.length === 0;
        btnUndo.style.opacity = undoStack.length === 0 ? 0.5 : 1;
    }
    if (btnRedo) {
        btnRedo.disabled = redoStack.length === 0;
        btnRedo.style.opacity = redoStack.length === 0 ? 0.5 : 1;
    }
}

// ============================================
// Hand Region UI (Checkboxes)
// ============================================
function initHandRegionUI() {
    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    const segments = ['base', 'middle', 'tip'];
    const sides = ['F', 'B', 'L', 'R'];
    
    ['left', 'right'].forEach(hand => {
        const container = document.getElementById(`${hand}HandRegions`);
        if (!container) return;
        
        let html = '';
        
        // Fingers
        fingers.forEach(finger => {
            html += `<div class="finger-group">
                <div class="finger-name">${finger.charAt(0).toUpperCase() + finger.slice(1)}</div>`;
            
            segments.forEach(segment => {
                html += `<div class="region-checkboxes">`;
                sides.forEach(side => {
                    const id = `${hand}_${finger}_${segment}_${side}`;
                    html += `<div class="region-cb">
                        <input type="checkbox" id="${id}" data-region="${id}">
                        <label for="${id}">${side}</label>
                    </div>`;
                });
                html += `</div>`;
            });
            html += `</div>`;
        });
        
        // Palm
        html += `<div class="finger-group">
            <div class="finger-name">Palm (front/back)</div>
            <div class="palm-regions">`;
        fingers.forEach(finger => {
            const id = `${hand}_palm_${finger}`;
            html += `<div class="region-cb">
                <input type="checkbox" id="${id}" data-region="${id}">
                <label for="${id}">${finger.charAt(0).toUpperCase()}</label>
            </div>`;
        });
        html += `</div>
            <div class="palm-regions">`;
        fingers.forEach(finger => {
            const id = `${hand}_back_${finger}`;
            html += `<div class="region-cb">
                <input type="checkbox" id="${id}" data-region="${id}">
                <label for="${id}">${finger.charAt(0).toUpperCase()}b</label>
            </div>`;
        });
        html += `</div></div>`;
        
        container.innerHTML = html;
    });
    
    // Add event listeners to checkboxes
    document.querySelectorAll('.region-cb input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', onHandRegionChange);
    });
}

function onHandRegionChange(event) {
    const regionId = event.target.dataset.region;
    const checked = event.target.checked;
    handRegionSelection[regionId] = checked;
    
    // If region has mapped faces, add/remove them from selection
    const mappedFaces = regionFaceMapping[regionId] || [];
    if (mappedFaces.length > 0) {
        // Save for undo
        pushUndoAction('region', mappedFaces);
        
        mappedFaces.forEach(fi => {
            if (checked) {
                selectedFaces.add(fi);
                setFaceColor(fi, COLORS.selected);
            } else {
                selectedFaces.delete(fi);
                restoreFaceColor(fi);
            }
        });
    }
    
    updateStats();
}

// ============================================
// Load Mapping from Server
// ============================================
function loadMappingFromServer() {
    // Load mapping.json from static files
    fetch('/static/mapping.json')
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            return {};
        })
        .then(data => {
            regionFaceMapping = data || {};
            console.log('Loaded face mapping:', Object.keys(regionFaceMapping).length, 'regions');
        })
        .catch(err => {
            console.warn('Failed to load mapping.json:', err);
            regionFaceMapping = {};
        });
}

// ============================================
// Data Loading
// ============================================
function loadTaskData() {
    const params = new URLSearchParams(window.location.search);
    const taskId = params.get('task_id');
    
    if (taskId) {
        socket.emit('request_manual_annotation_data', { task_id: parseInt(taskId) });
    }
}

// Socket events
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('manual_annotation_data', (data) => {
    currentTask = data.task;
    currentTaskId = data.task_id;

    manualMetrics = {
        task_id: currentTaskId,
        start_ts_ms: _nowMs(),
        end_ts_ms: null,
        duration_s: null,
        click_count: 0,
        drag_count: 0,
        zoom_count: 0,
        interaction_count: 0,
    };
    
    document.getElementById('taskPath').textContent = currentTask.relative_path;
    
    // Load reference image
    if (data.reference_image) {
        refImage.src = 'data:image/jpeg;base64,' + data.reference_image;
    }
    
    // Store original faces
    originalFaces = data.faces;
    
    // Create mesh from vertices and faces
    createHumanMesh(data.vertices, data.faces);
    
    // Load existing manual annotation if any
    if (data.existing_annotation && data.existing_annotation.length > 0) {
        selectedFaces = new Set(data.existing_annotation);
        refreshAllFaceColors();
    }
    
    // Update stats
    document.getElementById('statVertices').textContent = data.vertices.length;
    updateStats();
    updateUndoRedoButtons();
});

function createHumanMesh(vertices, faces) {
    // Remove existing mesh
    if (humanMesh) {
        scene.remove(humanMesh);
        humanGeometry.dispose();
    }
    
    // Create geometry
    humanGeometry = new THREE.BufferGeometry();
    
    // Convert to flat arrays for indexed geometry
    const positions = [];
    const indices = [];
    
    for (let i = 0; i < vertices.length; i++) {
        positions.push(vertices[i][0], vertices[i][1], vertices[i][2]);
    }
    
    for (let i = 0; i < faces.length; i++) {
        indices.push(faces[i][0], faces[i][1], faces[i][2]);
    }
    
    humanGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    humanGeometry.setIndex(indices);
    humanGeometry.computeVertexNormals();
    
    // Convert to non-indexed geometry for per-face coloring
    humanGeometry = humanGeometry.toNonIndexed();
    
    // Recompute normals for non-indexed geometry
    humanGeometry.computeVertexNormals();
    
    // Initialize colors
    const positionCount = humanGeometry.attributes.position.count;
    const colorArray = new Float32Array(positionCount * 3);
    for (let i = 0; i < positionCount; i++) {
        colorArray[i * 3] = 0.53;     // R
        colorArray[i * 3 + 1] = 0.60; // G
        colorArray[i * 3 + 2] = 0.67; // B
    }
    humanGeometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    
    // Create material with vertex colors
    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        flatShading: true
    });
    
    // Add subtle wireframe
    const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: 0x000000,
        wireframe: true,
        transparent: true,
        opacity: 0.05
    });
    
    humanMesh = new THREE.Mesh(humanGeometry, material);
    const wireframe = new THREE.Mesh(humanGeometry, wireframeMaterial);
    humanMesh.add(wireframe);
    
    scene.add(humanMesh);
    
    // Center camera on mesh
    const box = new THREE.Box3().setFromObject(humanMesh);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + size.length() * 1.2);
    
    updateStats();
}

// ============================================
// Statistics
// ============================================
function updateStats() {
    document.getElementById('statFaces').textContent = selectedFaces.size;
    
    const handRegionCount = Object.values(handRegionSelection).filter(v => v).length;
    document.getElementById('statHandRegions').textContent = handRegionCount;
    
    // Calculate approximate vertex count
    const vertexCount = selectedFaces.size * 3;  // Each face has ~3 unique vertices
    document.getElementById('statVertices').textContent = vertexCount || '-';
}

// ============================================
// Save & Navigation
// ============================================
function saveAnnotation() {
    // Send annotation back to main viewer (don't navigate away)
    manualMetrics.end_ts_ms = _nowMs();
    if (manualMetrics.start_ts_ms != null) {
        manualMetrics.duration_s = Math.max(0, (manualMetrics.end_ts_ms - manualMetrics.start_ts_ms) / 1000.0);
    }
    manualMetrics.interaction_count = (manualMetrics.click_count || 0) + (manualMetrics.drag_count || 0) + (manualMetrics.zoom_count || 0);
    const annotationData = {
        task_id: currentTaskId,
        selected_faces: Array.from(selectedFaces),
        hand_regions: handRegionSelection,
        metrics: manualMetrics,
    };
    
    // Emit to server to store temporarily
    socket.emit('save_manual_annotation', annotationData);
}

socket.on('annotation_saved', (data) => {
    if (data.success) {
        // Show success message and close window
        alert('Annotation saved! Returning to viewer...');
        // Try to close the window (works if opened via window.open)
        window.close();
        // Note: If window.close() doesn't work (not opened by script), 
        // the user will need to close the tab manually
    } else {
        alert('Error saving annotation: ' + data.error);
    }
});

function clearSelection() {
    // Save for undo
    if (selectedFaces.size > 0) {
        pushUndoAction('clear', Array.from(selectedFaces));
    }
    
    // Clear face selection
    selectedFaces.forEach(faceIndex => {
        restoreFaceColor(faceIndex);
    });
    selectedFaces.clear();
    
    // Clear stroke pending
    strokePendingFaces.clear();
    
    // Clear hand region selection
    document.querySelectorAll('.region-cb input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    handRegionSelection = {};
    
    refreshAllFaceColors();
    updateStats();
    updateBrushControlsVisibility();
}

function goBack() {
    // Warn if there are unsaved changes
    if (selectedFaces.size > 0) {
        if (!confirm('You have unsaved annotations. Are you sure you want to go back?')) {
            return;
        }
    }
    // Try to close the window (works if opened via window.open from viewer)
    window.close();
    // Note: If window.close() doesn't work, user will need to close tab manually
}

// ============================================
// Page Visibility
// ============================================
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        onWindowResize();
    }
});
