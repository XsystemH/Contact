// Socket.IO connection
const socket = io();

// State
let currentTask = null;
let currentTaskId = null;
let vizData = null;
let currentTool = 'navigate';
let brushMode = 'add';
let brushSize = 5;
let targetMesh = 'human';

// Annotation data
let humanAnnotation = null;  // Boolean array
let objectAnnotation = null; // Boolean array
let selectedRegions = new Set();

// Mesh face data for face-based selection
let humanFaces = null;
let objectFaces = null;

// SMPLX hand vertex mapping (based on SMPLX model specification)
// SMPLX has 10475 vertices total
// Left hand vertices: ~5443-5556 (varies by exact finger)
// Right hand vertices: ~5660-5772
// These ranges define approximate regions for each finger and area
const HAND_REGIONS = {
    left: {
        // Left hand fingers (approximate SMPLX vertex ranges)
        thumb: { 
            tip: Array.from({length: 12}, (_, i) => 5443 + i),
            middle: Array.from({length: 12}, (_, i) => 5455 + i),
            base: Array.from({length: 12}, (_, i) => 5467 + i)
        },
        index: { 
            tip: Array.from({length: 12}, (_, i) => 5479 + i),
            middle: Array.from({length: 12}, (_, i) => 5491 + i),
            base: Array.from({length: 12}, (_, i) => 5503 + i)
        },
        middle: { 
            tip: Array.from({length: 12}, (_, i) => 5515 + i),
            middle: Array.from({length: 12}, (_, i) => 5527 + i),
            base: Array.from({length: 12}, (_, i) => 5539 + i)
        },
        ring: { 
            tip: Array.from({length: 10}, (_, i) => 5551 + i),
            middle: Array.from({length: 10}, (_, i) => 5561 + i),
            base: Array.from({length: 10}, (_, i) => 5571 + i)
        },
        pinky: { 
            tip: Array.from({length: 10}, (_, i) => 5581 + i),
            middle: Array.from({length: 10}, (_, i) => 5591 + i),
            base: Array.from({length: 10}, (_, i) => 5601 + i)
        },
        palm: {
            center: Array.from({length: 30}, (_, i) => 5611 + i),
            thumb_side: Array.from({length: 15}, (_, i) => 5641 + i),
            pinky_side: Array.from({length: 15}, (_, i) => 5656 + i)
        }
    },
    right: {
        // Right hand fingers (approximate SMPLX vertex ranges)
        thumb: { 
            tip: Array.from({length: 12}, (_, i) => 5775 + i),
            middle: Array.from({length: 12}, (_, i) => 5787 + i),
            base: Array.from({length: 12}, (_, i) => 5799 + i)
        },
        index: { 
            tip: Array.from({length: 12}, (_, i) => 5811 + i),
            middle: Array.from({length: 12}, (_, i) => 5823 + i),
            base: Array.from({length: 12}, (_, i) => 5835 + i)
        },
        middle: { 
            tip: Array.from({length: 12}, (_, i) => 5847 + i),
            middle: Array.from({length: 12}, (_, i) => 5859 + i),
            base: Array.from({length: 12}, (_, i) => 5871 + i)
        },
        ring: { 
            tip: Array.from({length: 10}, (_, i) => 5883 + i),
            middle: Array.from({length: 10}, (_, i) => 5893 + i),
            base: Array.from({length: 10}, (_, i) => 5903 + i)
        },
        pinky: { 
            tip: Array.from({length: 10}, (_, i) => 5913 + i),
            middle: Array.from({length: 10}, (_, i) => 5923 + i),
            base: Array.from({length: 10}, (_, i) => 5933 + i)
        },
        palm: {
            center: Array.from({length: 30}, (_, i) => 5943 + i),
            thumb_side: Array.from({length: 15}, (_, i) => 5973 + i),
            pinky_side: Array.from({length: 15}, (_, i) => 5988 + i)
        }
    }
};

// Socket events
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('annotation_data', (data) => {
    console.log('Received annotation_data:', data);
    currentTask = data.task;
    currentTaskId = data.task_id;
    vizData = data.viz_data;
    
    console.log('Human vertices count:', vizData.human_verts.length);
    console.log('Object vertices count:', vizData.obj_verts.length);
    console.log('Human faces count:', vizData.human_faces ? vizData.human_faces.length : 0);
    console.log('Object faces count:', vizData.obj_faces ? vizData.obj_faces.length : 0);
    
    // Store face data for mesh-based selection
    humanFaces = vizData.human_faces || null;
    objectFaces = vizData.obj_faces || null;
    
    // Initialize annotation arrays
    humanAnnotation = new Array(vizData.human_verts.length).fill(false);
    objectAnnotation = new Array(vizData.obj_verts.length).fill(false);
    
    // Load existing annotation if any
    if (data.existing_annotation) {
        humanAnnotation = data.existing_annotation.human || humanAnnotation;
        objectAnnotation = data.existing_annotation.object || objectAnnotation;
    }
    
    displayVisualization();
    updateStats();
});

socket.on('annotation_saved', (data) => {
    if (data.success) {
        showMessage('Annotation saved successfully!', 'success');
    } else {
        showMessage('Failed to save annotation: ' + data.error, 'error');
    }
});

// Tool selection
function selectTool(tool) {
    const previousTool = currentTool;
    currentTool = tool;
    
    document.getElementById('toolNavigate').classList.toggle('active', tool === 'navigate');
    document.getElementById('toolBrush').classList.toggle('active', tool === 'brush');
    
    document.getElementById('brushSection').style.display = tool === 'brush' ? 'block' : 'none';
    
    // Re-render when switching between navigate and brush (different render modes)
    if (vizData && previousTool !== tool) {
        displayVisualization();
    }
}

// Brush controls
function updateBrushSize(value) {
    brushSize = parseInt(value);
    document.getElementById('brushSizeLabel').textContent = value;
}

function setBrushMode(mode) {
    brushMode = mode;
    document.getElementById('modeAdd').classList.toggle('active', mode === 'add');
    document.getElementById('modeErase').classList.toggle('active', mode === 'erase');
}

// Region selection
function toggleRegion(element) {
    const region = element.dataset.region;
    const checkbox = element.querySelector('.region-checkbox');
    
    if (selectedRegions.has(region)) {
        selectedRegions.delete(region);
        checkbox.checked = false;
        element.classList.remove('selected');
    } else {
        selectedRegions.add(region);
        checkbox.checked = true;
        element.classList.add('selected');
    }
}

function applyRegionSelection() {
    if (!humanAnnotation) return;
    
    selectedRegions.forEach(region => {
        const indices = getRegionIndices(region);
        indices.forEach(idx => {
            if (idx < humanAnnotation.length) {
                humanAnnotation[idx] = true;
            }
        });
    });
    
    displayVisualization();
    updateStats();
    showMessage(`Applied ${selectedRegions.size} region(s)`, 'success');
}

function getRegionIndices(region) {
    // Parse region string like "left_thumb_tip" or "left_palm_center"
    const parts = region.split('_');
    const side = parts[0]; // left or right
    const finger = parts[1]; // thumb, index, middle, ring, pinky, palm
    const area = parts[2]; // tip, middle, base, center, thumb_side, pinky_side
    
    let indices = [];
    
    if (HAND_REGIONS[side]) {
        if (finger === 'palm' && HAND_REGIONS[side].palm) {
            // Palm areas
            if (HAND_REGIONS[side].palm[area]) {
                indices = HAND_REGIONS[side].palm[area];
            }
        } else if (HAND_REGIONS[side][finger]) {
            // Finger areas
            if (HAND_REGIONS[side][finger][area]) {
                indices = HAND_REGIONS[side][finger][area];
            }
        }
    }
    
    return indices;
}

// Visualization
function displayVisualization() {
    if (!vizData) return;
    
    const camera = {
        eye: {
            x: 1.5 * Math.cos(vizData.camera.azimuth * Math.PI / 180) * Math.cos(vizData.camera.elevation * Math.PI / 180),
            y: 1.5 * Math.sin(vizData.camera.azimuth * Math.PI / 180) * Math.cos(vizData.camera.elevation * Math.PI / 180),
            z: 1.5 * Math.sin(vizData.camera.elevation * Math.PI / 180)
        }
    };
    
    const layout = {
        scene: {
            aspectmode: 'data',
            camera: camera,
            xaxis: {showgrid: false, zeroline: false, showticklabels: false},
            yaxis: {showgrid: false, zeroline: false, showticklabels: false},
            zaxis: {showgrid: false, zeroline: false, showticklabels: false},
            hovermode: 'closest'
        },
        margin: {l: 0, r: 0, t: 0, b: 0},
        showlegend: false,
        hovermode: 'closest',
        dragmode: currentTool === 'navigate' ? 'orbit' : false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    // Plot human mesh (clone layout for each plot)
    plotHumanMesh({...layout}, {...config});
    
    // Plot object mesh
    plotObjectMesh({...layout}, {...config});
    
    // Plot combined
    plotCombined({...layout}, {...config});
    
    // Reference image
    if (vizData.reference_image) {
        document.getElementById('refImage').src = 'data:image/jpeg;base64,' + vizData.reference_image;
    }
}

function plotHumanMesh(layout, config) {
    const humanVerts = vizData.human_verts;
    const traces = [];
    
    console.log('plotHumanMesh: verts=', humanVerts.length, 'faces=', humanFaces ? humanFaces.length : 0);
    
    // Use mesh3d for brush tool, point cloud for navigation (faster rendering)
    if (currentTool === 'brush' && humanFaces && humanFaces.length > 0) {
        // Calculate face colors based on annotation (if any vertex in face is annotated, highlight face)
        const faceColors = humanFaces.map(face => {
            const hasAnnotated = face.some(idx => humanAnnotation[idx]);
            return hasAnnotated ? 'rgba(255, 0, 0, 0.8)' : 'rgba(173, 216, 230, 0.4)';
        });
        
        traces.push({
            type: 'mesh3d',
            x: humanVerts.map(v => v[0]),
            y: humanVerts.map(v => v[1]),
            z: humanVerts.map(v => v[2]),
            i: humanFaces.map(f => f[0]),
            j: humanFaces.map(f => f[1]),
            k: humanFaces.map(f => f[2]),
            facecolor: faceColors,
            opacity: 0.7,
            flatshading: true,
            hoverinfo: 'skip',
            name: 'Human Mesh'
        });
        
        // Add invisible clickable point layer for brush interaction
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: humanVerts.map(v => v[0]),
            y: humanVerts.map(v => v[1]),
            z: humanVerts.map(v => v[2]),
            marker: {size: 2, color: 'rgba(0,0,0,0)', opacity: 0},
            customdata: humanVerts.map((_, i) => i),
            text: humanVerts.map((_, i) => `Vertex ${i}`),
            hoverinfo: 'text',
            name: 'Click Layer'
        });
    } else {
        // Fallback to point cloud if no faces
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: humanVerts.map(v => v[0]),
            y: humanVerts.map(v => v[1]),
            z: humanVerts.map(v => v[2]),
            marker: {size: 1, color: 'lightblue', opacity: 0.3},
            customdata: humanVerts.map((_, i) => i),
            hoverinfo: 'skip',
            name: 'Human Points'
        });
    }
    
    // Annotated vertices as overlay points for visibility
    const annotatedX = [];
    const annotatedY = [];
    const annotatedZ = [];
    const annotatedIndices = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (humanAnnotation[i]) {
            annotatedX.push(humanVerts[i][0]);
            annotatedY.push(humanVerts[i][1]);
            annotatedZ.push(humanVerts[i][2]);
            annotatedIndices.push(i);
        }
    }
    console.log('Human annotated vertices count:', annotatedX.length);
    if (annotatedX.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: annotatedX,
            y: annotatedY,
            z: annotatedZ,
            marker: {size: 3, color: 'red'},
            text: annotatedIndices.map(i => `Vertex ${i}`),
            hoverinfo: 'text',
            name: 'Annotated'
        });
    }
    
    const plotDiv = document.getElementById('plotHuman');
    console.log('Plotting human mesh with', traces.length, 'traces');
    Plotly.newPlot(plotDiv, traces, layout, config).then(() => {
        console.log('Human mesh plotted successfully');
        // Bind click handler for brush tool
        plotDiv.on('plotly_click', (data) => {
            console.log('Human mesh clicked, tool:', currentTool, 'target:', targetMesh);
            if (currentTool === 'brush' && targetMesh === 'human') {
                handleBrushClick(data, 'human');
            }
        });
    }).catch(error => {
        console.error('Error plotting human mesh:', error);
    });
}

function plotObjectMesh(layout, config) {
    const objVerts = vizData.obj_verts;
    const traces = [];
    
    console.log('plotObjectMesh: verts=', objVerts.length, 'faces=', objectFaces ? objectFaces.length : 0);
    
    // Use mesh3d for brush tool, point cloud for navigation (faster rendering)
    if (currentTool === 'brush' && objectFaces && objectFaces.length > 0) {
        // Calculate face colors based on annotation
        const faceColors = objectFaces.map(face => {
            const hasAnnotated = face.some(idx => objectAnnotation[idx]);
            return hasAnnotated ? 'rgba(255, 0, 0, 0.8)' : 'rgba(144, 238, 144, 0.4)';
        });
        
        traces.push({
            type: 'mesh3d',
            x: objVerts.map(v => v[0]),
            y: objVerts.map(v => v[1]),
            z: objVerts.map(v => v[2]),
            i: objectFaces.map(f => f[0]),
            j: objectFaces.map(f => f[1]),
            k: objectFaces.map(f => f[2]),
            facecolor: faceColors,
            opacity: 0.7,
            flatshading: true,
            hoverinfo: 'skip',
            name: 'Object Mesh'
        });
        
        // Add invisible clickable point layer for brush interaction
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: objVerts.map(v => v[0]),
            y: objVerts.map(v => v[1]),
            z: objVerts.map(v => v[2]),
            marker: {size: 2, color: 'rgba(0,0,0,0)', opacity: 0},
            customdata: objVerts.map((_, i) => i),
            text: objVerts.map((_, i) => `Vertex ${i}`),
            hoverinfo: 'text',
            name: 'Click Layer'
        });
    } else {
        // Fallback to point cloud if no faces
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: objVerts.map(v => v[0]),
            y: objVerts.map(v => v[1]),
            z: objVerts.map(v => v[2]),
            marker: {size: 1, color: 'lightgreen', opacity: 0.3},
            customdata: objVerts.map((_, i) => i),
            hoverinfo: 'skip',
            name: 'Object Points'
        });
    }
    
    // Annotated vertices as overlay points
    const annotatedX = [];
    const annotatedY = [];
    const annotatedZ = [];
    const annotatedIndices = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (objectAnnotation[i]) {
            annotatedX.push(objVerts[i][0]);
            annotatedY.push(objVerts[i][1]);
            annotatedZ.push(objVerts[i][2]);
            annotatedIndices.push(i);
        }
    }
    if (annotatedX.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: annotatedX,
            y: annotatedY,
            z: annotatedZ,
            marker: {size: 3, color: 'red'},
            text: annotatedIndices.map(i => `Vertex ${i}`),
            hoverinfo: 'text',
            name: 'Annotated'
        });
    }
    
    const plotDiv = document.getElementById('plotObject');
    console.log('Plotting object mesh with', traces.length, 'traces');
    Plotly.newPlot(plotDiv, traces, layout, config).then(() => {
        console.log('Object mesh plotted successfully');
        // Bind click handler for brush tool
        plotDiv.on('plotly_click', (data) => {
            console.log('Object mesh clicked, tool:', currentTool, 'target:', targetMesh);
            if (currentTool === 'brush' && targetMesh === 'object') {
                handleBrushClick(data, 'object');
            }
        });
    }).catch(error => {
        console.error('Error plotting object mesh:', error);
    });
}

function plotCombined(layout, config) {
    const traces = [];
    
    // Human vertices
    const humanVerts = vizData.human_verts;
    const humanAnnotatedPts = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (humanAnnotation[i]) {
            humanAnnotatedPts.push(humanVerts[i]);
        }
    }
    
    traces.push({
        type: 'scatter3d',
        mode: 'markers',
        x: humanVerts.map(v => v[0]),
        y: humanVerts.map(v => v[1]),
        z: humanVerts.map(v => v[2]),
        marker: {size: 1, color: 'lightblue', opacity: 0.2},
        hoverinfo: 'skip'
    });
    
    if (humanAnnotatedPts.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: humanAnnotatedPts.map(v => v[0]),
            y: humanAnnotatedPts.map(v => v[1]),
            z: humanAnnotatedPts.map(v => v[2]),
            marker: {size: 2, color: 'red', opacity: 0.8},
            hoverinfo: 'skip'
        });
    }
    
    // Object vertices
    const objVerts = vizData.obj_verts;
    const objAnnotatedPts = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (objectAnnotation[i]) {
            objAnnotatedPts.push(objVerts[i]);
        }
    }
    
    traces.push({
        type: 'scatter3d',
        mode: 'markers',
        x: objVerts.map(v => v[0]),
        y: objVerts.map(v => v[1]),
        z: objVerts.map(v => v[2]),
        marker: {size: 1, color: 'lightgreen', opacity: 0.2},
        hoverinfo: 'skip'
    });
    
    if (objAnnotatedPts.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: objAnnotatedPts.map(v => v[0]),
            y: objAnnotatedPts.map(v => v[1]),
            z: objAnnotatedPts.map(v => v[2]),
            marker: {size: 2, color: 'darkred', opacity: 0.8},
            hoverinfo: 'skip'
        });
    }
    
    Plotly.newPlot('plotCombined', traces, layout, config);
}

function handleBrushClick(data, mesh) {
    if (currentTool !== 'brush') return;
    
    console.log('handleBrushClick called for mesh:', mesh);
    console.log('Click data:', data);
    
    const point = data.points[0];
    const clickPos = [point.x, point.y, point.z];
    
    console.log('Click position:', clickPos);
    
    // Get vertices, faces, and annotation array
    const verts = mesh === 'human' ? vizData.human_verts : vizData.obj_verts;
    const faces = mesh === 'human' ? humanFaces : objectFaces;
    const annotation = mesh === 'human' ? humanAnnotation : objectAnnotation;
    
    // Calculate mesh scale to determine appropriate brush radius
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < verts.length; i++) {
        minX = Math.min(minX, verts[i][0]);
        maxX = Math.max(maxX, verts[i][0]);
        minY = Math.min(minY, verts[i][1]);
        maxY = Math.max(maxY, verts[i][1]);
        minZ = Math.min(minZ, verts[i][2]);
        maxZ = Math.max(maxZ, verts[i][2]);
    }
    const meshScale = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    
    const affectedIndices = new Set();
    // Brush radius relative to mesh scale (brushSize 1-20 maps to 0.5%-10% of mesh scale)
    const radius = (brushSize / 200) * meshScale;
    console.log('Mesh scale:', meshScale, 'Brush radius:', radius);
    
    // Face-based selection: find faces whose center is within brush radius
    if (faces && faces.length > 0) {
        for (let f = 0; f < faces.length; f++) {
            const face = faces[f];
            // Calculate face center
            const centerX = (verts[face[0]][0] + verts[face[1]][0] + verts[face[2]][0]) / 3;
            const centerY = (verts[face[0]][1] + verts[face[1]][1] + verts[face[2]][1]) / 3;
            const centerZ = (verts[face[0]][2] + verts[face[1]][2] + verts[face[2]][2]) / 3;
            
            const dist = Math.sqrt(
                Math.pow(centerX - clickPos[0], 2) +
                Math.pow(centerY - clickPos[1], 2) +
                Math.pow(centerZ - clickPos[2], 2)
            );
            
            if (dist < radius) {
                // Add all vertices of this face
                face.forEach(idx => affectedIndices.add(idx));
            }
        }
        console.log('Face-based selection: found', affectedIndices.size, 'vertices');
    } else {
        // Fallback: vertex-based selection
        for (let i = 0; i < verts.length; i++) {
            const dist = Math.sqrt(
                Math.pow(verts[i][0] - clickPos[0], 2) +
                Math.pow(verts[i][1] - clickPos[1], 2) +
                Math.pow(verts[i][2] - clickPos[2], 2)
            );
            
            if (dist < radius) {
                affectedIndices.add(i);
            }
        }
        console.log('Vertex-based selection: found', affectedIndices.size, 'vertices');
    }
    
    // Apply brush
    const isAdd = brushMode === 'add';
    affectedIndices.forEach(idx => {
        annotation[idx] = isAdd;
    });
    
    console.log('Applied brush, mode:', brushMode, 'affected:', affectedIndices.size);
    
    // Update visualization
    displayVisualization();
    updateStats();
}

function updateStats() {
    const humanCount = humanAnnotation ? humanAnnotation.filter(v => v).length : 0;
    const objCount = objectAnnotation ? objectAnnotation.filter(v => v).length : 0;
    
    document.getElementById('statHumanAnnotated').textContent = humanCount;
    document.getElementById('statObjectAnnotated').textContent = objCount;
    document.getElementById('statTotal').textContent = humanCount + objCount;
}

// Actions
function saveAnnotation() {
    if (!currentTaskId) {
        showMessage('No task loaded', 'error');
        return;
    }
    
    socket.emit('save_annotation', {
        task_id: currentTaskId,
        annotation: {
            human: humanAnnotation,
            object: objectAnnotation
        }
    });
    
    showMessage('Saving annotation...', 'info');
}

function clearAnnotation() {
    if (confirm('Clear all annotations? This cannot be undone.')) {
        if (humanAnnotation) humanAnnotation.fill(false);
        if (objectAnnotation) objectAnnotation.fill(false);
        selectedRegions.clear();
        
        // Clear region UI
        document.querySelectorAll('.region-item').forEach(el => {
            el.classList.remove('selected');
            el.querySelector('.region-checkbox').checked = false;
        });
        
        displayVisualization();
        updateStats();
        showMessage('Annotations cleared', 'info');
    }
}

function goBack() {
    window.location.href = '/viewer';
}

function showMessage(msg, type) {
    const messageArea = document.getElementById('messageArea');
    messageArea.innerHTML = `<div class="message ${type}">${msg}</div>`;
    
    if (type !== 'error') {
        setTimeout(() => {
            messageArea.innerHTML = '';
        }, 3000);
    }
}

// Update target mesh selector
document.getElementById('targetMesh').addEventListener('change', (e) => {
    targetMesh = e.target.value;
    displayVisualization(); // Re-attach event handlers
});

// Load task data on page load
window.addEventListener('load', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const taskId = urlParams.get('task_id');
    
    if (taskId) {
        socket.emit('load_annotation_task', {task_id: parseInt(taskId)});
        showMessage('Loading task data...', 'info');
    } else {
        showMessage('No task ID provided', 'error');
    }
});
