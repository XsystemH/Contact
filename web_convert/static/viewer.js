// Socket.IO connection
const socket = io();

// State
let currentTask = null;
let currentTaskId = null;
let currentVizData = null;
let manualAnnotation = null;  // Store manual annotation for current task
let useManualAnnotation = false;  // Whether to use manual annotation instead of calculated

// Elements
const taskCategory = document.getElementById('taskCategory');
const taskPath = document.getElementById('taskPath');
const taskId = document.getElementById('taskId');
const distanceRatio = document.getElementById('distanceRatio');
const contactnetThreshold = document.getElementById('contactnetThreshold');
const btnPreview = document.getElementById('btnPreview');
const btnUseContactNet = document.getElementById('btnUseContactNet');
const btnAccept = document.getElementById('btnAccept');
const btnSkip = document.getElementById('btnSkip');
const btnNext = document.getElementById('btnNext');
const messageArea = document.getElementById('messageArea');
const vizContainer = document.getElementById('vizContainer');
const statsBox = document.getElementById('statsBox');
const refImage = document.getElementById('refImage');

// Header stats
const headerTotal = document.getElementById('headerTotal');
const headerCompleted = document.getElementById('headerCompleted');
const headerUsers = document.getElementById('headerUsers');

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    showMessage('Connected to server', 'success');
});

socket.on('connected', (data) => {
    console.log('User ID:', data.user_id);
});

socket.on('stats_update', (stats) => {
    headerTotal.textContent = stats.total;
    headerCompleted.textContent = stats.completed;
    headerUsers.textContent = stats.active_users;
});

socket.on('task_data', (data) => {
    currentTask = data.task;
    currentTaskId = data.task_id;
    currentVizData = data.viz_data;
    
    displayTask();
    displayVisualization(currentVizData);
    hideMessage();

    // Auto-run ContactNet on each newly loaded dataset (requires local :8000 server)
    if (btnUseContactNet) {
        runContactNet();
    }
});

socket.on('no_tasks', (data) => {
    showMessage('No more tasks available. All tasks completed! 🎉', 'success');
    disableControls();
});

socket.on('visualization_updated', (data) => {
    currentVizData = data.viz_data;
    displayVisualization(currentVizData);
    btnPreview.disabled = false;
    btnPreview.textContent = '🔄 Update Preview';
});

socket.on('contactnet_updated', (data) => {
    if (data.task_id !== currentTaskId) return;
    currentVizData = data.viz_data;

    // Clear manual overlay when switching to ContactNet prediction
    useManualAnnotation = false;
    manualAnnotation = null;
    updateManualAnnotationStatus();

    displayVisualization(currentVizData);
    if (btnUseContactNet) {
        btnUseContactNet.disabled = false;
        btnUseContactNet.textContent = '🤖 Use ContactNet';
    }
    showMessage('Loaded ContactNet prediction for human contact.', 'success');
});

socket.on('decision_accepted', (data) => {
    if (data.success) {
        showMessage('Task processed successfully! Loading next task...', 'success');
        setTimeout(() => {
            requestNextTask();
        }, 1500);
    } else {
        showMessage('Task accepted but processing failed: ' + (data.error || 'Unknown error'), 'error');
        setTimeout(() => {
            requestNextTask();
        }, 2000);
    }
});

socket.on('error', (data) => {
    showMessage('Error: ' + data.message, 'error');
    btnPreview.disabled = false;
    btnPreview.textContent = '🔄 Update Preview';

    if (btnUseContactNet) {
        btnUseContactNet.disabled = false;
        btnUseContactNet.textContent = '🤖 Use ContactNet';
    }
});

// Button handlers
btnNext.addEventListener('click', () => {
    requestNextTask();
});

btnPreview.addEventListener('click', () => {
    updateVisualization();
});

if (btnUseContactNet) {
    btnUseContactNet.addEventListener('click', () => {
        runContactNet();
    });
}

btnAccept.addEventListener('click', () => {
    submitDecision('accept');
});

btnSkip.addEventListener('click', () => {
    submitDecision('skip');
});

// Manual annotate button
const btnManualAnnotate = document.getElementById('btnManualAnnotate');
if (btnManualAnnotate) {
    btnManualAnnotate.addEventListener('click', () => {
        if (currentTaskId !== null) {
            // Seed manual annotation using the currently displayed preview, then open page
            seedManualAnnotationFromCurrentPreview(() => {
                window.open(`/manual_annotate?task_id=${currentTaskId}`, '_blank');
            });
        } else {
            showMessage('No task loaded. Please load a task first.', 'error');
        }
    });
}

function getEffectiveHumanContactMask() {
    if (!currentVizData) return null;

    // If user is currently viewing manual overlay, seed from that (still counts as current preview)
    if (useManualAnnotation && manualAnnotation && currentVizData.human_faces) {
        const mask = new Array(currentVizData.human_contact.length).fill(false);
        const faces = currentVizData.human_faces;
        manualAnnotation.forEach(faceIdx => {
            if (faces[faceIdx]) {
                faces[faceIdx].forEach(vertIdx => {
                    if (vertIdx < mask.length) mask[vertIdx] = true;
                });
            }
        });
        return mask;
    }

    // Otherwise seed from current viz human_contact
    return currentVizData.human_contact ? currentVizData.human_contact.slice() : null;
}

function seedManualAnnotationFromCurrentPreview(onDone) {
    const mask = getEffectiveHumanContactMask();
    if (!mask) {
        showMessage('No preview loaded to seed annotation.', 'error');
        return;
    }

    socket.emit(
        'seed_manual_annotation',
        {
            task_id: currentTaskId,
            contact_mask: mask,
        },
        (resp) => {
            if (resp && resp.success) {
                onDone && onDone();
            } else {
                showMessage('Failed to seed manual annotation: ' + ((resp && resp.error) || 'Unknown error'), 'error');
            }
        }
    );
}

// Listen for manual annotation saved event
socket.on('manual_annotation_received', (data) => {
    if (data.task_id === currentTaskId) {
        manualAnnotation = data.selected_faces;
        useManualAnnotation = true;
        showMessage(`Manual annotation received! ${manualAnnotation.length} faces selected. Click Accept to save, or Update Preview to recalculate.`, 'success');
        
        // Update visualization to show manual annotation
        if (currentVizData) {
            displayVisualizationWithManualAnnotation();
        }
        
        // Update button style to indicate manual annotation is active
        updateManualAnnotationStatus();
    }
});

function updateManualAnnotationStatus() {
    const statusIndicator = document.getElementById('manualAnnotationStatus');
    if (statusIndicator) {
        if (useManualAnnotation && manualAnnotation) {
            statusIndicator.style.display = 'block';
            statusIndicator.textContent = `📌 Manual: ${manualAnnotation.length} faces`;
        } else {
            statusIndicator.style.display = 'none';
        }
    }
}

function displayVisualizationWithManualAnnotation() {
    if (!currentVizData) return;
    
    // Create modified viz data with manual annotation
    const modifiedVizData = JSON.parse(JSON.stringify(currentVizData));
    
    // Reset human contact arrays
    for (let i = 0; i < modifiedVizData.human_contact.length; i++) {
        modifiedVizData.human_contact[i] = false;
        modifiedVizData.human_interior[i] = false;
        modifiedVizData.human_proximity[i] = false;
    }
    
    // Get vertex indices from face indices
    if (manualAnnotation && currentVizData.human_faces) {
        const faces = currentVizData.human_faces;
        manualAnnotation.forEach(faceIdx => {
            if (faces[faceIdx]) {
                faces[faceIdx].forEach(vertIdx => {
                    if (vertIdx < modifiedVizData.human_contact.length) {
                        modifiedVizData.human_contact[vertIdx] = true;
                        modifiedVizData.human_interior[vertIdx] = true;
                    }
                });
            }
        });
    }
    
    // Update stats
    const contactCount = modifiedVizData.human_contact.filter(x => x).length;
    modifiedVizData.stats.human_total = contactCount;
    modifiedVizData.stats.total_contact = contactCount + modifiedVizData.stats.obj_total;
    modifiedVizData.stats.human_interior = contactCount;
    modifiedVizData.stats.human_proximity = 0;
    modifiedVizData.stats.human_both = 0;
    
    displayVisualization(modifiedVizData);
}

distanceRatio.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        updateVisualization();
    }
});

if (contactnetThreshold) {
    contactnetThreshold.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            runContactNet();
        }
    });
}

// Functions
function requestNextTask() {
    showMessage('Requesting next task...', 'info');
    
    // Reset manual annotation state for new task
    manualAnnotation = null;
    useManualAnnotation = false;
    updateManualAnnotationStatus();
    
    disableControls();
    socket.emit('request_task');
}

function updateVisualization() {
    const ratio = parseFloat(distanceRatio.value);
    if (isNaN(ratio) || ratio < 0.01 || ratio > 3.0) {
        showMessage('Distance ratio must be between 0.01 and 3.0', 'error');
        return;
    }
    
    // Clear manual annotation when recalculating
    useManualAnnotation = false;
    updateManualAnnotationStatus();
    
    btnPreview.disabled = true;
    btnPreview.textContent = '⏳ Updating...';
    
    socket.emit('update_visualization', {
        task_id: currentTaskId,
        distance_ratio: ratio
    });
}

function runContactNet() {
    if (currentTaskId === null) {
        showMessage('No task loaded. Please load a task first.', 'error');
        return;
    }

    let threshold = 0.5;
    if (contactnetThreshold) {
        threshold = parseFloat(contactnetThreshold.value);
        if (isNaN(threshold) || threshold < 0.0 || threshold > 1.0) {
            showMessage('ContactNet threshold must be between 0.0 and 1.0', 'error');
            return;
        }
    }

    // Clear manual annotation when switching to ContactNet
    useManualAnnotation = false;
    manualAnnotation = null;
    updateManualAnnotationStatus();

    if (btnUseContactNet) {
        btnUseContactNet.disabled = true;
        btnUseContactNet.textContent = '⏳ Running ContactNet...';
    }
    showMessage('Calling ContactNet server on port 8000...', 'info');

    socket.emit('run_contactnet', {
        task_id: currentTaskId,
        threshold: threshold,
    });
}

function submitDecision(decision) {
    const ratio = parseFloat(distanceRatio.value);
    
    disableControls();
    showMessage(`Submitting decision: ${decision}...`, 'info');
    
    const payload = {
        task_id: currentTaskId,
        decision: decision,
        distance_ratio: ratio
    };
    
    // Include manual annotation if active
    if (useManualAnnotation && manualAnnotation) {
        payload.manual_annotation = manualAnnotation;
    }
    
    socket.emit('submit_decision', payload);
}

function displayTask() {
    taskCategory.textContent = currentTask.category;
    taskPath.textContent = currentTask.relative_path;
    taskId.textContent = currentTaskId;
    
    enableControls();
}

function displayVisualization(vizData) {
    vizContainer.style.display = 'grid';
    statsBox.style.display = 'block';
    
    // Display statistics
    document.getElementById('statHumanTotal').textContent = vizData.stats.human_total;
    document.getElementById('statObjTotal').textContent = vizData.stats.obj_total;
    document.getElementById('statTotal').textContent = vizData.stats.total_contact;
    document.getElementById('statHumanInterior').textContent = vizData.stats.human_interior;
    document.getElementById('statHumanProximity').textContent = vizData.stats.human_proximity;
    document.getElementById('statHumanBoth').textContent = vizData.stats.human_both;
    
    // Display reference image
    if (vizData.reference_image) {
        refImage.src = 'data:image/jpeg;base64,' + vizData.reference_image;
    }
    
    // Camera settings
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
            zaxis: {showgrid: false, zeroline: false, showticklabels: false}
        },
        margin: {l: 0, r: 0, t: 0, b: 0},
        showlegend: false
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    // Plot human mesh
    plotHumanMesh(vizData, layout, config);
    
    // Plot object mesh
    plotObjectMesh(vizData, layout, config);
    
    // Plot combined view
    plotCombinedView(vizData, layout, config);
}

function plotHumanMesh(vizData, layout, config) {
    const humanVerts = vizData.human_verts;
    const humanContact = vizData.human_contact;
    const humanInterior = vizData.human_interior;
    const humanProximity = vizData.human_proximity;
    
    const traces = [];
    
    // Non-contact vertices
    const nonContact = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (!humanContact[i]) {
            nonContact.push(humanVerts[i]);
        }
    }
    if (nonContact.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: nonContact.map(v => v[0]),
            y: nonContact.map(v => v[1]),
            z: nonContact.map(v => v[2]),
            marker: {size: 1, color: 'lightblue', opacity: 0.3},
            name: 'Non-contact'
        });
    }
    
    // Interior only
    const interiorOnly = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (humanInterior[i] && !humanProximity[i]) {
            interiorOnly.push(humanVerts[i]);
        }
    }
    if (interiorOnly.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: interiorOnly.map(v => v[0]),
            y: interiorOnly.map(v => v[1]),
            z: interiorOnly.map(v => v[2]),
            marker: {size: 1.1, color: 'orange'},
            name: 'Interior'
        });
    }
    
    // Proximity only
    const proximityOnly = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (humanProximity[i] && !humanInterior[i]) {
            proximityOnly.push(humanVerts[i]);
        }
    }
    if (proximityOnly.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: proximityOnly.map(v => v[0]),
            y: proximityOnly.map(v => v[1]),
            z: proximityOnly.map(v => v[2]),
            marker: {size: 1.1, color: 'yellow'},
            name: 'Proximity'
        });
    }
    
    // Both
    const both = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (humanInterior[i] && humanProximity[i]) {
            both.push(humanVerts[i]);
        }
    }
    if (both.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: both.map(v => v[0]),
            y: both.map(v => v[1]),
            z: both.map(v => v[2]),
            marker: {size: 1.1, color: 'red'},
            name: 'Both'
        });
    }
    
    Plotly.newPlot('plotHuman', traces, layout, config);
}

function plotObjectMesh(vizData, layout, config) {
    const objVerts = vizData.obj_verts;
    const objContact = vizData.obj_contact;
    const objInterior = vizData.obj_interior;
    const objProximity = vizData.obj_proximity;
    
    const traces = [];
    
    // Non-contact vertices
    const nonContact = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (!objContact[i]) {
            nonContact.push(objVerts[i]);
        }
    }
    if (nonContact.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: nonContact.map(v => v[0]),
            y: nonContact.map(v => v[1]),
            z: nonContact.map(v => v[2]),
            marker: {size: 1, color: 'lightgreen', opacity: 0.3},
            name: 'Non-contact'
        });
    }
    
    // Interior only
    const interiorOnly = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (objInterior[i] && !objProximity[i]) {
            interiorOnly.push(objVerts[i]);
        }
    }
    if (interiorOnly.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: interiorOnly.map(v => v[0]),
            y: interiorOnly.map(v => v[1]),
            z: interiorOnly.map(v => v[2]),
            marker: {size: 1.1, color: 'orange'},
            name: 'Interior'
        });
    }
    
    // Proximity only
    const proximityOnly = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (objProximity[i] && !objInterior[i]) {
            proximityOnly.push(objVerts[i]);
        }
    }
    if (proximityOnly.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: proximityOnly.map(v => v[0]),
            y: proximityOnly.map(v => v[1]),
            z: proximityOnly.map(v => v[2]),
            marker: {size: 1.1, color: 'yellow'},
            name: 'Proximity'
        });
    }
    
    // Both
    const both = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (objInterior[i] && objProximity[i]) {
            both.push(objVerts[i]);
        }
    }
    if (both.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: both.map(v => v[0]),
            y: both.map(v => v[1]),
            z: both.map(v => v[2]),
            marker: {size: 1.1, color: 'red'},
            name: 'Both'
        });
    }
    
    Plotly.newPlot('plotObject', traces, layout, config);
}

function plotCombinedView(vizData, layout, config) {
    const humanVerts = vizData.human_verts;
    const objVerts = vizData.obj_verts;
    const humanContact = vizData.human_contact;
    const objContact = vizData.obj_contact;
    
    const traces = [];
    
    // Human non-contact
    const humanNonContact = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (!humanContact[i]) {
            humanNonContact.push(humanVerts[i]);
        }
    }
    if (humanNonContact.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: humanNonContact.map(v => v[0]),
            y: humanNonContact.map(v => v[1]),
            z: humanNonContact.map(v => v[2]),
            marker: {size: 1, color: 'lightblue', opacity: 0.2},
            name: 'Human'
        });
    }
    
    // Object non-contact
    const objNonContact = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (!objContact[i]) {
            objNonContact.push(objVerts[i]);
        }
    }
    if (objNonContact.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: objNonContact.map(v => v[0]),
            y: objNonContact.map(v => v[1]),
            z: objNonContact.map(v => v[2]),
            marker: {size: 1, color: 'lightgreen', opacity: 0.2},
            name: 'Object'
        });
    }
    
    // Human contact
    const humanContactPts = [];
    for (let i = 0; i < humanVerts.length; i++) {
        if (humanContact[i]) {
            humanContactPts.push(humanVerts[i]);
        }
    }
    if (humanContactPts.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: humanContactPts.map(v => v[0]),
            y: humanContactPts.map(v => v[1]),
            z: humanContactPts.map(v => v[2]),
            marker: {size: 1.1, color: 'red', opacity: 0.6},
            name: 'H-Contact'
        });
    }
    
    // Object contact
    const objContactPts = [];
    for (let i = 0; i < objVerts.length; i++) {
        if (objContact[i]) {
            objContactPts.push(objVerts[i]);
        }
    }
    if (objContactPts.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: objContactPts.map(v => v[0]),
            y: objContactPts.map(v => v[1]),
            z: objContactPts.map(v => v[2]),
            marker: {size: 1.1, color: 'darkred', opacity: 0.6},
            name: 'O-Contact'
        });
    }
    
    Plotly.newPlot('plotCombined', traces, layout, config);
}

function showMessage(msg, type) {
    const className = type === 'error' ? 'error' : (type === 'success' ? 'success' : 'info');
    messageArea.innerHTML = `<div class="${className}">${msg}</div>`;
}

function hideMessage() {
    messageArea.innerHTML = '';
}

function disableControls() {
    btnPreview.disabled = true;
    if (btnUseContactNet) btnUseContactNet.disabled = true;
    btnAccept.disabled = true;
    btnSkip.disabled = true;
    distanceRatio.disabled = true;
}

function enableControls() {
    btnPreview.disabled = false;
    if (btnUseContactNet) btnUseContactNet.disabled = false;
    btnAccept.disabled = false;
    btnSkip.disabled = false;
    distanceRatio.disabled = false;
}

// Auto-request first task on load
window.addEventListener('load', () => {
    setTimeout(() => {
        requestNextTask();
    }, 500);
});
