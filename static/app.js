document.addEventListener('DOMContentLoaded', () => {
    // Initialize Map
    const map = L.map('map').setView([45.767328, 4.833362], 13);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
    
    // State
    let clusterLayers = {}; // id -> L.polygon
    let ws = null;
    let eventsData = [];
    
    // Initialize WebSocket
    function connectWS() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log("WebSocket connected");
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "label_update") {
                updateLabel(data.cluster_id, data.label);
            }
        };
        
        ws.onclose = () => {
            console.log("WebSocket closed. Reconnecting in 3s...");
            setTimeout(connectWS, 3000);
        };
    }
    
    connectWS();
    
    // Load Stats & Events
    fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                document.getElementById('stats-display').innerText = "Error loading data.";
                return;
            }
            
            document.getElementById('stats-display').innerHTML = `
                Total Points: <strong>${data.total_points}</strong><br>
                Range: ${data.min_date.split('T')[0]} to ${data.max_date.split('T')[0]}
            `;
            
            // Set default years
            document.getElementById('year-min').value = parseInt(data.min_date.substring(0,4));
            document.getElementById('year-max').value = parseInt(data.max_date.substring(0,4));
            
            // Populate Events
            const eventsList = document.getElementById('events-list');
            eventsData = data.events;
            
            eventsData.forEach(e => {
                const div = document.createElement('div');
                div.className = 'event-item';
                div.innerHTML = `
                    <input type="checkbox" id="evt-${e.id}" value="${e.id}">
                    <label for="evt-${e.id}" title="${e.start} to ${e.end}">
                        ${e.label} (${e.count})
                    </label>
                `;
                eventsList.appendChild(div);
            });
        });
        
    // Initialize Filter Toggles
    const yearAllCb = document.getElementById('year-all');
    const yearInputs = document.getElementById('year-inputs');
    yearAllCb.addEventListener('change', (e) => {
        if (e.target.checked) {
            yearInputs.style.opacity = '0.5';
            yearInputs.style.pointerEvents = 'none';
        } else {
            yearInputs.style.opacity = '1';
            yearInputs.style.pointerEvents = 'auto';
        }
    });

    const hourAllCb = document.getElementById('hour-all');
    const hourInputs = document.getElementById('hour-inputs');
    hourAllCb.addEventListener('change', (e) => {
        if (e.target.checked) {
            hourInputs.style.opacity = '0.5';
            hourInputs.style.pointerEvents = 'none';
        } else {
            hourInputs.style.opacity = '1';
            hourInputs.style.pointerEvents = 'auto';
        }
    });

    const eventsAllCb = document.getElementById('events-all');
    eventsAllCb.addEventListener('change', (e) => {
        const checkboxes = document.querySelectorAll('#events-list input[type="checkbox"]');
        checkboxes.forEach(cb => cb.checked = e.target.checked);
    });

    // Run Clustering
    document.getElementById('run-btn').addEventListener('click', async () => {
        const btn = document.getElementById('run-btn');
        btn.disabled = true;
        btn.innerText = "Processing...";
        
        // Collect Params
        const params = {};
        
        if (!yearAllCb.checked) {
            params.min_year = parseInt(document.getElementById('year-min').value);
            params.max_year = parseInt(document.getElementById('year-max').value);
        }
        
        if (!hourAllCb.checked) {
            params.start_hour = parseInt(document.getElementById('hour-start').value);
            params.end_hour = parseInt(document.getElementById('hour-end').value);
        }
        
        const excludeEvents = [];
        document.querySelectorAll('#events-list input:checked').forEach(cb => {
            excludeEvents.push(parseInt(cb.value));
        });
        params.exclude_events = excludeEvents;
        
        try {
            const resp = await fetch('/api/cluster', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            });
            const result = await resp.json();
            
            renderClusters(result.clusters);
            
        } catch (e) {
            console.error("Cluster error", e);
            alert("Error running clustering");
        } finally {
            btn.disabled = false;
            btn.innerText = "Run Clustering";
        }
    });
    
    function renderClusters(clusters) {
        // Clear existing
        Object.values(clusterLayers).forEach(l => map.removeLayer(l));
        clusterLayers = {};
        
        if (clusters.length === 0) {
            alert("No clusters found with current filters.");
            return;
        }
        
        const bounds = new L.LatLngBounds();
        
        clusters.forEach(c => {
            const color = getRandomColor();
            // c.points is [[lat, lon], ...]
            const polygon = L.polygon(c.points, {
                color: color,
                fillColor: color,
                fillOpacity: 0.4,
                weight: 1
            }).addTo(map);
            
            polygon.bindTooltip(`Cluster ${c.id} (Loading...)`, {
                permanent: true, 
                direction: 'center',
                className: 'cluster-label'
            });
            
            polygon.on('click', () => {
                prioritizeCluster(c.id);
                document.getElementById('cluster-info').innerHTML = `
                    <h3>Cluster ${c.id}</h3>
                    <p>Size: ${c.size} points</p>
                    <p>Label: <strong>Loading...</strong></p>
                `;
            });
            
            // Store for updates
            // Add custom property to layer to store label
            polygon._clusterId = c.id;
            polygon._currentLabel = "Loading...";
            
             // Expand bounds
            c.points.forEach(p => bounds.extend(p));
            
            clusterLayers[c.id] = polygon;
        });
        
        map.fitBounds(bounds);
    }
    
    function updateLabel(clusterId, label) {
        const poly = clusterLayers[clusterId];
        if (poly) {
            poly._currentLabel = label;
            poly.setTooltipContent(label);
            
            // If this is the currently selected cluster info
            const infoDiv = document.getElementById('cluster-info');
            if (infoDiv.innerHTML.includes(`Cluster ${clusterId}`)) {
                 // lazy update using regex or DOM assumption
                 // Easier: just let the user click again or use data binding
                 // But let's try to update text
                 const b = infoDiv.querySelector('strong');
                 if(b) b.innerText = label;
            }
        }
    }
    
    function prioritizeCluster(clusterId) {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({
            action: "prioritize",
            cluster_id: clusterId
        }));
    }
    
    function getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }
});
