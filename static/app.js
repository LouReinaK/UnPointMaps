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
  let pendingLabels = {}; // id -> label (for race conditions)
    
  // Carousel State
  let currentClusterId = null;
  let currentImageIndex = 0;
  let clusterImages = [];
    
  // Initialize WebSocket
  function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
        
    ws = new WebSocket(wsUrl);
        
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
        
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WS Message:', data.type, data);
                
        if (data.type === 'label_update') {
          updateLabel(data.cluster_id, data.label);
        } else if (data.type === 'cluster_update') {
          renderClusters(data.clusters, true); // true = silent (no alert if empty)
        } else if (data.type === 'progress') {
          // ... existing progress logic ...
          const statusDiv = document.getElementById('status-display');
          const statusMsg = document.getElementById('status-message');
          const iterDiv = document.getElementById('iteration-display');
                    
          if (statusDiv && statusMsg) {
            statusDiv.style.display = 'block';
            statusMsg.innerText = data.message;
                        
            if (iterDiv) {
              if (data.iteration) {
                iterDiv.innerText = `Current Iteration: ${data.iteration}`;
                iterDiv.style.display = 'block';
              } else if (data.message.includes('Clustering complete')) {
                iterDiv.innerText = 'Final Iteration Reached';
              }
            }
          }
        }
      } catch (e) {
        console.error('WS Error parsing message:', e);
      }
    };
        
    ws.onclose = () => {
      console.log('WebSocket closed. Reconnecting in 3s...');
      setTimeout(connectWS, 3000);
    };
  }
    
  connectWS();
    
  // Load Stats & Events
  fetch('/api/stats')
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        document.getElementById('stats-display').innerText = 'Error loading data.';
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

  const dateAllCb = document.getElementById('date-all');
  const dateInputs = document.getElementById('date-inputs');
  dateAllCb.addEventListener('change', (e) => {
    if (e.target.checked) {
      dateInputs.style.opacity = '0.5';
      dateInputs.style.pointerEvents = 'none';
    } else {
      dateInputs.style.opacity = '1';
      dateInputs.style.pointerEvents = 'auto';
    }
  });

  const eventsAllCb = document.getElementById('events-all');
  eventsAllCb.addEventListener('change', (e) => {
    const checkboxes = document.querySelectorAll('#events-list input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = e.target.checked);
  });
  
  // Logic for Event Mode Select
  const eventModeSelect = document.getElementById('event-mode-select');

  // Run Clustering
  document.getElementById('run-btn').addEventListener('click', async () => {
    console.time('Frontend Clustering Process');
    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    btn.innerText = 'Processing...';
        
    // Show status panel immediately
    const statusDiv = document.getElementById('status-display');
    const statusMsg = document.getElementById('status-message');
    if (statusDiv && statusMsg) {
      statusDiv.style.display = 'block';
      statusMsg.innerText = 'Starting clustering...';
    }
        
    // Collect Params
    console.time('Collect Params');
    const params = {};
        
    if (!yearAllCb.checked) {
      params.min_year = parseInt(document.getElementById('year-min').value);
      params.max_year = parseInt(document.getElementById('year-max').value);
    }
        
    if (!dateAllCb.checked) {
      params.start_date = document.getElementById('date-start').value;
      params.end_date = document.getElementById('date-end').value;
    }
        
    if (!hourAllCb.checked) {
      params.start_hour = parseInt(document.getElementById('hour-start').value);
      params.end_hour = parseInt(document.getElementById('hour-end').value);
    }

    // Algo
    const algoSelect = document.getElementById('algorithm-select');
    if (algoSelect) {
      params.algorithm = algoSelect.value;
    }

    // Labelling
    const labelingSelect = document.getElementById('labelling-select');
    if (labelingSelect) {
      params.labelling_method = labelingSelect.value;
    }
        
    const selectedEvents = [];
    document.querySelectorAll('#events-list input:checked').forEach(cb => {
      selectedEvents.push(parseInt(cb.value));
    });

    // Check mode
    const mode = eventModeSelect ? eventModeSelect.value : 'exclude';
    if (mode === 'exclude') {
      params.exclude_events = selectedEvents;
    } else {
      params.include_events = selectedEvents;
    }
    console.timeEnd('Collect Params');
        
    try {
      console.time('API Call');
      const resp = await fetch('/api/cluster', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
      });
      const result = await resp.json();
      console.timeEnd('API Call');
            
      if (result.status === 'started') {
        console.log('Clustering started in background...');
        // Optionally clear map to indicate start
        renderClusters([], true);
      } else {
        console.time('Render Clusters');
        renderClusters(result.clusters);
        console.timeEnd('Render Clusters');
      }
            
    } catch (e) {
      console.error('Cluster error', e);
      alert('Error running clustering');
    } finally {
      btn.disabled = false;
      btn.innerText = 'Run Clustering';
      console.timeEnd('Frontend Clustering Process');
      // Note: Sort frontend tasks manually from console logs
    }
  });
    
  function renderClusters(clusters, silent = false) {
    console.time('Render Clusters Internal');
    // Clear existing
    Object.values(clusterLayers).forEach(l => map.removeLayer(l));
    clusterLayers = {};
        
    if (clusters.length === 0) {
      if (!silent) alert('No clusters found with current filters.');
      console.timeEnd('Render Clusters Internal');
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
            

            
      polygon.on('click', () => {
        prioritizeCluster(c.id);
        displayClusterInfo(c.id, c.size);
        fetchClusterImages(c.id);
      });
            
      // Store for updates
      // Add custom property to layer to store label
      polygon._clusterId = c.id;
            
      // Check pending labels
      // if (pendingLabels[c.id]) {
      //   polygon._currentLabel = pendingLabels[c.id];
      //   polygon.setTooltipContent(pendingLabels[c.id]);
      //   // Remove from pending? Maybe keep it for safety or just depend on layer property
      // }
            
      // Expand bounds
      c.points.forEach(p => bounds.extend(p));
            
      clusterLayers[c.id] = polygon;
    });
        
    // Cleanup old pending labels that didn't match (optional)
    // pendingLabels = {}; 
        
    map.fitBounds(bounds);
    console.timeEnd('Render Clusters Internal');
  }
    
  function updateLabel(clusterId, label) {
    console.log('Updating label for', clusterId, ':', label);
    // Normalize ID to string for lookup
    const poly = clusterLayers[clusterId];
        
    if (poly) {
      poly._currentLabel = label;
      // Removed tooltip update
      // poly.setTooltipContent(label);
            
      // Normalize for comparison
      if (String(currentClusterId) === String(clusterId)) {
        console.log('Updating displayed info for', clusterId);
        const infoContent = document.getElementById('cluster-info-content');
        // We need to match the specific DOM structure created in displayClusterInfo
        // Note: displayClusterInfo creates <p>Label: <strong>...</strong></p>
        const labelElement = infoContent.querySelector('p strong'); 
        if (labelElement) {
          labelElement.innerText = label;
        } else {
          console.warn('Label element not found in DOM');
        }
        // Refetch images in case metadata was updated
        fetchClusterImages(clusterId);
      }
    } else {
      console.warn('Cluster layer not found for id:', clusterId, ' - queuing as pending.');
      pendingLabels[clusterId] = label;
    }
  }
    
  function prioritizeCluster(clusterId) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      action: 'prioritize',
      cluster_id: clusterId
    }));
  }
    
  function displayClusterInfo(clusterId, size) {
    currentClusterId = clusterId;
    currentImageIndex = 0;
        
    const infoContent = document.getElementById('cluster-info-content');
    const carouselContainer = document.getElementById('cluster-carousel-container');
        
    // Ensure we handle non-existent layer gracefully
    const label = (clusterLayers[clusterId] && clusterLayers[clusterId]._currentLabel) || 'Loading...';
        
    infoContent.innerHTML = `
            <h3>Cluster ${clusterId}</h3>
            <p>Size: ${size} points</p>
            <p>Label: <strong>${label}</strong></p>
        `;
        
    // Show loading state for carousel
    carouselContainer.style.display = 'block';
    document.getElementById('carousel-counter').textContent = 'Loading images...';
    document.getElementById('thumbnail-container').innerHTML = '';
    document.getElementById('carousel-image').src = '';
    document.getElementById('carousel-image').alt = 'Loading...';
  }
    
  async function fetchClusterImages(clusterId) {
    try {
      const response = await fetch(`/api/cluster_images?cluster_id=${clusterId}`);
      const data = await response.json();
            
      if (data.images && data.images.length > 0) {
        clusterImages = data.images;
        displayCarousel();
      } else {
        document.getElementById('carousel-counter').textContent = 'No images available for this cluster';
        document.getElementById('thumbnail-container').innerHTML = '';
      }
    } catch (error) {
      console.error('Error fetching cluster images:', error);
      document.getElementById('carousel-counter').textContent = 'Error loading images';
    }
  }
    
  function displayCarousel() {
    if (clusterImages.length === 0) return;
        
    const imageElement = document.getElementById('carousel-image');
    const counterElement = document.getElementById('carousel-counter');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const thumbnailContainer = document.getElementById('thumbnail-container');
        
    const currentImg = clusterImages[currentImageIndex];
        
    // Update main image
    imageElement.src = currentImg.url;
    imageElement.alt = `Cluster ${currentClusterId} image ${currentImageIndex + 1}`;
        
    // Add link behavior
    if (currentImg.page_url) {
      imageElement.style.cursor = 'pointer';
      imageElement.onclick = () => window.open(currentImg.page_url, '_blank');
      imageElement.title = 'Click to open on Flickr';
    } else {
      imageElement.style.cursor = 'default';
      imageElement.onclick = null;
      imageElement.removeAttribute('title');
    }
        
    // Update counter
    counterElement.textContent = `${currentImageIndex + 1} / ${clusterImages.length}`;
        
    // Update button states
    prevBtn.disabled = currentImageIndex === 0;
    nextBtn.disabled = currentImageIndex === clusterImages.length - 1;
        
    // Generate thumbnails
    thumbnailContainer.innerHTML = '';
    clusterImages.forEach((imgObj, index) => {
      const thumb = document.createElement('img');
      thumb.src = imgObj.url;
      thumb.className = 'thumbnail';
      if (index === currentImageIndex) {
        thumb.classList.add('active');
      }
      thumb.addEventListener('click', () => {
        currentImageIndex = index;
        displayCarousel();
      });
      thumbnailContainer.appendChild(thumb);
    });
        
    // Add event listeners for navigation buttons
    prevBtn.onclick = () => {
      if (currentImageIndex > 0) {
        currentImageIndex--;
        displayCarousel();
      }
    };
        
    nextBtn.onclick = () => {
      if (currentImageIndex < clusterImages.length - 1) {
        currentImageIndex++;
        displayCarousel();
      }
    };
        
    // Add keyboard navigation
    document.addEventListener('keydown', handleKeyboardNavigation);
  }
    
  function handleKeyboardNavigation(e) {
    if (!currentClusterId) return;
        
    if (e.key === 'ArrowLeft') {
      if (currentImageIndex > 0) {
        currentImageIndex--;
        displayCarousel();
      }
    } else if (e.key === 'ArrowRight') {
      if (currentImageIndex < clusterImages.length - 1) {
        currentImageIndex++;
        displayCarousel();
      }
    }
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
