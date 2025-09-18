document.addEventListener('DOMContentLoaded', () => {
    const fetchInterval = 3000; // Fetch data every 3 seconds
    let vehicleCountChart;

    function updateUI(data) {
        const { lanes, system_config, analytics, events } = data;

        // Update System Summary
        const totalVehicles = Object.values(lanes).reduce((acc, lane) => acc + lane.total_vehicles_handled, 0);
        const avgCompliance = analytics.total_processing_cycles > 0
            ? (Object.values(lanes).reduce((acc, lane) => acc + calculateEfficiency(lane), 0) / 4).toFixed(0)
            : 0;
        const totalEvents = events.length;
        document.getElementById('total-vehicles').textContent = totalVehicles;
        document.getElementById('avg-compliance').textContent = `${avgCompliance}%`;
        document.getElementById('total-events').textContent = totalEvents;

        // Update Video Feeds and Overlays
        Object.values(lanes).forEach(lane => {
            const videoBox = document.querySelector(`.video-box[data-lane-id="${lane.name.toLowerCase().replace(' ', '-')}"]`);
            if (videoBox) {
                const signalStatusSpan = videoBox.querySelector('.signal-status');
                signalStatusSpan.textContent = lane.signal_status;
                signalStatusSpan.className = `signal-status ${lane.signal_status}`;
                videoBox.dataset.signal = lane.signal_status;
            }
        });

        // Update Vehicle Count Chart
        updateVehicleCountChart(lanes);

        // Update Event Log
        const eventLog = document.getElementById('event-log');
        eventLog.innerHTML = '';
        events.slice(-5).reverse().forEach(event => {
            const li = document.createElement('li');
            const time = new Date(event.timestamp * 1000).toLocaleTimeString();
            li.innerHTML = `<span class="event-type">${event.type}</span><span class="event-time">${time}</span>`;
            eventLog.appendChild(li);
        });

        // Update Auto Mode toggle
        const autoModeToggle = document.getElementById('auto-mode-toggle');
        autoModeToggle.checked = system_config.auto_control;

        // Handle emergency mode UI
        if (system_config.emergency_override) {
             document.getElementById('system-status').textContent = 'EMERGENCY';
             document.getElementById('system-status').className = 'status-indicator emergency';
        } else {
             document.getElementById('system-status').textContent = 'System Online';
             document.getElementById('system-status').className = 'status-indicator online';
        }
    }

    function calculateEfficiency(laneData) {
        let efficiency = 50;
        const { traffic_density, vehicle_count, signal_status, throughput_rate } = laneData;

        if (signal_status === 'GREEN' && vehicle_count > 10) efficiency += 30;
        else if (signal_status === 'RED' && vehicle_count < 5) efficiency += 20;
        else if (signal_status === 'GREEN' && vehicle_count < 3) efficiency -= 20;
        else if (signal_status === 'RED' && vehicle_count > 15) efficiency -= 25;

        efficiency += Math.min(throughput_rate * 2, 20);
        if (traffic_density >= 0.3 && traffic_density <= 0.7) efficiency += 10;
        else if (traffic_density > 0.8) efficiency -= 15;

        return Math.max(0, Math.min(100, efficiency));
    }

    function updateVehicleCountChart(lanes) {
        const labels = Object.values(lanes).map(lane => lane.name);
        const data = Object.values(lanes).map(lane => lane.vehicle_count);

        if (vehicleCountChart) {
            vehicleCountChart.data.labels = labels;
            vehicleCountChart.data.datasets[0].data = data;
            vehicleCountChart.update();
        } else {
            const ctx = document.getElementById('vehicle-count-chart').getContext('2d');
            vehicleCountChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Current Vehicle Count',
                        data: data,
                        backgroundColor: [
                            '#00ffff', '#ffa500', '#ff00ff', '#00ff00'
                        ],
                        borderColor: [
                            '#00ffff', '#ffa500', '#ff00ff', '#00ff00'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#a0a0a0' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#a0a0a0' }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
    }

    async function fetchTrafficData() {
        try {
            const response = await fetch('/api/traffic_data');
            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error('Failed to fetch traffic data:', error);
        }
    }

    // Event Listeners for Manual Controls
    document.querySelectorAll('.manual-button').forEach(button => {
        button.addEventListener('click', async () => {
            const laneId = button.dataset.lane;
            const signalStatus = button.dataset.signal;
            try {
                const response = await fetch('/api/control_signal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lane_id: laneId, signal_status: signalStatus })
                });
                const result = await response.json();
                if (result.success) {
                    fetchTrafficData();
                } else {
                    console.error('Failed to update signal:', result.error);
                }
            } catch (error) {
                console.error('Error sending control request:', error);
            }
        });
    });

    document.getElementById('emergency-stop-button').addEventListener('click', async () => {
        try {
            const response = await fetch('/api/emergency_stop', { method: 'POST' });
            const result = await response.json();
            if (result.success) {
                fetchTrafficData();
            }
        } catch (error) {
            console.error('Error activating emergency stop:', error);
        }
    });

    document.getElementById('auto-mode-toggle').addEventListener('change', async (event) => {
        const autoControl = event.target.checked;
        try {
            const response = await fetch('/api/set_auto_control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ auto_control: autoControl })
            });
            const result = await response.json();
            if (result.success) {
                fetchTrafficData();
            }
        } catch (error) {
            console.error('Error toggling auto control:', error);
        }
    });

    window.resetSystem = async () => {
        try {
            const response = await fetch('/api/reset_system', { method: 'POST' });
            const result = await response.json();
            if (result.success) {
                fetchTrafficData();
            }
        } catch (error) {
            console.error('Error resetting system:', error);
        }
    };

    // Initial fetch and start polling
    fetchTrafficData();
    setInterval(fetchTrafficData, fetchInterval);
});
