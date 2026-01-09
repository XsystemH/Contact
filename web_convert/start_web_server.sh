#!/bin/bash

# Quick start script for Web-based Data Convert Tool
# Usage: ./start_web_server.sh [port]

PORT=${1:-5000}
HOST="0.0.0.0"

echo "=========================================="
echo "Starting Web-based Data Convert Tool"
echo "=========================================="
echo "Port: $PORT"
echo "Host: $HOST (accessible from LAN)"
echo ""

# Get local IP addresses
echo "Access URLs:"
echo "  Local:   http://localhost:$PORT"
echo "  Network: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
echo "Share the network URL with team members!"
echo "=========================================="
echo ""

# Check if dependencies are installed
python3 -c "import flask, flask_socketio, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Web dependencies not installed!"
    echo "Installing required packages..."
    pip install flask flask-socketio plotly python-socketio
    echo ""
fi

# Start the server
cd "$(dirname "$0")"

# NOTE: Set env vars before running:
#   SOURCE_DIR=/abs/path/to/source TARGET_DIR=/abs/path/to/target OBJECT_DIR=/abs/path/to/objects ./start_web_server.sh
if [ -z "${SOURCE_DIR:-}" ] || [ -z "${TARGET_DIR:-}" ]; then
    echo "Error: Please set SOURCE_DIR and TARGET_DIR env vars."
    echo "Example: SOURCE_DIR=/path/to/data TARGET_DIR=/path/to/data_contact OBJECT_DIR=/path/to/data_object ./start_web_server.sh"
    exit 1
fi

python3 web_interface.py \
    --source_dir "${SOURCE_DIR:-}" \
    --target_dir "${TARGET_DIR:-}" \
    --object_dir "${OBJECT_DIR:-}" \
    --host "$HOST" \
    --port "$PORT"
