import queue
import threading
import time
import requests
from flask import Flask, request, jsonify
import logging
import uuid
from datetime import datetime, timedelta
import configparser
import os, sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dispatcher:
    def __init__(self):
        # Load configuration
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'service.cfg')
        try:
            config.read(config_path)
            service_config = config['service']
            self.endpoint_url = f"{service_config['protocol']}://{service_config['minikube_ip']}:{service_config['node_port']}"
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
        
        # Request queue
        self.request_queue = queue.Queue(maxsize=200)
        
        # Result storage for async requests
        self.results = {}  # {request_id: {status, result, timestamp}}
        
        # Current replica index for round-robin
        self.current_replica = 0
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # New metrics
        self.start_time = time.time()
        self.peak_queue_size = 0
        self.total_processing_time = 0
        self.total_queue_time = 0
        self.last_request_time = None
        self.queue_capacity = 200  # matches maxsize from queue initialization
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        logger.info("Dispatcher initialized")
    
    def _process_queue(self):
        """Background worker to process queued requests"""
        while True:
            try:
                # Blocking request
                request_data = self.request_queue.get(timeout=1)
                self._forward_request(request_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
    
    def _forward_request(self, request_data):
        """Forward request to a replica and store result"""
        request_id = request_data['request_id']        
        
        self.results[request_id]['status'] = 'processing'
        self.results[request_id]['replica_used'] = self.endpoint_url
        
        try:
            # Calculate queue time
            queue_time = time.time() - request_data['timestamp']
            self.total_queue_time += queue_time
            
            # Forward the image to ML Server
            files = {'image': request_data['image_data']}
            start_time = time.time()
            response = requests.post(f"{self.endpoint_url}/predict", files=files, timeout=30)
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            if response.status_code == 200:
                self.successful_requests += 1
                
                self.results[request_id].update({
                    'status': 'completed',
                    'result': response.json(),
                    'processing_time': processing_time,
                    'completed_at': datetime.now().isoformat()
                })
                logger.info(f"Request {request_id} processed successfully by {self.endpoint_url}")
            else:
                self.failed_requests += 1
                
                self.results[request_id].update({
                    'status': 'failed',
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'completed_at': datetime.now().isoformat()
                })
                logger.error(f"Request {request_id} failed: {response.status_code}")
                
        except Exception as e:
            self.failed_requests += 1
            
            self.results[request_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            })
            logger.error(f"Error forwarding request {request_id}: {e}")
    
    def queue_request(self, image_data, filename):
        """Add request to queue and return request ID"""
        self.total_requests += 1
        
        request_id = str(uuid.uuid4())
        
        try:
            request_data = {
                'request_id': request_id,
                'image_data': (filename, image_data),
                'timestamp': time.time()
            }
            
            self.results[request_id] = {
                'status': 'queued',
                'filename': filename,
                'queued_at': datetime.now().isoformat(),
                'request_id': request_id
            }
            
            # Update peak queue size
            current_size = self.request_queue.qsize()
            self.peak_queue_size = max(self.peak_queue_size, current_size)
            self.last_request_time = time.time()
            
            self.request_queue.put(request_data, block=False)
            logger.info(f"Request {request_id} queued. Queue size: {self.request_queue.qsize()}")
            return request_id
            
        except queue.Full:
            self.failed_requests += 1
            
            self.results[request_id].update({
                'status': 'failed',
                'error': 'Queue is full',
                'completed_at': datetime.now().isoformat()
            })
            logger.error("Queue is full, dropping request")
            return request_id  # return ID so user can check the error
    
    def get_result(self, request_id):
        """Get result for a specific request ID"""
        return self.results.get(request_id, None)
    
    def get_status(self):
        """Get enhanced dispatcher status"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate derived metrics
        avg_processing_time = (
            self.total_processing_time / self.successful_requests 
            if self.successful_requests > 0 else 0
        )
        avg_queue_time = (
            self.total_queue_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        error_rate = (
            (self.failed_requests / self.total_requests) * 100 
            if self.total_requests > 0 else 0
        )
        throughput = self.successful_requests / uptime if uptime > 0 else 0
        
        return {
            # Existing metrics
            "queue_size": self.request_queue.qsize(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "stored_results": len(self.results),
            "endpoint_url": self.endpoint_url,
            
            # New metrics
            "performance_metrics": {
                "avg_processing_time": round(avg_processing_time, 3),
                "avg_queue_time": round(avg_queue_time, 3),
                "throughput": round(throughput, 2)
            },
            "queue_metrics": {
                "peak_queue_size": self.peak_queue_size,
                "queue_capacity": self.queue_capacity,
                "queue_utilization": round((self.request_queue.qsize() / self.queue_capacity) * 100, 2)
            },
            "health_metrics": {
                "error_rate": round(error_rate, 2),
                "uptime": round(uptime, 2),
                "last_request_timestamp": self.last_request_time
            }
        }

# Flask app
app = Flask(__name__)
dispatcher = Dispatcher()

@app.route('/predict_async', methods=['POST'])
def predict_async():
    """Asynchronous prediction - queue the request and return request ID"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Queue the request and get request ID
    request_id = dispatcher.queue_request(image_file.read(), image_file.filename)
    
    if request_id:
        return jsonify({
            "message": "Request queued for processing",
            "request_id": request_id,
            "queue_size": dispatcher.request_queue.qsize(),
            "status_url": f"/result/{request_id}"
        }), 202
    else:
        return jsonify({"error": "Failed to queue request"}), 503

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id):
    """Get result for a specific request ID"""
    result = dispatcher.get_result(request_id)
    
    if result is None:
        return jsonify({"error": "Request ID not found"}), 404
    
    return jsonify(result), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy"})

@app.route('/status', methods=['GET'])
def status():
    """Get dispatcher status"""
    return jsonify(dispatcher.get_status())

@app.route('/results', methods=['GET'])
def list_results():
    """List all stored results (for debugging)"""
    return jsonify({
        "total_results": len(dispatcher.results),
        "results": list(dispatcher.results.keys())
    }), 200

if __name__ == '__main__':
    print("Starting Simple Dispatcher with Async Results...")
    print("Endpoints:")
    print("  POST /predict_async     - Asynchronous prediction (queued)")
    print("  GET  /result/<id>       - Get result by request ID")
    print("  GET  /results           - List all result IDs")
    print("  GET  /health            - Health check")
    print("  GET  /status            - Dispatcher status")
    
    app.run(host='0.0.0.0', port=8080, debug=False)