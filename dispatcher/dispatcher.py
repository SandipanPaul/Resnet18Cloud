import queue
import threading
import time
import requests
from flask import Flask, request, jsonify
import logging
import uuid
from datetime import datetime, timedelta
import hashlib
from collections import OrderedDict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageCache:
    """LRU Cache for processed images"""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = OrderedDict()  # LRU cache using OrderedDict
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()  # Thread safety
    
    def _generate_hash(self, image_data):
        """Generate SHA-256 hash of image content"""
        return hashlib.blake2b(image_data, digest_size=16).hexdigest()
    
    def get(self, image_data):
        """Get cached result for image"""
        image_hash = self._generate_hash(image_data)
        
        with self.lock:
            if image_hash in self.cache:
                # Move to end (most recently used)
                result = self.cache.pop(image_hash)
                self.cache[image_hash] = result
                self.hits += 1
                logger.info(f"Cache HIT for image hash: {image_hash[:12]}...")
                return result
            else:
                self.misses += 1
                logger.info(f"Cache MISS for image hash: {image_hash[:12]}...")
                return None
    
    def put(self, image_data, result):
        """Store result in cache"""
        image_hash = self._generate_hash(image_data)
        
        with self.lock:
            # Remove oldest items if cache is full
            while len(self.cache) >= self.max_size:
                oldest_hash, _ = self.cache.popitem(last=False)
                logger.info(f"Cache evicted oldest entry: {oldest_hash[:12]}...")
            
            # Add new result
            self.cache[image_hash] = {
                'result': result,
                'cached_at': datetime.now().isoformat(),
                'image_hash': image_hash
            }
            logger.info(f"Cached result for image hash: {image_hash[:12]}...")
    
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'utilization': round((len(self.cache) / self.max_size * 100), 2)
            }
    
    def clear(self):
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")

class Dispatcher:
    def __init__(self):
        # Default endpoint if config isn’t found or is invalid
        default_url = "http://resnet18-service.default.svc.cluster.local:5000"
        #default_url = "http://127.0.0.1:5000"
        
        self.endpoint_url = default_url
        
        # Initialize cache
        self.cache = ImageCache(max_size=1000)  # Configurable cache size
        
        # Request queue
        self.request_queue = queue.Queue(maxsize=200)
        
        # Result storage for async requests
        self.results = {}  # {request_id: {status, result, timestamp}}
        
        # Current replica index for round-robin
        self.current_replica = 0
        
        # Metrics
        self.total_requests = 0
        self.cache_hits=0
        self.successful_requests = 0
        self.failed_requests = 0
        self.queue_full_errors = 0  # Separate queue-full errors from processing errors
        
        # Performance metrics
        self.start_time = time.time()
        self.peak_queue_size = 0
        self.total_processing_time = 0
        self.total_queue_time = 0
        self.last_request_time = None
        self.queue_capacity = 200  # matches maxsize from queue initialization
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        logger.info("Dispatcher initialized with caching enabled")
    
    def _process_queue(self):
        """Background worker to process queued requests"""
        while True:
            try:
                # Blocking request
                request_data = self.request_queue.get(timeout=0.5)
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
                result_data = response.json()
                
                # Store result in cache for future use
                image_content = request_data['image_data'][1]  # Extract image bytes
                self.cache.put(image_content, result_data)
                
                self.results[request_id].update({
                    'status': 'completed',
                    'result': result_data,
                    'processing_time': processing_time,
                    'completed_at': datetime.now().isoformat(),
                    'from_cache': False
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
        """Add request to queue and return request ID, with cache check"""
        self.total_requests += 1
        request_id = str(uuid.uuid4())
        
        # Check cache first
        cached_result = self.cache.get(image_data)
        if cached_result:
            self.cache_hits += 1
            self.successful_requests += 1
            
            # Return cached result immediately
            self.results[request_id] = {
                'status': 'completed',
                'result': cached_result['result'],
                'filename': filename,
                'request_id': request_id,
                'queued_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat(),
                'from_cache': True,
                'cached_at': cached_result['cached_at'],
                'processing_time': 0.001  # Negligible cache lookup time
            }
            
            logger.info(f"Request {request_id} served from cache immediately")
            return request_id
        
        # Not in cache, proceed with normal queueing
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
                'request_id': request_id,
                'from_cache': False
            }
            
            # Update peak queue size and last request time
            current_size = self.request_queue.qsize()
            self.peak_queue_size = max(self.peak_queue_size, current_size)
            self.last_request_time = time.time()
            
            self.request_queue.put(request_data, block=False)
            logger.info(f"Request {request_id} queued for processing. Queue size: {self.request_queue.qsize()}")
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
        """Get enhanced dispatcher status with corrected metrics"""
        
        self.stats_lock = threading.Lock()
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Get cache stats to avoid redundant tracking
        cache_stats = self.cache.get_stats()
        
        # Thread-safe access to metrics
        with self.stats_lock:
            total_requests = self.total_requests
            successful_requests = self.successful_requests
            failed_requests = self.failed_requests
            queue_full_errors = self.queue_full_errors
            total_processing_time = self.total_processing_time
            total_queue_time = self.total_queue_time
            peak_queue_size = self.peak_queue_size
            last_request_time = self.last_request_time
        
        # Calculate requests that went through the queue (not cached)
        queued_requests = total_requests - cache_stats['hits']
        processing_errors = failed_requests - queue_full_errors
        
        # Calculate derived metrics with proper denominators
        avg_processing_time = (
            total_processing_time / successful_requests 
            if successful_requests > 0 else 0
        )
        
        # Fixed: Use queued_requests instead of total_requests
        avg_queue_time = (
            total_queue_time / queued_requests 
            if queued_requests > 0 else 0
        )
        
        error_rate = (
            (failed_requests / total_requests) * 100 
            if total_requests > 0 else 0
        )
        
        processing_error_rate = (
            (processing_errors / queued_requests) * 100 
            if queued_requests > 0 else 0
        )
        
        throughput = successful_requests / uptime if uptime > 0 else 0
        
        # Use cache's internal hit rate calculation
        cache_hit_rate = cache_stats['hit_rate']
        
        return {
            # Existing metrics
            "queue_size": self.request_queue.qsize(),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "queued_requests": queued_requests,  # Add this for transparency
            "stored_results": len(self.results),
            "endpoint_url": self.endpoint_url,
            
            # Performance metrics
            "performance_metrics": {
                "avg_processing_time": round(avg_processing_time, 3),
                "avg_queue_time": round(avg_queue_time, 3),  # Now correctly calculated
                "throughput": round(throughput, 2)
            },
            "queue_metrics": {
                "peak_queue_size": peak_queue_size,
                "queue_capacity": self.queue_capacity,
                "queue_utilization": round((self.request_queue.qsize() / self.queue_capacity) * 100, 2)
            },
            "health_metrics": {
                "error_rate": round(error_rate, 2),
                "processing_error_rate": round(processing_error_rate, 2),
                "queue_full_errors": queue_full_errors,
                "processing_errors": processing_errors,
                "uptime": round(uptime, 2),
                "last_request_timestamp": last_request_time
            },
            
            # Use cache's internal metrics instead of redundant tracking
            "cache_metrics": {
                "cache_hit_rate": cache_hit_rate,
                **cache_stats
            }
        }
    
    def clear_cache(self):
        """Clear the image cache"""
        self.cache.clear()

# Flask app
app = Flask(__name__)
dispatcher = Dispatcher()

@app.route('/predict_async', methods=['POST'])
def predict_async():
    """Asynchronous prediction - check cache first, then queue if needed"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Queue the request (cache check happens inside)
    request_id = dispatcher.queue_request(image_file.read(), image_file.filename)
    
    if request_id:
        # Check if result was served from cache
        result = dispatcher.get_result(request_id)
        if result and result.get('from_cache', False):
            return jsonify({
                "message": "Result served from cache",
                "request_id": request_id,
                "from_cache": True,
                "result": result['result'],
                "status_url": f"/result/{request_id}"
            }), 200
        else:
            return jsonify({
                "message": "Request queued for processing",
                "request_id": request_id,
                "queue_size": dispatcher.request_queue.qsize(),
                "from_cache": False,
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
    """Get dispatcher status including cache metrics"""
    return jsonify(dispatcher.get_status())

@app.route('/results', methods=['GET'])
def list_results():
    """List all stored results (for debugging)"""
    return jsonify({
        "total_results": len(dispatcher.results),
        "results": list(dispatcher.results.keys())
    }), 200

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the image cache"""
    dispatcher.clear_cache()
    return jsonify({"message": "Cache cleared successfully"}), 200

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get detailed cache statistics"""
    return jsonify(dispatcher.cache.get_stats()), 200

if __name__ == '__main__':
    print("Starting Enhanced Dispatcher with Image Caching...")
    print("Endpoints:")
    print("  POST /predict_async     - Asynchronous prediction (cached)")
    print("  GET  /result/<id>       - Get result by request ID")
    print("  GET  /results           - List all result IDs")
    print("  GET  /health            - Health check")
    print("  GET  /status            - Dispatcher status with cache metrics")
    print("  GET  /cache/stats       - Detailed cache statistics")
    print("  POST /cache/clear       - Clear image cache")
    
    app.run(host='0.0.0.0', port=8080, debug=False)