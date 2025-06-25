#!/usr/bin/env python3
"""
Image Load Tester
Sends POST requests with random images to simulate load testing
"""

import os
import time
import random
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageLoadTester:
    def __init__(self, image_dir="images/imagenet-sample-images", base_url="http://127.0.0.1:8080"):
        self.image_dir = Path(image_dir)
        self.base_url = base_url
        self.endpoint = f"{base_url}/predict_async"
        self.image_files = self._get_image_files()
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.lock = threading.Lock()
        
    def _get_image_files(self):
        """Get list of all image files in the directory"""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} not found")
            
        # Common image extensions
        extensions = {'.jpg', '.jpeg'}
        image_files = []
        
        for file_path in self.image_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                image_files.append(file_path)
                
        if not image_files:
            raise ValueError(f"No image files found in {self.image_dir}")
            
        logger.info(f"Found {len(image_files)} image files")
        return image_files
    
    def send_request(self, image_path):
        """Send a single POST request with an image"""
        start_time = time.time()
        
        try:
            with open(image_path, 'rb') as img_file:
                files = {'image': (image_path.name, img_file, 'image/jpeg')}
                response = requests.post(self.endpoint, files=files, timeout=30)
                
            end_time = time.time()
            response_time = end_time - start_time
            
            with self.lock:
                self.total_requests += 1
                self.response_times.append(response_time)
                
                if response.status_code in [200,202]:
                    self.successful_requests += 1
                    logger.debug(f"✓ Success: {image_path.name} - {response.status_code} - {response_time:.3f}s")
                else:
                    self.failed_requests += 1
                    logger.warning(f"✗ Failed: {image_path.name} - {response.status_code} - {response_time:.3f}s")
                    
            return {
                'success': response.status_code in [200,202],
                'status_code': response.status_code,
                'response_time': response_time,
                'image': image_path.name
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            with self.lock:
                self.total_requests += 1
                self.failed_requests += 1
                self.response_times.append(response_time)
                
            logger.error(f"✗ Error: {image_path.name} - {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time,
                'image': image_path.name
            }
    
    def run_load_test(self, images_per_second_schedule, max_workers=50):
        """
        Run load test with varying number of images sent per second
        
        Args:
            images_per_second_schedule: List of images per second values
            max_workers: Maximum number of concurrent threads
        """
        logger.info(f"Starting load test - Images per second schedule: {images_per_second_schedule}")
        logger.info(f"Max workers: {max_workers}")
        
        total_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            current_time = total_start_time
            second_counter = 0
            
            for images_per_sec in images_per_second_schedule:
                logger.info(f"Second {second_counter + 1}: Sending {images_per_sec} images")
                
                if images_per_sec == 0:
                    logger.info("Resting - no images sent this second")
                    time.sleep(1.0)
                    second_counter += 1
                    continue
                
                # Calculate interval between images for this second
                interval_between_images = 1.0 / images_per_sec
                second_start_time = time.time()
                
                # Send the specified number of images in this second
                for i in range(images_per_sec):
                    # Select random image from the 1000 available
                    random_image = random.choice(self.image_files)
                    
                    # Submit image POST request
                    future = executor.submit(self.send_request, random_image)
                    futures.append(future)
                    
                    # Wait for the interval before next image (except for the last one)
                    if i < images_per_sec - 1:
                        time.sleep(interval_between_images)
                
                # Ensure we spent exactly 1 second on this batch
                elapsed = time.time() - second_start_time
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
                
                second_counter += 1
            
            # Wait for all remaining requests to complete
            logger.info(f"Waiting for {len(futures)} total image requests to complete...")
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Image request failed: {e}")
        
        total_duration = time.time() - total_start_time
        logger.info(f"\n=== Load Test Completed in {total_duration:.1f}s ===")
    
    '''
    def print_stats(self):
        """Print current statistics"""
        with self.lock:
            if self.total_requests == 0:
                return
                
            success_rate = (self.successful_requests / self.total_requests) * 100
            avg_response_time = sum(self.response_times) / len(self.response_times)
            
            logger.info(f"Stats: {self.total_requests} total, {self.successful_requests} success, "
                       f"{self.failed_requests} failed, {success_rate:.1f}% success rate, "
                       f"{avg_response_time:.3f}s avg response time")
    
    def print_final_stats(self):
        """Print detailed final statistics"""
        with self.lock:
            if not self.response_times:
                logger.info("No requests completed")
                return
                
            self.response_times.sort()
            total = len(self.response_times)
            
            stats = {
                'Total Requests': self.total_requests,
                'Successful': self.successful_requests,
                'Failed': self.failed_requests,
                'Success Rate': f"{(self.successful_requests / self.total_requests) * 100:.1f}%",
                'Avg Response Time': f"{sum(self.response_times) / total:.3f}s",
                'Min Response Time': f"{min(self.response_times):.3f}s",
                'Max Response Time': f"{max(self.response_times):.3f}s",
                'P50 Response Time': f"{self.response_times[int(total * 0.5)]:.3f}s",
                'P95 Response Time': f"{self.response_times[int(total * 0.95)]:.3f}s",
                'P99 Response Time': f"{self.response_times[int(total * 0.99)]:.3f}s",
            }
            
            print("\n" + "="*50)
            print("FINAL LOAD TEST RESULTS")
            print("="*50)
            for key, value in stats.items():
                print(f"{key:20}: {value}")
            print("="*50)
        '''

def main():
    # Configuration
    IMAGE_DIR = "images/imagenet-sample-images"
    BASE_URL = "http://127.0.0.1:36135"
    
    # Define your load test schedule (number of images to POST per second)
    # Each number represents how many random images to send in that second
    LOAD_SCHEDULE = [4, 8, 11, 15, 19, 22, 26, 30, 30, 30, 30, 30, 30, 30, 10, 2, 2, 2, 2, 2, 30, 30, 30, 30, 27, 2, 2, 2, 2, 2, 30, 30, 30, 30, 30, 30, 1, 2, 2, 2, 2, 2, 30, 30, 30, 30, 30, 27, 30, 26, 23, 19, 16, 12, 9, 5]
    
    # Maximum concurrent workers
    MAX_WORKERS = 50
    
    try:
        # Initialize load tester
        tester = ImageLoadTester(IMAGE_DIR, BASE_URL)
        
        # Run the load test
        tester.run_load_test(
            images_per_second_schedule=LOAD_SCHEDULE,
            max_workers=MAX_WORKERS
        )
        
    except KeyboardInterrupt:
        logger.info("\nLoad test interrupted by user")
    except Exception as e:
        logger.error(f"Load test failed: {e}")


if __name__ == "__main__":
    main()