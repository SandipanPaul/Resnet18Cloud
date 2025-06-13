#!/usr/bin/env python3
"""
Simple Image Load Tester
A minimal implementation for testing image APIs.
Automatically counts images in a folder and serves each image exactly once
using a realistic ramp-up pattern.
"""

import json
import base64
import os
import random
from pathlib import Path
from typing import Tuple, List
from barazmoon import BarAzmoon


class SimpleImageTester(BarAzmoon):
    def __init__(self, workload: List[int], endpoint: str, image_folder: str):
        super().__init__(workload=workload, endpoint=endpoint, http_method="post")
        self.image_folder = Path(image_folder)
        self.image_count = self._count_images()
        self._counter = 0
    
    def _count_images(self) -> int:
        """Count total number of images in the folder."""
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
        
        count = 0
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            count += len(list(self.image_folder.glob(f"*{ext}")))
        
        if count == 0:
            raise ValueError(f"No images found in {self.image_folder}")
        
        return count
    
    def get_request_data(self) -> Tuple[str, str]:
        """Load and encode an image."""
        # Get list of images each time (to avoid pickling issues)
        images = []
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            images.extend(list(self.image_folder.glob(f"*{ext}")))
        
        # Use counter to cycle through images
        image_path = images[self._counter % len(images)]
        self._counter += 1
        
        request_id = f"{image_path.stem}_{random.randint(1000, 9999)}"
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        request_data = {
            "image": encoded_image,
            "filename": image_path.name,
            "request_id": request_id
        }
        
        return request_id, json.dumps(request_data)
    
    def process_response(self, data_id: str, response: dict):
        """Handle the API response."""
        if hasattr(response, 'status_code'):
            if response['status_code'] == 200:
                print(f"✓ {data_id}: Success")
            else:
                print(f"✗ {data_id}: Failed ({response['status_code']})")
        else:
            print(f"? {data_id}: Unknown response")
        return True

def create_realistic_workload(num_images: int) -> List[int]:

    if num_images <= 0:
        return []
    
    # For very small counts, just use 1 RPS
    if num_images <= 10:
        return [1] * num_images
    
    # Define load levels
    min_rps = 1
    valley_rps = max(2, min_rps + 1)  # Valley between peaks
    peak_rps = min(30, max(3, num_images // 20))  # Scale peak based on image count, max 30
    
    # Calculate phase durations
    ramp_up_duration = max(3, min(8, num_images // 25))
    ramp_down_duration = max(3, min(8, num_images // 25))
    
    # Number of peaks (2-4 depending on image count)
    num_peaks = min(4, max(2, num_images // 50))
    valley_duration = max(2, min(5, num_images // 40))
    
    workload = []
    total_requests_used = 0
    
    # Phase 1: Initial ramp up
    for i in range(ramp_up_duration):
        progress = (i + 1) / ramp_up_duration
        current_rps = min_rps + int((peak_rps - min_rps) * progress)
        workload.append(current_rps)
        total_requests_used += current_rps
    
    # Calculate requests needed for valleys and final ramp down
    valley_requests = valley_duration * valley_rps * (num_peaks - 1)  # Valleys between peaks
    
    ramp_down_requests = 0
    for i in range(ramp_down_duration):
        progress = (ramp_down_duration - i) / ramp_down_duration
        current_rps = valley_rps + int((peak_rps - valley_rps) * progress)
        ramp_down_requests += current_rps
    
    # Remaining requests for all peaks
    remaining_for_peaks = num_images - total_requests_used - valley_requests - ramp_down_requests
    
    if remaining_for_peaks > 0:
        # Distribute requests among peaks with some variation
        peak_requests = []
        base_requests_per_peak = remaining_for_peaks // num_peaks
        
        for i in range(num_peaks):
            # Add some variation to peak sizes (±20%)
            variation = int(base_requests_per_peak * 0.2)
            if i == 0:  # First peak slightly higher
                peak_size = base_requests_per_peak + variation // 2
            elif i == num_peaks - 1:  # Last peak gets remainder
                peak_size = remaining_for_peaks - sum(peak_requests)
            else:
                import random
                peak_size = base_requests_per_peak + random.randint(-variation, variation)
            
            peak_requests.append(max(1, peak_size))
        
        # Create peaks with valleys between them
        for i, peak_size in enumerate(peak_requests):
            # Add peak
            if peak_size > 0:
                peak_duration = peak_size // peak_rps
                peak_remainder = peak_size % peak_rps
                
                # Add full peak periods
                workload.extend([peak_rps] * peak_duration)
                total_requests_used += peak_duration * peak_rps
                
                # Add partial peak if needed
                if peak_remainder > 0:
                    workload.append(peak_remainder)
                    total_requests_used += peak_remainder
            
            # Add valley between peaks (except after last peak)
            if i < num_peaks - 1:
                workload.extend([valley_rps] * valley_duration)
                total_requests_used += valley_duration * valley_rps
    
    # Phase 3: Final ramp down
    requests_left = num_images - total_requests_used
    for i in range(ramp_down_duration):
        if requests_left <= 0:
            break
        progress = (ramp_down_duration - i) / ramp_down_duration
        target_rps = valley_rps + int((peak_rps - valley_rps) * progress)
        actual_rps = min(target_rps, requests_left)
        workload.append(actual_rps)
        requests_left -= actual_rps
    
    # Add any final remaining requests
    if requests_left > 0:
        workload.append(requests_left)
    
    return workload



# Example usage
if __name__ == "__main__":
    # Configuration
    API_ENDPOINT = "http://127.0.0.1:8080/predict_async"  # Change this to your API
    IMAGE_FOLDER = "./images/imagenet-sample-images"  # Folder containing your test images
    
    # First, create a temporary tester to get the number of images
    try:
        temp_tester = SimpleImageTester(workload=[1], endpoint=API_ENDPOINT, image_folder=IMAGE_FOLDER)
        num_images = temp_tester.image_count
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your image folder exists and contains image files.")
        exit(1)
    
    # Use realistic workload pattern
    workload = create_realistic_workload(num_images)
    print(f"Using realistic workload pattern - each image will be tested once")
    
    print(f"Found {num_images} images in {IMAGE_FOLDER}")
    print(f"Test duration: {len(workload)} seconds")
    print(f"Peak load: {max(workload)} requests/second")
    print(f"Total requests: {sum(workload)}")
    print(f"workload: {workload}" )
    
    try:
        # Create and run the tester
        tester = SimpleImageTester(
            workload=workload,
            endpoint=API_ENDPOINT,
            image_folder=IMAGE_FOLDER,
        )
        
        print(f"\nStarting load test against {API_ENDPOINT}")
        print("Press Ctrl+C to stop early\n")
        
        # Start the test
        tester.start()
        
        print("\nLoad test completed!")
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error during test: {e}")