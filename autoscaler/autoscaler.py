#!/usr/bin/env python3
"""
Queue-Based Kubernetes Autoscaler
Monitors dispatcher metrics and scales pods based on queue size and latency.
Uses Little's Law principles for optimal scaling decisions.
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from kubernetes import client, config
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScalingConfig:
    """Configuration for autoscaling behavior"""
    # Kubernetes settings
    namespace: str = "default"
    deployment_name: str = "resnet18-app"
    
    # Scaling thresholds (based on Little's Law)
    queue_size_threshold: int = 50  # Scale up when queue > 3 requests
    high_queue_threshold: int = 100   # Emergency scaling when queue > 8
    latency_threshold: float = 0.4  # Scale up when avg latency > 400ms
    error_rate_threshold: float = 5.0  # Scale up when error rate > 5%
    queue_utilization_threshold: float = 40.0  # Scale up when queue > 15% full
    
    # Replica limits
    min_replicas: int = 5
    max_replicas: int = 50
    
    # Scaling behavior
    scale_up_factor: int = 2      # Multiply replicas by this factor for emergency
    scale_up_increment: int = 1   # Normal scale up increment
    scale_down_increment: int = 1 # Scale down increment
    
    # Cooldown periods (seconds)
    scale_up_cooldown: int = 10
    scale_down_cooldown: int = 60
    
    # Monitoring
    metrics_url: str = "http://dispatcher-service:8080/status"
    poll_interval: int = 5  # seconds

class KubernetesAutoscaler:
    def __init__(self, scaling_config: ScalingConfig):
        self.config = scaling_config
        self.last_scale_up = None
        self.last_scale_down = None
        self.current_replicas = 0
        
        # Initialize Kubernetes client
        try:
            # Try to load in-cluster config first
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except:
            # Fall back to local kubeconfig
            config.load_kube_config()
            logger.info("Loaded local Kubernetes config")
        
        self.k8s_apps_v1 = client.AppsV1Api()
        
        # Get initial replica count
        self._update_current_replicas()
    
    def _update_current_replicas(self):
        """Get current replica count from deployment"""
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.config.namespace
            )
            self.current_replicas = deployment.spec.replicas
            logger.debug(f"Current replicas: {self.current_replicas}")
        except Exception as e:
            logger.error(f"Failed to get current replica count: {e}")
    
    def get_metrics(self) -> Optional[dict]:
        """Fetch metrics from dispatcher service"""
        try:
            response = requests.get(self.config.metrics_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
            return None
    
    def calculate_desired_replicas(self, metrics: dict) -> int:
        """
        Calculate desired replica count based on metrics using Little's Law
        L = λW (queue_length = arrival_rate * processing_time)
        """
        queue_size = metrics.get("queue_size", 0)
        queue_utilization = metrics["queue_metrics"].get("queue_utilization", 0)
        avg_processing_time = metrics["performance_metrics"].get("avg_processing_time", 0)
        avg_queue_time = metrics["performance_metrics"].get("avg_queue_time", 0)
        error_rate = metrics["health_metrics"].get("error_rate", 0)
        throughput = metrics["performance_metrics"].get("throughput", 0)
        
        # Total latency = processing time + queue time
        total_latency = avg_processing_time + avg_queue_time
        
        logger.info(f"Metrics - Queue: {queue_size}, Utilization: {queue_utilization}%, "
                   f"Latency: {total_latency:.3f}s, Error Rate: {error_rate}%, "
                   f"Throughput: {throughput:.2f}/s")
        
        desired_replicas = self.current_replicas
        scale_reason = ""
        
        # Emergency scaling: High queue size
        if queue_size >= self.config.high_queue_threshold:
            desired_replicas = min(
                self.current_replicas * self.config.scale_up_factor,
                self.config.max_replicas
            )
            scale_reason = f"Emergency: Queue size {queue_size} >= {self.config.high_queue_threshold}"
        
        # Scale up conditions
        elif (queue_size >= self.config.queue_size_threshold or
              total_latency >= self.config.latency_threshold or
              error_rate >= self.config.error_rate_threshold or
              queue_utilization >= self.config.queue_utilization_threshold):
            
            # Calculate replicas needed using Little's Law
            if throughput > 0 and avg_processing_time > 0:
                # Desired replicas = (arrival_rate * processing_time) / utilization_target
                # Using 70% utilization target for safety margin
                target_utilization = 0.7
                calculated_replicas = max(
                    int((throughput * avg_processing_time) / target_utilization) + 1,
                    self.current_replicas + self.config.scale_up_increment
                )
                desired_replicas = min(calculated_replicas, self.config.max_replicas)
            else:
                desired_replicas = min(
                    self.current_replicas + self.config.scale_up_increment,
                    self.config.max_replicas
                )
            
            scale_reason = (f"Scale up: Queue={queue_size}, Latency={total_latency:.3f}s, "
                          f"Error={error_rate}%, QueueUtil={queue_utilization}%")
        
        # Scale down conditions: Low load for sustained period
        elif (queue_size == 0 and 
              total_latency < self.config.latency_threshold * 0.5 and
              error_rate < self.config.error_rate_threshold * 0.5 and
              queue_utilization < self.config.queue_utilization_threshold * 0.3 and
              self.current_replicas > self.config.min_replicas):
            
            desired_replicas = max(
                self.current_replicas - self.config.scale_down_increment,
                self.config.min_replicas
            )
            scale_reason = "Scale down: Low load sustained"
        
        if scale_reason:
            logger.info(f"{scale_reason} -> Desired replicas: {desired_replicas}")
        
        return desired_replicas
    
    def can_scale(self, scale_up: bool) -> bool:
        """Check if scaling action is allowed based on cooldown periods"""
        now = datetime.now()
        
        if scale_up:
            if self.last_scale_up:
                time_since = (now - self.last_scale_up).total_seconds()
                return time_since >= self.config.scale_up_cooldown
            return True
        else:
            if self.last_scale_down:
                time_since = (now - self.last_scale_down).total_seconds()
                return time_since >= self.config.scale_down_cooldown
            return True
    
    def scale_deployment(self, desired_replicas: int) -> bool:
        """Scale the deployment to desired replica count"""
        if desired_replicas == self.current_replicas:
            return True
        
        scale_up = desired_replicas > self.current_replicas
        
        # Check cooldown
        if not self.can_scale(scale_up):
            cooldown = self.config.scale_up_cooldown if scale_up else self.config.scale_down_cooldown
            logger.info(f"Scaling blocked by {cooldown}s cooldown period")
            return False
        
        try:
            # Update deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.config.namespace
            )
            
            deployment.spec.replicas = desired_replicas
            
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.config.namespace,
                body=deployment
            )
            
            # Update tracking
            old_replicas = self.current_replicas
            self.current_replicas = desired_replicas
            
            if scale_up:
                self.last_scale_up = datetime.now()
            else:
                self.last_scale_down = datetime.now()
            
            logger.info(f"Scaled deployment from {old_replicas} to {desired_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    def run_scaling_loop(self):
        """Main scaling loop"""
        logger.info("Starting queue-based autoscaler...")
        logger.info(f"Monitoring: {self.config.metrics_url}")
        logger.info(f"Target: {self.config.deployment_name} in {self.config.namespace}")
        logger.info(f"Thresholds: Queue={self.config.queue_size_threshold}, "
                   f"Latency={self.config.latency_threshold}s, "
                   f"ErrorRate={self.config.error_rate_threshold}%")
        
        while True:
            try:
                # Get metrics
                metrics = self.get_metrics()
                if not metrics:
                    logger.warning("No metrics available, skipping scaling decision")
                    time.sleep(self.config.poll_interval)
                    continue
                
                # Update current state
                self._update_current_replicas()
                
                # Calculate desired replicas
                desired_replicas = self.calculate_desired_replicas(metrics)
                
                # Scale if needed
                self.scale_deployment(desired_replicas)
                
            except KeyboardInterrupt:
                logger.info("Autoscaler stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
            
            time.sleep(self.config.poll_interval)

def main():
    """Main entry point"""
    # Load configuration from environment variables
    scaling_config = ScalingConfig()
    
    # Start autoscaler
    autoscaler = KubernetesAutoscaler(scaling_config)
    autoscaler.run_scaling_loop()

if __name__ == "__main__":
    main()