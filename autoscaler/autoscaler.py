from kubernetes import client, config
import requests
import time
import logging
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import threading
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveLatencyAutoscaler:
    """
    Advanced Kubernetes autoscaler optimized for < 0.5s latency using:
    - Predictive scaling with time series analysis
    - Little's Law for queue-based decision making
    - Proactive scaling considering pod startup time
    - Multi-metric latency-sensitive scaling logic
    
    Inspired by:
    - Google Cloud's predictive autoscaling (https://cloud.google.com/compute/docs/autoscaler/predictive-autoscaling)
    - PredictKube AI-based predictive autoscaler (https://keda.sh/blog/2022-02-09-predictkube-scaler/)
    - Research on LSTM-based autoscaling (https://www.mdpi.com/2076-3417/11/9/3835)
    - Little's Law for queue optimization (https://blog.danslimmon.com/2022/06/07/using-littles-law-to-scale-applications/)
    """
    
    def __init__(self):
        # Load Kubernetes configuration
        config.load_kube_config()
        self.k8s_apps_v1 = client.AppsV1Api()
        
        # Deployment configuration
        self.min_replicas = 3
        self.max_replicas = 100
        self.deployment_name = "resnet18-app"
        self.namespace = "default"
        
        # Latency-optimized thresholds (much more aggressive)
        self.target_latency = 0.4  # Target 400ms to stay under 500ms with buffer
        self.max_acceptable_latency = 0.5  # Hard limit
        self.queue_latency_target = 0.1  # Target queue time < 100ms
        
        # Little's Law parameters (L = λW)
        self.target_queue_length = 5  # Much lower than original 50%
        self.max_safe_queue_length = 10
        
        # Pod startup considerations
        self.pod_startup_time = 30  # seconds
        self.scale_ahead_buffer = 45  # seconds to scale ahead of predicted load
        
        # Predictive scaling parameters
        self.history_window = 1440  # 24 hours of minute-level data
        self.prediction_horizon = 15  # Predict 15 minutes ahead
        self.seasonal_patterns = True
        
        # Scaling aggressiveness for low latency
        self.emergency_scale_factor = 2.0  # Double replicas on emergency
        self.normal_scale_factor = 1.5  # 50% increase on normal scale-up
        self.min_scale_step = 2  # Minimum replicas to add/remove
        
        # Faster polling for responsiveness
        self.check_interval = 5  # Check every 5 seconds instead of 15
        self.metrics_history = deque(maxlen=self.history_window)
        
        # Dispatcher metrics endpoint
        self.dispatcher_url = "http://localhost:8080/status"
        
        # Threading for concurrent operations
        self.prediction_lock = threading.Lock()
        self.current_prediction = None
        self.last_prediction_time = None
        
        logger.info("Initialized PredictiveLatencyAutoscaler with aggressive latency optimization")

    def get_enhanced_metrics(self) -> Optional[Dict]:
        """Fetch enhanced metrics including latency percentiles and queue statistics"""
        try:
            response = requests.get(self.dispatcher_url, timeout=2)
            metrics = response.json()
            
            # Calculate enhanced metrics
            enhanced_metrics = self._calculate_enhanced_metrics(metrics)
            
            # Store for historical analysis
            timestamp = datetime.now()
            enhanced_metrics['timestamp'] = timestamp
            self.metrics_history.append(enhanced_metrics)
            
            return enhanced_metrics
            
        except Exception as e:
            logger.error(f"Failed to fetch enhanced metrics: {e}")
            return None

    def _calculate_enhanced_metrics(self, raw_metrics: Dict) -> Dict:
        """Calculate enhanced metrics using Little's Law and queue theory"""
        
        # Extract basic metrics
        queue_size = raw_metrics['queue_metrics']['queue_size']
        queue_utilization = raw_metrics['queue_metrics']['queue_utilization']
        avg_processing_time = raw_metrics['performance_metrics']['avg_processing_time']
        request_rate = raw_metrics['performance_metrics'].get('request_rate', 0)
        
        # Calculate latency components using Little's Law (L = λW)
        if request_rate > 0:
            # Queue waiting time using Little's Law
            queue_wait_time = queue_size / request_rate if request_rate > 0 else 0
            
            # Total estimated latency = queue_wait + processing_time
            estimated_total_latency = queue_wait_time + avg_processing_time
            
            # Queue pressure indicator
            queue_pressure = queue_size / self.max_safe_queue_length
        else:
            queue_wait_time = 0
            estimated_total_latency = avg_processing_time
            queue_pressure = 0
        
        # Calculate utilization ratios
        latency_pressure = estimated_total_latency / self.target_latency
        emergency_threshold = estimated_total_latency > self.max_acceptable_latency
        
        return {
            'queue_size': queue_size,
            'queue_utilization': queue_utilization,
            'avg_processing_time': avg_processing_time,
            'request_rate': request_rate,
            'queue_wait_time': queue_wait_time,
            'estimated_total_latency': estimated_total_latency,
            'latency_pressure': latency_pressure,
            'queue_pressure': queue_pressure,
            'emergency_threshold': emergency_threshold,
            'error_rate': raw_metrics['health_metrics']['error_rate']
        }

    def simple_time_series_prediction(self, history: List[float], horizon: int = 5) -> List[float]:
        """
        Simplified predictive model using moving averages and trend analysis.
        Inspired by research on time series forecasting for autoscaling.
        """
        if len(history) < 10:
            return [history[-1]] * horizon if history else [0] * horizon
        
        # Convert to numpy for easier manipulation
        data = np.array(history)
        
        # Simple trend analysis
        recent_window = min(30, len(data) // 2)
        recent_data = data[-recent_window:]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_data))
        trend = np.polyfit(x, recent_data, 1)[0] if len(recent_data) > 1 else 0
        
        # Seasonal component (basic daily pattern detection)
        seasonal_adjustment = 0
        if len(data) >= 60:  # Need at least 1 hour of data
            hour_of_day = datetime.now().hour
            same_hour_data = [data[i] for i in range(len(data)) if (len(data) - i - 1) % 60 == (60 - datetime.now().minute) % 60]
            if same_hour_data:
                seasonal_adjustment = np.mean(same_hour_data) - np.mean(data)
        
        # Generate predictions
        last_value = data[-1]
        predictions = []
        
        for i in range(horizon):
            # Base prediction with trend
            predicted = last_value + (trend * (i + 1))
            
            # Add seasonal adjustment
            predicted += seasonal_adjustment * 0.3  # Dampen seasonal effect
            
            # Ensure non-negative
            predicted = max(0, predicted)
            predictions.append(predicted)
        
        return predictions

    def predict_future_load(self) -> Optional[Tuple[List[float], List[float]]]:
        """
        Predict future request rate and latency using time series analysis.
        Based on research findings from LSTM and Facebook Prophet models.
        """
        with self.prediction_lock:
            if len(self.metrics_history) < 10:
                return None
            
            # Extract time series data
            request_rates = [m['request_rate'] for m in self.metrics_history]
            latencies = [m['estimated_total_latency'] for m in self.metrics_history]
            
            # Generate predictions
            predicted_rates = self.simple_time_series_prediction(request_rates, self.prediction_horizon)
            predicted_latencies = self.simple_time_series_prediction(latencies, self.prediction_horizon)
            
            self.current_prediction = (predicted_rates, predicted_latencies)
            self.last_prediction_time = datetime.now()
            
            return self.current_prediction

    def calculate_required_replicas_little_law(self, metrics: Dict, current_replicas: int) -> int:
        """
        Calculate required replicas using Little's Law and queue theory principles.
        Inspired by: https://blog.danslimmon.com/2022/06/07/using-littles-law-to-scale-applications/
        """
        request_rate = metrics['request_rate']
        target_latency = self.target_latency
        avg_processing_time = metrics['avg_processing_time']
        
        if request_rate == 0:
            return max(self.min_replicas, current_replicas)
        
        # Using Little's Law: L = λW
        # We want: total_latency <= target_latency
        # total_latency = queue_wait_time + processing_time
        # queue_wait_time = queue_size / request_rate
        # queue_size depends on replicas (more replicas = lower queue)
        
        # Calculate required service rate to meet latency target
        max_acceptable_queue_time = target_latency - avg_processing_time
        max_acceptable_queue_time = max(0, max_acceptable_queue_time)
        
        if max_acceptable_queue_time <= 0:
            # Processing time alone exceeds target, need immediate scaling
            required_processing_capacity = request_rate / (target_latency * 0.8)  # 20% buffer
        else:
            # Calculate replicas needed to keep queue time within limits
            # Assuming each replica can handle a certain rate
            service_rate_per_replica = 1.0 / avg_processing_time if avg_processing_time > 0 else 1.0
            
            # Total service rate needed to maintain queue size
            required_service_rate = request_rate + (request_rate * max_acceptable_queue_time)
            required_processing_capacity = required_service_rate / service_rate_per_replica
        
        # Add buffer for safety and pod startup time
        safety_buffer = 1.2
        required_replicas = int(math.ceil(required_processing_capacity * safety_buffer))
        
        # Ensure within bounds
        required_replicas = max(self.min_replicas, min(self.max_replicas, required_replicas))
        
        return required_replicas

    def calculate_predictive_replicas(self, current_replicas: int) -> int:
        """
        Calculate replicas needed based on predicted future load.
        Incorporates pod startup time for proactive scaling.
        """
        prediction = self.predict_future_load()
        if not prediction:
            return current_replicas
        
        predicted_rates, predicted_latencies = prediction
        
        # Look ahead considering pod startup time
        startup_steps = min(len(predicted_rates), self.pod_startup_time // (self.check_interval * 60))
        if startup_steps == 0:
            startup_steps = 1
        
        # Get the maximum predicted values in the startup window
        max_predicted_rate = max(predicted_rates[:startup_steps])
        max_predicted_latency = max(predicted_latencies[:startup_steps])
        
        # Calculate replicas needed for predicted load
        if max_predicted_latency > self.target_latency or max_predicted_rate > 0:
            # Simulate metrics for prediction
            predicted_metrics = {
                'request_rate': max_predicted_rate,
                'estimated_total_latency': max_predicted_latency,
                'avg_processing_time': max_predicted_latency * 0.8  # Estimate
            }
            
            return self.calculate_required_replicas_little_law(predicted_metrics, current_replicas)
        
        return current_replicas

    def get_current_replicas(self) -> Optional[int]:
        """Get current number of replicas"""
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            return deployment.spec.replicas
        except Exception as e:
            logger.error(f"Failed to get deployment info: {e}")
            return None

    def scale_deployment(self, replicas: int) -> bool:
        """Scale deployment to specified number of replicas"""
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            
            old_replicas = deployment.spec.replicas
            deployment.spec.replicas = replicas
            
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment from {old_replicas} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False

    def calculate_optimal_replicas(self, metrics: Dict, current_replicas: int) -> Tuple[int, str]:
        """
        Main scaling decision logic using multiple advanced techniques.
        Returns (desired_replicas, reason)
        """
        
        # Emergency scaling - immediate response to latency violations
        if metrics['emergency_threshold']:
            emergency_replicas = min(
                int(current_replicas * self.emergency_scale_factor),
                self.max_replicas
            )
            return emergency_replicas, f"EMERGENCY: Latency {metrics['estimated_total_latency']:.3f}s > {self.max_acceptable_latency}s"
        
        # Calculate replicas using different methods
        little_law_replicas = self.calculate_required_replicas_little_law(metrics, current_replicas)
        predictive_replicas = self.calculate_predictive_replicas(current_replicas)
        
        # Reactive scaling based on current metrics
        reactive_replicas = current_replicas
        reasons = []
        
        # High latency pressure
        if metrics['latency_pressure'] > 1.2:
            reactive_replicas = max(reactive_replicas, int(current_replicas * self.normal_scale_factor))
            reasons.append(f"Latency pressure {metrics['latency_pressure']:.2f}")
        
        # Queue pressure
        if metrics['queue_pressure'] > 0.7:
            reactive_replicas = max(reactive_replicas, current_replicas + self.min_scale_step)
            reasons.append(f"Queue pressure {metrics['queue_pressure']:.2f}")
        
        # High error rate
        if metrics['error_rate'] > 5:
            reactive_replicas = max(reactive_replicas, current_replicas + self.min_scale_step)
            reasons.append(f"Error rate {metrics['error_rate']:.1f}%")
        
        # Choose the most aggressive scaling recommendation
        max_replicas = max(little_law_replicas, predictive_replicas, reactive_replicas)
        
        # Scale down logic - be more conservative
        if max_replicas < current_replicas:
            # Only scale down if all indicators suggest it's safe
            if (metrics['latency_pressure'] < 0.5 and 
                metrics['queue_pressure'] < 0.3 and 
                metrics['error_rate'] < 1):
                
                scale_down_target = max(
                    current_replicas - self.min_scale_step,
                    self.min_replicas
                )
                return scale_down_target, "Safe to scale down - low pressure"
            else:
                return current_replicas, "Maintaining current scale - mixed signals"
        
        # Determine primary reason
        if max_replicas == little_law_replicas:
            primary_reason = f"Little's Law optimization for latency target"
        elif max_replicas == predictive_replicas:
            primary_reason = f"Predictive scaling for anticipated load"
        else:
            primary_reason = f"Reactive scaling: {', '.join(reasons)}"
        
        return max_replicas, primary_reason

    def run(self):
        """Main autoscaling loop with enhanced monitoring and prediction"""
        logger.info("Starting Advanced Predictive Latency Autoscaler...")
        
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while True:
            try:
                start_time = time.time()
                
                # Get enhanced metrics
                metrics = self.get_enhanced_metrics()
                current_replicas = self.get_current_replicas()
                
                if metrics and current_replicas is not None:
                    # Calculate optimal replica count
                    desired_replicas, reason = self.calculate_optimal_replicas(metrics, current_replicas)
                    
                    # Log current state
                    logger.info(
                        f"Metrics - Latency: {metrics['estimated_total_latency']:.3f}s, "
                        f"Queue: {metrics['queue_size']}, Rate: {metrics['request_rate']:.1f}/s, "
                        f"Replicas: {current_replicas} -> {desired_replicas}"
                    )
                    
                    # Execute scaling if needed
                    if desired_replicas != current_replicas:
                        if self.scale_deployment(desired_replicas):
                            logger.info(f"Scaling reason: {reason}")
                        else:
                            logger.error("Failed to execute scaling")
                    
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    logger.warning(f"Failed to get metrics or replica count (attempt {consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Too many consecutive errors, entering safe mode")
                        time.sleep(30)  # Back off on persistent errors
                        consecutive_errors = 0
                
                # Ensure consistent polling interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Autoscaler stopped by user")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in autoscaling loop: {e}")
                time.sleep(self.check_interval)

if __name__ == "__main__":
    # Example usage with configuration validation
    autoscaler = PredictiveLatencyAutoscaler()
    
    logger.info("Configuration Summary:")
    logger.info(f"Target latency: {autoscaler.target_latency}s")
    logger.info(f"Max acceptable latency: {autoscaler.max_acceptable_latency}s")
    logger.info(f"Check interval: {autoscaler.check_interval}s")
    logger.info(f"Replica range: {autoscaler.min_replicas}-{autoscaler.max_replicas}")
    logger.info(f"Pod startup buffer: {autoscaler.pod_startup_time}s")
    
    try:
        autoscaler.run()
    except KeyboardInterrupt:
        logger.info("Autoscaler terminated gracefully")