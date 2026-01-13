"""
Monitoring : Latence, requêtes, erreurs
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
from collections import deque

logger = logging.getLogger(__name__)

class MonitoringManager:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.request_times = deque(maxlen=max_history)
        self.error_count = 0
        self.total_requests = 0
        self.predictions_cache = deque(maxlen=max_history)
        self.start_time = datetime.now()
    
    def record_request(self, latency: float, success: bool = True, model_version: str = None):
        """Enregistrer une requête"""
        self.total_requests += 1
        
        if not success:
            self.error_count += 1
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency * 1000,
            "success": success,
            "model_version": model_version
        }
        
        self.request_times.append(record)
    
    def record_prediction(self, prediction: int, probability: float, model_version: str):
        """Enregistrer une prédiction"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "probability": probability,
            "model_version": model_version
        }
        self.predictions_cache.append(record)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques"""
        if not self.request_times:
            return {
                "total_requests": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "uptime_seconds": 0
            }
        
        latencies = [r["latency_ms"] for r in self.request_times]
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_requests if self.total_requests > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 0 else 0,
            "uptime_seconds": uptime,
            "requests_per_minute": (self.total_requests / uptime * 60) if uptime > 0 else 0
        }
    
    def get_prediction_distribution(self) -> Dict[str, Any]:
        """Obtenir la distribution des prédictions"""
        if not self.predictions_cache:
            return {"class_0": 0, "class_1": 0, "avg_confidence": 0.0}
        
        class_0_count = sum(1 for p in self.predictions_cache if p["prediction"] == 0)
        class_1_count = sum(1 for p in self.predictions_cache if p["prediction"] == 1)
        avg_conf = sum(p["probability"] for p in self.predictions_cache) / len(self.predictions_cache)
        
        return {
            "class_0_count": class_0_count,
            "class_1_count": class_1_count,
            "class_0_ratio": class_0_count / len(self.predictions_cache),
            "class_1_ratio": class_1_count / len(self.predictions_cache),
            "avg_confidence": avg_conf
        }
    
    def save_metrics(self, output_file="metrics/monitoring.json"):
        """Sauvegarder les métriques"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "performance": self.get_stats(),
            "predictions": self.get_prediction_distribution(),
            "recent_requests": list(self.request_times)[-10:]  # Dernières 10 requêtes
        }
        
        with open(output_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics saved to {output_file}")
    
    def print_stats(self):
        """Afficher les statistiques"""
        stats = self.get_stats()
        pred_dist = self.get_prediction_distribution()
        
        logger.info("\n" + "="*60)
        logger.info("📊 MONITORING STATS")
        logger.info("="*60)
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Error Count: {stats['error_count']}")
        logger.info(f"Error Rate: {stats['error_rate']*100:.2f}%")
        logger.info(f"Avg Latency: {stats['avg_latency_ms']:.2f}ms")
        logger.info(f"Min Latency: {stats['min_latency_ms']:.2f}ms")
        logger.info(f"Max Latency: {stats['max_latency_ms']:.2f}ms")
        logger.info(f"P95 Latency: {stats['p95_latency_ms']:.2f}ms")
        logger.info(f"P99 Latency: {stats['p99_latency_ms']:.2f}ms")
        logger.info(f"Uptime: {stats['uptime_seconds']:.1f}s")
        logger.info(f"Requests/min: {stats['requests_per_minute']:.1f}")
        
        if pred_dist['class_0_count'] + pred_dist['class_1_count'] > 0:
            logger.info(f"\nPrediction Distribution:")
            logger.info(f"  Class 0 (Bénin): {pred_dist['class_0_ratio']*100:.1f}%")
            logger.info(f"  Class 1 (Malin): {pred_dist['class_1_ratio']*100:.1f}%")
            logger.info(f"  Avg Confidence: {pred_dist['avg_confidence']:.3f}")
        
        logger.info("="*60 + "\n")

# Instance globale
monitor = MonitoringManager()