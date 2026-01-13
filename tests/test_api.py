"""
Tests pour l'API FastAPI

EXÉCUTION:
    pytest tests/test_api.py -v

RÉSULTAT ATTENDU:
    - Tous les tests passent
    - Coverage > 70%
"""

import pytest
import requests
import json
from typing import List
import time

# Configuration
API_URL = "http://localhost:8000"

# Feature de test (30 features)
TEST_FEATURES = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
    0.05373, 0.01587, 0.03039, 0.006193, 25.38, 17.33, 184.6, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.11890
]


# ===== Fixtures =====

@pytest.fixture(scope="session")
def api_ready():
    """Vérifier que l'API est prête"""
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✓ API is ready")
                return True
        except:
            if i < max_retries - 1:
                time.sleep(1)
    
    print("✗ API not ready after 10 attempts")
    pytest.skip("API not running on port 8000")


# ===== Test Health & Info =====

class TestHealthAndInfo:
    """Tests pour health check et informations"""
    
    def test_health_check(self, api_ready):
        """Test health endpoint"""
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "model_version" in data
    
    def test_root_endpoint(self, api_ready):
        """Test root endpoint"""
        response = requests.get(f"{API_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_model_info(self, api_ready):
        """Test model info endpoint"""
        response = requests.get(f"{API_URL}/model-info")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        assert "model_type" in data
        assert data["n_features"] == 30
        assert "classes" in data


# ===== Test Predictions =====

class TestPredictions:
    """Tests pour les prédictions"""
    
    def test_single_prediction(self, api_ready):
        """Test single prediction"""
        payload = {"features": TEST_FEATURES}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "model_version" in data
        
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 1
    
    def test_batch_prediction(self, api_ready):
        """Test batch predictions"""
        payloads = [{"features": TEST_FEATURES} for _ in range(3)]
        response = requests.post(f"{API_URL}/batch-predict", json=payloads)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert data["count"] == 3
        assert len(data["predictions"]) == 3
    
    def test_invalid_input_wrong_size(self, api_ready):
        """Test invalid input (wrong number of features)"""
        payload = {"features": [1.0, 2.0, 3.0]}  # Seulement 3 features
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        # Devrait retourner 422 (validation error)
        assert response.status_code == 422
    
    def test_invalid_input_empty(self, api_ready):
        """Test empty input"""
        payload = {"features": []}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        assert response.status_code == 422
    
    def test_prediction_output_format(self, api_ready):
        """Test que le format de sortie est correct"""
        payload = {"features": TEST_FEATURES}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Vérifier les types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability"], float)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["model_version"], str)


# ===== Test Performance =====

class TestPerformance:
    """Tests de performance"""
    
    def test_prediction_latency(self, api_ready):
        """Test latency de prédiction"""
        payload = {"features": TEST_FEATURES}
        
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json=payload)
        latency = time.time() - start
        
        assert response.status_code == 200
        assert latency < 3.0, f"Latency {latency:.3f}s exceeds 3.0s"
        print(f"Latency: {latency*1000:.2f}ms")
    
    def test_batch_prediction_latency(self, api_ready):
        """Test batch prediction latency"""
        payloads = [{"features": TEST_FEATURES} for _ in range(10)]
        
        start = time.time()
        response = requests.post(f"{API_URL}/batch-predict", json=payloads)
        latency = time.time() - start
        
        assert response.status_code == 200
        assert latency < 5.0, f"Latency {latency:.3f}s exceeds 5.0s"
        print(f"Batch latency (10 samples): {latency*1000:.2f}ms")


# ===== Test Consistency =====

class TestConsistency:
    """Tests de cohérence"""
    
    def test_same_input_same_output(self, api_ready):
        """Test que le même input donne le même output"""
        payload = {"features": TEST_FEATURES}
        
        response1 = requests.post(f"{API_URL}/predict", json=payload)
        response2 = requests.post(f"{API_URL}/predict", json=payload)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["prediction"] == data2["prediction"]
        assert abs(data1["probability"] - data2["probability"]) < 0.0001
        assert abs(data1["confidence"] - data2["confidence"]) < 0.0001
    
    def test_prediction_bounds(self, api_ready):
        """Test que les prédictions respectent les bounds"""
        payload = {"features": TEST_FEATURES}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Vérifier les bounds
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 1
        assert data["prediction"] in [0, 1]


# ===== Test Error Handling =====

class TestErrorHandling:
    """Tests pour la gestion d'erreurs"""
    
    def test_invalid_json(self, api_ready):
        """Test invalid JSON"""
        response = requests.post(
            f"{API_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_features(self, api_ready):
        """Test missing features field"""
        payload = {}
        response = requests.post(f"{API_URL}/predict", json=payload)
        assert response.status_code == 422
    
    def test_wrong_field_name(self, api_ready):
        """Test wrong field name"""
        payload = {"data": TEST_FEATURES}
        response = requests.post(f"{API_URL}/predict", json=payload)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])