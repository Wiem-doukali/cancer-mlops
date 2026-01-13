#!/usr/bin/env python3
"""
Script de test du monitoring

EXÉCUTION:
    python scripts/test_monitoring.py
"""

import requests
import time

API_URL = "http://localhost:8000"
features = [17.99, 10.38, 122.8] + [0.0]*27

print("=== MONITORING TEST ===\n")

# Test 1 : Health check
print("1️⃣  Testing health check...")
response = requests.get(f"{API_URL}/health")
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}\n")

# Test 2 : 10 prédictions
print("2️⃣  Testing 10 predictions...")
latencies = []

for i in range(10):
    start = time.time()
    response = requests.post(f"{API_URL}/predict", json={"features": features})
    latency = time.time() - start
    latencies.append(latency)
    
    pred = response.json()["prediction"]
    print(f"   Request {i+1}: Status {response.status_code} - Prediction: {pred} - Latency: {latency*1000:.2f}ms")

print(f"\n   Average latency: {sum(latencies)/len(latencies)*1000:.2f}ms")

# Test 3 : Monitoring endpoint
print("\n3️⃣  Testing monitoring endpoint...")
response = requests.get(f"{API_URL}/monitoring/stats")
if response.status_code == 200:
    data = response.json()
    print(f"   Status: {response.status_code}")
    for key, val in data.items():
        print(f"      {key}: {val}")
else:
    print(f"   ⚠️  Endpoint not available (status: {response.status_code})")

print("\n✅ MONITORING TEST COMPLETED")