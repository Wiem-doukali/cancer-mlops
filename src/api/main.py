"""
API FastAPI pour l'inférence sur Cancer du Sein
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === Modèles Pydantic ===
class BreastCancerInput(BaseModel):
    """Input pour prédiction (30 features)"""
    features: List[float] = Field(..., min_items=30, max_items=30)
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [17.99, 10.38, 122.8] + [0.0] * 27
            }
        }

class PredictionOutput(BaseModel):
    """Output de prédiction"""
    prediction: int  # 0: bénin, 1: malin
    probability: float  # Probabilité classe 1
    confidence: float  # Confiance
    model_version: str

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str

# === FastAPI App ===
app = FastAPI(
    title="Cancer du Sein - API Inférence",
    description="Prédiction bénin/malin pour tumeurs mammaires",
    version="1.0.0"
)

# === État global ===
class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_version = "1.0.0"
        self.model_path = Path("models/rf_baseline.pkl")
        self.scaler_path = Path("data/processed/scaler.pkl")
    
    def load_model(self):
        """Charger le modèle et le scaler"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Modèle chargé depuis {self.model_path}")
            else:
                logger.warning(f"Modèle non trouvé : {self.model_path}")
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler chargé depuis {self.scaler_path}")
            else:
                logger.warning(f"Scaler non trouvé : {self.scaler_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle : {e}")
    
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Effectuer une prédiction"""
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        # Normaliser
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Prédire
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(proba[1]),
            "confidence": float(max(proba)),
            "model_version": self.model_version
        }

model_manager = ModelManager()

# === Startup Event ===
@app.on_event("startup")
async def startup():
    """Charger le modèle au démarrage"""
    model_manager.load_model()
    logger.info("API démarrée avec succès")

# === Routes ===

@app.get("/", tags=["Root"])
async def root():
    """Route racine"""
    return {"message": "API Inférence - Cancer du Sein", "version": "1.0.0"}

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Vérifier la santé de l'API"""
    return HealthCheck(
        status="healthy",
        model_loaded=model_manager.is_ready(),
        model_version=model_manager.model_version
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: BreastCancerInput):
    """
    Prédire si une tumeur est bénigne ou maligne
    
    - Bénin : 0
    - Malin : 1
    """
    try:
        result = model_manager.predict(input_data.features)
        return PredictionOutput(**result)
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(inputs: List[BreastCancerInput]):
    """Prédictions en batch"""
    try:
        results = []
        for input_data in inputs:
            result = model_manager.predict(input_data.features)
            results.append(result)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Erreur batch : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", tags=["Model"])
async def model_info():
    """Informations sur le modèle courant"""
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "version": model_manager.model_version,
        "model_type": type(model_manager.model).__name__,
        "scaler_type": type(model_manager.scaler).__name__,
        "n_features": 30,
        "classes": ["Bénin", "Malin"]
    }

# === Main ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")