"""
Chargement et préprocessing du dataset UCI Breast Cancer
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir="data/raw", seed=42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
    def download_and_save(self):
        """Télécharger le dataset et le sauvegarder"""
        logger.info("Téléchargement du dataset UCI Breast Cancer...")
        
        # Charger depuis sklearn
        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = pd.Series(cancer.target, name='target')
        
        # Sauvegarder en CSV
        X.to_csv(self.data_dir / 'X.csv', index=False)
        y.to_csv(self.data_dir / 'y.csv', index=False)
        
        logger.info(f"Dataset sauvegardé dans {self.data_dir}")
        logger.info(f"Shape : X={X.shape}, y={y.shape}")
        logger.info(f"Classes : {y.value_counts().to_dict()}")
        
        return X, y
    
    def load_raw(self):
        """Charger les données brutes"""
        X = pd.read_csv(self.data_dir / 'X.csv')
        y = pd.read_csv(self.data_dir / 'y.csv').squeeze()
        return X, y
    
    def preprocess(self, X_train, X_test, scaler=None):
        """Normaliser les features"""
        if scaler is None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)
        
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def split_train_test(self, X, y, test_size=0.2):
        """Diviser train/test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def create_train_test_sets(self, output_dir="data/processed"):
        """Pipeline complet : load → split → scale → save"""
        logger.info("Démarrage du preprocessing...")
        
        X, y = self.load_raw()
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        X_train_scaled, X_test_scaled, scaler = self.preprocess(X_train, X_test)
        
        # Sauvegarder
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'X_train.npy', X_train_scaled)
        np.save(output_path / 'X_test.npy', X_test_scaled)
        np.save(output_path / 'y_train.npy', y_train.values)
        np.save(output_path / 'y_test.npy', y_test.values)
        
        # Sauvegarder le scaler
        import joblib
        joblib.dump(scaler, output_path / 'scaler.pkl')
        
        logger.info(f"Données sauvegardées dans {output_path}")
        logger.info(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    
    # Télécharger et sauvegarder
    loader.download_and_save()
    
    # Préprocessing
    loader.create_train_test_sets()