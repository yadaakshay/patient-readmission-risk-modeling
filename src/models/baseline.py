"""
Baseline dummy models for establishing minimum clinical performance benchmarks.
"""
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from src.features import ClinicalFeatureEngineer


def build_dummy_baseline_pipeline(strategy: str = "prior", random_state: int = 42) -> Pipeline:
    """
    Constructs a naive baseline pipeline using scikit-learn DummyClassifier.
    
    Strategies:
    - 'prior': Always predicts class prior distribution (majority probability).
    - 'most_frequent': Always predicts the majority class (0 / not readmitted).
    - 'stratified': Generates predictions by respecting the training set class distribution.
    """
    return Pipeline(steps=[
        ('features', ClinicalFeatureEngineer()),
        ('classifier', DummyClassifier(strategy=strategy, random_state=random_state))
    ])
