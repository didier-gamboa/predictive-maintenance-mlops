# Short Tech Report — Predictive Maintenance MLOps

## Results
- Implemented a reproducible **end-to-end training pipeline** using Airflow, covering data ingestion, feature engineering, labeling, splitting, training, and evaluation.  
- Integrated **MLflow** for experiment tracking and model artifact storage, enabling reproducibility and comparison of models.  
- Built a working **Docker Compose stack** that launches Airflow and MLflow services locally. Initial prototyping for the API and dashboard is included.  

## Key Decisions
- Chose **multiclass classification** (failure type per component) instead of binary classification to provide more actionable insights, while retaining a binary risk signal if needed.  
- Adopted **time-aware data splitting with a gap** to prevent leakage from rolling features.  
- Selected **HistGradientBoosting and XGBoost** as baseline models for structured tabular data.  
- Prioritized reproducibility and maintainability with YAML-driven configs and containerization.  

## Lessons Learned
- The design of time splits and rolling windows is critical in predictive maintenance problems to avoid overly optimistic metrics.  
- Integration of orchestration (Airflow) with tracking (MLflow) adds some setup complexity, but ensures a strong MLOps foundation.  
- Given the limited time for this task, trade-offs were necessary: the API and Streamlit dashboard are scaffolded but not yet fully functional. Focusing first on pipeline reproducibility and documentation ensured a working foundation.  

## Next Steps
- Complete the **Flask API** and **Streamlit dashboard** for serving and visualization.  
- Extend the **CI/CD pipeline** to automatically run tests, build/push Docker images, and optionally promote models via MLflow registry.  
- Add **monitoring capabilities** (data drift detection, performance backtesting) to track model health post-deployment.  
- Explore advanced models (CatBoost, LightGBM) for improved performance.  
