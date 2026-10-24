from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import MODEL_BUNDLE_PATH


class PredictionResult(BaseModel):
    """Standardized output contract for single-customer churn risk inference."""

    customer_id: str = Field(..., description="Unique customer identifier")
    churn_probability: float = Field(..., description="Continuous churn probability score")
    churn_probability_pct: str = Field(..., description="Formatted probability percentage")
    is_high_risk: bool = Field(..., description="Flag determining cost-optimal retention intervention")
    risk_tier: str = Field(..., description="Human-readable business risk tier")
    threshold_applied: float = Field(..., description="Cutoff threshold applied for high-risk flag")


class ChurnPredictor:
    """Production ML Inference Service for Telco Churn.

    Loads the serialized model bundle, handles feature hygiene and encoding transforms,
    and applies the cost-optimal business decision threshold (t* = 0.17) and risk tiers.
    """

    DEFAULT_RISK_TIERS = {
        "bins": [0.0, 0.50, 0.80, 0.95, 1.0],
        "labels": ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
    }

    def __init__(self, bundle_path: Optional[Union[str, Path]] = None):
        self.bundle_path = Path(bundle_path or MODEL_BUNDLE_PATH)
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle artifact not found at '{self.bundle_path}'. "
                "Ensure artifacts/telco_churn_model_bundle.joblib is present."
            )

        payload = joblib.load(self.bundle_path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid model bundle format in '{self.bundle_path}'. Expected dictionary."
            )

        self.model_name: str = payload.get("model_name", "LogisticRegression")
        self.model: Any = payload.get("classifier")
        if self.model is None or not hasattr(self.model, "predict_proba"):
            raise ValueError("Model bundle must contain a valid 'classifier' with predict_proba().")

        self.feature_cols: List[str] = payload.get("feature_cols", [])
        self.numeric_cols: List[str] = payload.get(
            "numeric_cols", ["tenure", "MonthlyCharges", "TotalCharges"]
        )
        self.categorical_cols: List[str] = payload.get(
            "categorical_cols",
            [
                "MultipleLines",
                "InternetService",
                "OnlineBackup",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaymentMethod",
            ],
        )
        self.boolean_cols: List[str] = payload.get(
            "boolean_cols", ["SeniorCitizen", "Dependents", "PaperlessBilling"]
        )
        self.thresholds: Dict[str, float] = payload.get(
            "thresholds", {"t_star": 0.17, "t_default": 0.50}
        )
        self.t_star: float = float(self.thresholds.get("t_star", 0.17))
        self.t_default: float = float(self.thresholds.get("t_default", 0.50))
        self.cost_matrix: Dict[str, float] = payload.get(
            "cost_matrix", {"C_FN": 700.0, "C_FP": 150.0}
        )
        self.risk_tiers: Dict[str, Any] = payload.get("risk_tiers", self.DEFAULT_RISK_TIERS)

        # Frozen bin edges and WoE encodings for the linear model representation
        self.bin_edges: Dict[str, List[float]] = payload.get(
            "bin_edges",
            {
                "tenure": [-np.inf, 1.5, 5.5, 10.5, 15.5, 22.5, 43.5, 53.5, 59.5, 70.5, np.inf],
                "MonthlyCharges": [
                    -np.inf, 19.725, 23.1, 27.625, 68.45, 78.925, 93.525, 97.6, 101.475, 106.175, np.inf,
                ],
                "TotalCharges": [
                    -np.inf, 108.95, 225.625, 456.175, 680.625, 1240.525, 3326.625, 4838.375, 5644.6, 6987.55, np.inf,
                ],
            },
        )
        self.woe_maps: Dict[str, Dict[str, float]] = payload.get(
            "woe_maps",
            {
                "MultipleLines": {
                    "No": -0.077413, "No phone service": -0.165800, "Yes": 0.120037
                },
                "InternetService": {
                    "DSL": -0.420082, "Fiber optic": 0.688874, "No": -1.573646
                },
                "OnlineBackup": {
                    "No": 0.624564, "No internet service": -1.573646, "Yes": -0.272146
                },
                "StreamingTV": {
                    "No": 0.335943, "No internet service": -1.573646, "Yes": 0.187360
                },
                "StreamingMovies": {
                    "No": 0.339750, "No internet service": -1.573646, "Yes": 0.182546
                },
                "Contract": {
                    "Month-to-month": 0.730436, "One year": -1.076405, "Two year": -2.438046
                },
                "PaymentMethod": {
                    "Bank transfer (automatic)": -0.616373,
                    "Credit card (automatic)": -0.737017,
                    "Electronic check": 0.843092,
                    "Mailed check": -0.398479,
                },
            },
        )

    def assign_risk_tier(self, probability: float) -> str:
        """Categorizes continuous probability into business risk tiers:

        - 0.00 <= P < 0.50: Low Risk
        - 0.50 <= P < 0.80: Moderate Risk
        - 0.80 <= P < 0.95: High Risk
        - 0.95 <= P <= 1.00: Critical Risk
        """
        p = max(0.0, min(1.0, float(probability)))
        if p < 0.50:
            return "Low Risk"
        elif p < 0.80:
            return "Moderate Risk"
        elif p < 0.95:
            return "High Risk"
        else:
            return "Critical Risk"

    def _transform_features_for_model(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw input features into the exact matrix shape expected by the classifier."""
        expected_cols = getattr(self.model, "feature_names_in_", None)
        if expected_cols is None or len(expected_cols) == len(self.feature_cols):
            # Model accepts raw features directly
            return df_features

        # 1. Numerical discretization -> One-Hot binned indicators
        num_dummies = []
        for col in self.numeric_cols:
            if col in df_features.columns and col in self.bin_edges:
                binned = pd.cut(df_features[col], bins=self.bin_edges[col])
                for feat in expected_cols:
                    if feat.startswith(f"{col}_"):
                        interval_str = feat[len(f"{col}_"):]
                        is_match = (binned.astype(str) == interval_str).astype(float)
                        num_dummies.append(pd.Series(is_match, name=feat, index=df_features.index))

        x_num = pd.concat(num_dummies, axis=1) if num_dummies else pd.DataFrame(index=df_features.index)

        # 2. Categoricals -> Weight of Evidence (WoE)
        x_cat = pd.DataFrame(index=df_features.index)
        for col in self.categorical_cols:
            if col in df_features.columns and col in self.woe_maps:
                x_cat[col] = df_features[col].astype(str).map(self.woe_maps[col]).fillna(0.0)
            else:
                x_cat[col] = 0.0

        # 3. Booleans -> Float (0.0 / 1.0)
        x_bool = pd.DataFrame(index=df_features.index)
        for col in self.boolean_cols:
            if col in df_features.columns:
                x_bool[col] = df_features[col].astype(float)
            else:
                x_bool[col] = 0.0

        # Align transformed feature arrays to the frozen feature column schema
        x_full = pd.concat([x_num, x_cat, x_bool], axis=1)
        return x_full.reindex(columns=expected_cols, fill_value=0.0)

    def _prepare_feature_frame(
        self, data: Union[Dict[str, Any], pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """Converts input dictionary or Series to DataFrame and strips non-feature columns."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data.to_dict()])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported data type for inference: {type(data)}")

        # Exclude synthetic CRM metadata, IDs, and ground truth labels
        metadata_cols = [
            "customerID", "CustomerID", "full_name", "email", "phone", "Churn", "churn"
        ]
        df_clean = df.drop(columns=[col for col in metadata_cols if col in df.columns], errors="ignore")

        # Guarantee all feature_cols exist in input
        for col in self.feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0

        # Align column positions with baseline training features
        df_ordered = df_clean[self.feature_cols].copy()

        # Transform to internal matrix representation expected by classifier
        return self._transform_features_for_model(df_ordered)

    def predict_proba(
        self, data: Union[Dict[str, Any], pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        """Estimates churn probability P(Churn = True) for the given input."""
        x_matrix = self._prepare_feature_frame(data)
        probas = self.model.predict_proba(x_matrix)
        # Extract positive class probability P(Churn=True)
        return probas[:, 1]

    def predict_customer(
        self,
        customer: Union[Dict[str, Any], pd.Series],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Inference for a single customer profile returning a structured, JSON-serializable dictionary.

        Scores churn probability, evaluates the cost-optimal business intervention threshold
        (t* = 0.17 by default), assigns risk tier, and validates output against PredictionResult.

        Args:
            customer (Union[Dict[str, Any], pd.Series]): Customer record containing demographic and account fields.
            threshold (Optional[float]): Custom decision cutoff. Defaults to cost-optimal t_star (0.17).

        Returns:
            Dict[str, Any]: Serialized PredictionResult dictionary with keys:
                - customer_id: Alphanumeric customer identifier.
                - churn_probability: Continuous probability score [0.0 - 1.0].
                - churn_probability_pct: Formatted percentage string.
                - is_high_risk: Boolean indicating whether retention intervention is recommended.
                - risk_tier: Assigned tier ('Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk').
                - threshold_applied: The numeric threshold cutoff applied.

        Raises:
            TypeError: If customer is not a dict or pandas Series.
            pydantic.ValidationError: If prediction output violates the PredictionResult schema.
        """
        # Extract customer identifier if present
        customer_id = "UNKNOWN"
        if isinstance(customer, dict):
            customer_id = customer.get("customerID") or customer.get("CustomerID") or "UNKNOWN"
        elif isinstance(customer, pd.Series):
            customer_id = customer.get("customerID", customer.get("CustomerID", "UNKNOWN"))

        applied_thr = float(threshold if threshold is not None else self.t_star)

        prob = float(self.predict_proba(customer)[0])
        prob_rounded = round(prob, 4)
        is_high_risk = bool(prob >= applied_thr)
        risk_tier = self.assign_risk_tier(prob)

        result = PredictionResult(
            customer_id=str(customer_id),
            churn_probability=prob_rounded,
            churn_probability_pct=f"{prob * 100:.2f}%",
            is_high_risk=is_high_risk,
            risk_tier=risk_tier,
            threshold_applied=applied_thr,
        )
        return result.model_dump()

    def predict_batch(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Batch scoring on a DataFrame of customer records, adding prediction and tier columns.

        Missing expected features are zero-filled automatically before model scoring.

        Args:
            df (pd.DataFrame): Customer DataFrame.
            threshold (Optional[float]): Decision cutoff. Defaults to self.t_star (0.17).

        Returns:
            pd.DataFrame: Original DataFrame enriched with columns:
                - Predicted_Churn_Prob: Continuous probability rounded to 4 decimals.
                - Churn_Probability_Pct: Human-readable probability percentage.
                - Is_High_Risk: Boolean flag whether probability exceeds threshold.
                - Risk_Tier: Segment tier label.
                - Threshold_Applied: The cutoff applied.

        Raises:
            TypeError: If input is not a pandas DataFrame.
        """
        applied_thr = float(threshold if threshold is not None else self.t_star)
        probas = self.predict_proba(df)

        df_out = df.copy()
        df_out["Predicted_Churn_Prob"] = np.round(probas, 4)
        df_out["Churn_Probability_Pct"] = [f"{p * 100:.2f}%" for p in probas]
        df_out["Is_High_Risk"] = probas >= applied_thr
        df_out["Risk_Tier"] = [self.assign_risk_tier(p) for p in probas]
        df_out["Threshold_Applied"] = applied_thr
        return df_out
