"""Standalone CLI adapter for train/predict workflows.

Creates a cyclopts CLI app from user-provided train/predict functions.
No config class needed — the user bakes configuration into closures.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import cyclopts
import pandas as pd  # type: ignore[import-untyped]
from typing import Annotated


def create_cli_app(
    train_fn: Callable[[pd.DataFrame], Any],
    predict_fn: Callable[[Any, pd.DataFrame, pd.DataFrame], pd.DataFrame],
    name: str = "multistep-model",
) -> cyclopts.App:
    """Create a cyclopts CLI app with train-cmd and predict-cmd subcommands.

    Args:
        train_fn: Training function: (data: DataFrame) -> model.
        predict_fn: Prediction function: (model, historic, future) -> predictions DataFrame.
        name: Name of the CLI application.

    Returns:
        Configured cyclopts.App.

    Example:
        >>> app = create_cli_app(my_train, my_predict, name="my-model")
        >>> app()  # Run from command line
    """
    app = cyclopts.App(name=name, help=f"{name} CLI")

    @app.command
    def train_cmd(
        train_data: Annotated[Path, cyclopts.Parameter(help="Path to training data CSV")],
        model_output: Annotated[Path, cyclopts.Parameter(help="Path to save trained model (pickle)")],
    ) -> None:
        """Train the model from CSV data."""
        data = pd.read_csv(train_data)
        model = train_fn(data)

        with open(model_output, "wb") as f:
            pickle.dump(model, f)

        print(f"Model trained and saved to {model_output}")

    @app.command
    def predict_cmd(
        model_path: Annotated[Path, cyclopts.Parameter(help="Path to trained model (pickle)")],
        historic_data: Annotated[Path, cyclopts.Parameter(help="Path to historic data CSV")],
        future_data: Annotated[Path, cyclopts.Parameter(help="Path to future periods CSV")],
        output: Annotated[Path, cyclopts.Parameter(help="Path to save predictions CSV")],
    ) -> None:
        """Generate predictions from a trained model."""
        with open(model_path, "rb") as f:
            model: Any = pickle.load(f)

        historic = pd.read_csv(historic_data)
        future = pd.read_csv(future_data)

        predictions = predict_fn(model, historic, future)

        # Expand nested samples column to wide format if present
        if "samples" in predictions.columns:
            samples_list = predictions["samples"].tolist()
            predictions = predictions.drop(columns=["samples"])
            if samples_list:
                import pandas as pd_inner
                samples_df = pd_inner.DataFrame(
                    samples_list,
                    columns=[f"sample_{i}" for i in range(len(samples_list[0]))],
                    index=predictions.index,
                )
                predictions = pd.concat([predictions, samples_df], axis=1)

        predictions.to_csv(output, index=False)
        print(f"Predictions saved to {output}")

    return app
