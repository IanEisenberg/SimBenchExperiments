"""Pipeline — composes a Predictor and a Calibrator into one prediction step.

    pipeline = Pipeline(ZeroShotPredictor(client))          # default: no calibration
    pipeline = Pipeline(ZeroShotPredictor(client), TempScaling())  # swap a stage

The pipeline is the single object the evaluation harness consumes, so the rest
of the analysis chain never needs to know which simulation or calibration is
inside. Batch prediction is threaded because LLM calls are IO-bound; the
on-disk cache makes re-runs free regardless of concurrency.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from .calibrate import Calibrator, IdentityCalibrator
from .data import SimBenchRecord
from .predict import Predictor


class Pipeline:
    def __init__(
        self,
        predictor: Predictor,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.predictor = predictor
        self.calibrator = calibrator or IdentityCalibrator()

    @property
    def name(self) -> str:
        return f"{self.predictor.name}+{self.calibrator.name}"

    def predict(self, record: SimBenchRecord) -> dict[str, float]:
        raw = self.predictor.predict(record)
        return self.calibrator.transform(record, raw)

    def predict_batch(
        self,
        records: Sequence[SimBenchRecord],
        max_workers: int = 8,
        progress: bool = True,
    ) -> list[dict[str, float]]:
        """Predict for many records concurrently, preserving input order."""
        results: list[dict[str, float] | None] = [None] * len(records)

        def _one(i_rec):
            i, rec = i_rec
            return i, self.predict(rec)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            it = ex.map(_one, enumerate(records))
            if progress:
                it = tqdm(it, total=len(records), desc=self.name)
            for i, pred in it:
                results[i] = pred
        return [r if r is not None else {} for r in results]
