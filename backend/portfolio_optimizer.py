"""
ShadowFlow Intelligence — Gemini portfolio construction layer.

Turns the quant engine's shadow-flow metrics into three allocation tiers using Gemini
structured output, then verifies every tier (schema, tickers, price floors, and weights
summing to 100%) before anything reaches the client. Rejected generations are sent back
to the model with the specific violations and retried.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger("shadowflow.portfolio")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# gemini-2.5-flash is closed to new API users; override with SHADOWFLOW_GEMINI_MODEL if your key still has it.
GEMINI_MODEL = os.getenv("SHADOWFLOW_GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_TEMPERATURE = 0.15
GEMINI_TIMEOUT_MS = int(os.getenv("SHADOWFLOW_GEMINI_TIMEOUT_MS", "90000"))
MAX_GENERATION_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

WEIGHT_SUM_TARGET = 100.0
WEIGHT_SUM_TOLERANCE = 0.01     # percentage points; absorbs float representation error only
FLOOR_PRICE_TOLERANCE = 0.02    # reported floor must sit within 2% of the engine's floor
MIN_ASSETS_PER_TIER, MAX_ASSETS_PER_TIER = 2, 6
TIER_NAMES = ("high_conviction_whale_growth", "defensive_liquidity_shelter", "balanced_dark_flow")


# --------------------------------------------------------------------------------------
# Structured output schema (sent to Gemini as response_schema)
# --------------------------------------------------------------------------------------

class SelectedAsset(BaseModel):
    ticker: str = Field(description="Ticker symbol copied verbatim from the supplied dataset.")
    accumulation_price_floor: float = Field(
        description="The asset's price_floor_reference value copied exactly from the dataset."
    )
    institutional_sentiment: Literal["Accumulation", "Distribution"] = Field(
        description="Net institutional posture implied by the off-exchange order flow."
    )
    allocation_weight_pct: float = Field(
        description="Percent of this tier's capital, two decimals. Weights within one tier sum to exactly 100."
    )
    order_flow_rationale: str = Field(
        description="Microstructural justification citing the dataset's off-exchange and risk metrics."
    )

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class PortfolioProfile(BaseModel):
    assets: list[SelectedAsset] = Field(
        description=f"{MIN_ASSETS_PER_TIER}-{MAX_ASSETS_PER_TIER} distinct assets selected for this tier."
    )


class DarkPoolAnalysisResponse(BaseModel):
    high_conviction_whale_growth: PortfolioProfile
    defensive_liquidity_shelter: PortfolioProfile
    balanced_dark_flow: PortfolioProfile


class OptimizePortfoliosResponse(BaseModel):
    generated_at: str
    model: str
    attempts: int
    metrics_as_of: str | None
    metrics_generated_at: str | None
    tier_weight_totals: dict[str, float]
    data_provenance: dict[str, str]
    portfolios: DarkPoolAnalysisResponse


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------

class GeminiConfigurationError(RuntimeError):
    """The Gemini client cannot be created (e.g. missing API key)."""


class GeminiUpstreamError(RuntimeError):
    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class PortfolioValidationError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues))
        self.issues = issues


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def _finite(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def reference_floors(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Engine-side price floor per ticker: last accumulation cluster price, else last close."""
    floors: dict[str, float | None] = {}
    for target in metrics.get("targets", []):
        floor = _finite((target.get("dark_pool") or {}).get("accumulation_floor_price"))
        floors[str(target["ticker"]).upper()] = floor if floor is not None else _finite(target.get("last_close"))
    return floors


def tier_weight_total(profile: PortfolioProfile) -> float:
    return math.fsum(asset.allocation_weight_pct for asset in profile.assets)


def validate_portfolios(analysis: DarkPoolAnalysisResponse, floors: dict[str, float | None]) -> None:
    issues: list[str] = []
    for tier in TIER_NAMES:
        assets = getattr(analysis, tier).assets
        if not MIN_ASSETS_PER_TIER <= len(assets) <= MAX_ASSETS_PER_TIER:
            issues.append(f"{tier}: has {len(assets)} assets, must have {MIN_ASSETS_PER_TIER}-{MAX_ASSETS_PER_TIER}")

        seen: set[str] = set()
        for asset in assets:
            label = f"{tier}.{asset.ticker or '<blank>'}"
            if asset.ticker in seen:
                issues.append(f"{label}: duplicate ticker within tier")
            seen.add(asset.ticker)

            if asset.ticker not in floors:
                issues.append(f"{label}: ticker not in the supplied dataset")

            weight = _finite(asset.allocation_weight_pct)
            if weight is None or weight <= 0 or weight > WEIGHT_SUM_TARGET:
                issues.append(f"{label}: allocation_weight_pct {asset.allocation_weight_pct} must be in (0, 100]")

            floor = _finite(asset.accumulation_price_floor)
            expected = floors.get(asset.ticker)
            if floor is None or floor <= 0:
                issues.append(f"{label}: accumulation_price_floor must be a positive number")
            elif expected and abs(floor / expected - 1.0) > FLOOR_PRICE_TOLERANCE:
                issues.append(f"{label}: accumulation_price_floor {floor} deviates from dataset value {expected}")

            if not asset.order_flow_rationale.strip():
                issues.append(f"{label}: order_flow_rationale is empty")

        total = tier_weight_total(getattr(analysis, tier))
        if not math.isfinite(total) or abs(total - WEIGHT_SUM_TARGET) > WEIGHT_SUM_TOLERANCE:
            issues.append(f"{tier}: weights sum to {total:.4f}%, must equal exactly {WEIGHT_SUM_TARGET:.2f}%")

    if issues:
        raise PortfolioValidationError(issues)


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------

SYSTEM_INSTRUCTION = f"""\
You are the head of market-microstructure research at an institutional asset manager.
You convert off-exchange (dark pool) order-flow analytics into model allocations.

Hard rules, all mandatory:
1. Select only tickers present in DATASET. Never introduce outside symbols.
2. Each tier holds {MIN_ASSETS_PER_TIER}-{MAX_ASSETS_PER_TIER} distinct tickers. A ticker may appear in several tiers.
3. allocation_weight_pct uses two decimals and the weights inside EACH tier sum to exactly 100.00.
   Verify the arithmetic before answering; adjust the largest weight if rounding leaves a residual.
4. accumulation_price_floor must equal that ticker's price_floor_reference from DATASET exactly.
5. institutional_sentiment is "Accumulation" when recent off-exchange clusters show absorption at the
   floor (high recency, elevated whale score); otherwise "Distribution".
6. order_flow_rationale (2-3 sentences) must cite concrete DATASET figures: off-exchange share,
   cluster count/length, recency, whale score components, volatility, or Sharpe. Use only DATASET
   facts; do not reference news, earnings, or any information not supplied.

Tier mandates:
- high_conviction_whale_growth: maximize exposure to the strongest, most recent accumulation clusters
  (whale score, persistence, recency). Higher volatility is acceptable; concentrate weight.
- defensive_liquidity_shelter: prioritize low annualized volatility and positive Sharpe where
  off-exchange liquidity is steady. Favor diversification; avoid weights above 40.
- balanced_dark_flow: blend accumulation strength with risk-adjusted return, sizing positions
  roughly by conviction_score while keeping volatility moderate.
"""


def _dataset_row(target: dict[str, Any], floor: float | None) -> dict[str, Any]:
    dark_pool = target.get("dark_pool") or {}
    return {
        "ticker": target.get("ticker"),
        "signal": target.get("signal"),
        "conviction_score": target.get("conviction_score"),
        "whale_accumulation_cluster_score": target.get("whale_accumulation_cluster_score"),
        "score_components": target.get("score_components"),
        "last_close": target.get("last_close"),
        "price_floor_reference": floor,
        "cagr": target.get("cagr"),
        "annualized_volatility": target.get("annualized_volatility"),
        "sharpe_ratio": target.get("sharpe_ratio"),
        "avg_off_exchange_share": dark_pool.get("avg_off_exchange_share"),
        "anomalous_days": dark_pool.get("anomalous_days"),
        "cluster_count": dark_pool.get("cluster_count"),
        "longest_cluster_days": dark_pool.get("longest_cluster_days"),
        "sessions_since_last_anomaly": dark_pool.get("sessions_since_last_anomaly"),
        "recent_clusters": dark_pool.get("recent_clusters"),
    }


def build_prompt(metrics: dict[str, Any], floors: dict[str, float | None]) -> str:
    dataset = [
        _dataset_row(t, floors.get(str(t.get("ticker")).upper()))
        for t in metrics.get("targets", [])
    ]
    context = {
        "as_of": metrics.get("as_of"),
        "lookback": metrics.get("lookback"),
        "risk_free_rate": metrics.get("risk_free_rate"),
        "off_exchange_threshold": metrics.get("off_exchange_threshold"),
        "field_units": {
            "cagr / annualized_volatility / avg_off_exchange_share": "decimal fractions (0.25 = 25%)",
            "whale_accumulation_cluster_score / conviction_score": "0-100",
            "score_components": "0-1 each",
        },
    }
    return (
        "Construct the three portfolio tiers from the institutional order-flow dataset below.\n\n"
        f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        f"DATASET:\n{json.dumps(dataset, indent=2)}\n\n"
        "Return only the JSON object defined by the response schema."
    )


def _correction_prompt(issues: list[str]) -> str:
    bullet_list = "\n".join(f"- {issue}" for issue in issues)
    return (
        "Your previous answer was rejected by the risk-control validator:\n"
        f"{bullet_list}\n\n"
        "Return a corrected, complete JSON object that satisfies every hard rule. "
        "Recompute each tier's weights so they sum to exactly 100.00."
    )


# --------------------------------------------------------------------------------------
# Gemini client
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise GeminiConfigurationError("GEMINI_API_KEY is not set")
    return genai.Client(http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))


def _generation_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=DarkPoolAnalysisResponse,
        temperature=GEMINI_TEMPERATURE,
    )


def _finish_reason(response: types.GenerateContentResponse) -> str:
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return f"prompt blocked: {response.prompt_feedback.block_reason}"
    if response.candidates and response.candidates[0].finish_reason:
        return str(response.candidates[0].finish_reason)
    return "unknown"


async def generate_portfolios(metrics: dict[str, Any]) -> tuple[DarkPoolAnalysisResponse, int]:
    """Call Gemini until it returns a portfolio set that passes validation. Returns (result, attempts)."""
    floors = reference_floors(metrics)
    if not floors:
        raise PortfolioValidationError(["quant engine returned no targets to allocate"])

    client = get_gemini_client()
    config = _generation_config()
    prompt_turn = types.Content(role="user", parts=[types.Part.from_text(text=build_prompt(metrics, floors))])
    contents: list[types.Content] = [prompt_turn]
    issues: list[str] = []

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            response = await client.aio.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
        except errors.ClientError as exc:
            if exc.code == 429 and attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Gemini rate limited (attempt %d); backing off", attempt)
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise GeminiUpstreamError(exc.code, f"Gemini rejected the request: {exc.message}") from exc
        except errors.ServerError as exc:
            if attempt < MAX_GENERATION_ATTEMPTS:
                logger.warning("Gemini server error %s (attempt %d); retrying", exc.code, attempt)
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise GeminiUpstreamError(exc.code, f"Gemini server error: {exc.message}") from exc
        except errors.APIError as exc:
            raise GeminiUpstreamError(exc.code, f"Gemini API error: {exc.message}") from exc

        raw = response.text
        try:
            if not raw:
                raise PortfolioValidationError([f"empty response ({_finish_reason(response)})"])
            analysis = DarkPoolAnalysisResponse.model_validate_json(raw)
            validate_portfolios(analysis, floors)
            return analysis, attempt
        except ValidationError as exc:
            issues = [f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}" for err in exc.errors()]
        except PortfolioValidationError as exc:
            issues = exc.issues

        logger.warning("Gemini output rejected (attempt %d/%d): %s", attempt, MAX_GENERATION_ATTEMPTS, issues)
        contents = [prompt_turn]
        if raw:
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=raw)]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=_correction_prompt(issues))]))

    raise PortfolioValidationError(issues)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

class PortfolioOptimizer:
    """Caches one Gemini result per quant-engine snapshot so page loads don't re-bill the model."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cached: tuple[str | None, OptimizePortfoliosResponse] | None = None

    async def optimize(self, metrics: dict[str, Any], force_refresh: bool = False) -> OptimizePortfoliosResponse:
        snapshot = metrics.get("generated_at")
        async with self._lock:
            if not force_refresh and self._cached and self._cached[0] == snapshot:
                return self._cached[1]

            analysis, attempts = await generate_portfolios(metrics)
            result = OptimizePortfoliosResponse(
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model=GEMINI_MODEL,
                attempts=attempts,
                metrics_as_of=metrics.get("as_of"),
                metrics_generated_at=snapshot,
                tier_weight_totals={t: round(tier_weight_total(getattr(analysis, t)), 4) for t in TIER_NAMES},
                data_provenance={
                    **(metrics.get("data_provenance") or {}),
                    "portfolio_construction": (
                        f"Generated by {GEMINI_MODEL} from the metrics above and validated for schema, "
                        "tickers, price floors and 100% tier weights. Not investment advice."
                    ),
                },
                portfolios=analysis,
            )
            self._cached = (snapshot, result)
            return result


optimizer = PortfolioOptimizer()
router = APIRouter(prefix="/api", tags=["portfolio"])


@router.get("/optimize-portfolios", response_model=OptimizePortfoliosResponse)
async def optimize_portfolios(
    request: Request,
    refresh: bool = Query(False, description="Regenerate with Gemini even if this metrics snapshot is cached."),
) -> OptimizePortfoliosResponse:
    engine = getattr(request.app.state, "shadowflow_engine", None)
    if engine is None:
        raise HTTPException(status_code=500, detail="Quant engine is not registered on app.state")

    try:
        metrics = (await engine.get_metrics()).model_dump(mode="json")
    except Exception as exc:
        logger.exception("Quant engine failed while preparing portfolio optimization")
        raise HTTPException(status_code=503, detail=f"Quant engine unavailable: {exc}") from exc

    try:
        return await optimizer.optimize(metrics, force_refresh=refresh)
    except GeminiConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiUpstreamError as exc:
        status = 429 if exc.status_code == 429 else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except PortfolioValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"Gemini output failed validation after {MAX_GENERATION_ATTEMPTS} attempts",
                "issues": exc.issues,
            },
        ) from exc
    except Exception as exc:
        logger.exception("Portfolio optimization failed")
        raise HTTPException(status_code=500, detail="Portfolio optimization failed") from exc
