"""
ShadowFlow Intelligence — institutional order-flow monitoring backend.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Data provenance:
    * Price and volume history is real (Yahoo Finance via yfinance, split/dividend adjusted).
    * The dark pool transaction tape is SYNTHETIC. It is produced by a seeded simulator
      layered on top of real volume so the frontend and scoring pipeline can be built
      before a licensed off-exchange feed (e.g. FINRA ATS / TRF data) is wired in.
      Nothing derived from it is a trading signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env before any module-level os.getenv() reads (including portfolio_optimizer).
# Real environment variables (e.g. set in Vercel) take precedence.
load_dotenv(Path(__file__).with_name(".env"))

import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from portfolio_optimizer import router as portfolio_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shadowflow")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

TICKERS: list[str] = [
    "NVDA", "AAPL", "PLTR", "MSFT", "AMD",
    "JNJ", "XOM", "O", "TSLA", "AMZN",
    "GOOGL", "META", "JPM", "AVGO", "COST",
]
LOOKBACK_PERIOD = "1y"
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = float(os.getenv("SHADOWFLOW_RISK_FREE_RATE", "0.045"))
CACHE_TTL_SECONDS = int(os.getenv("SHADOWFLOW_CACHE_TTL_SECONDS", "900"))
TOP_N_TARGETS = 8
FRONTEND_ORIGINS = [
    o.strip()
    for o in os.getenv("SHADOWFLOW_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

# Anomaly filter
OFF_EXCHANGE_THRESHOLD = 0.45   # off-exchange share of total volume that marks a spike
FLOOR_WINDOW = 20               # rolling-low lookback (trading days) defining the price floor
FLOOR_BAND = 0.02               # dark VWAP must print within 2% above the floor
UNIFORMITY_TOL = 0.004          # max coefficient of variation of block-print prices in a session
CLUSTER_MAX_GAP = 1             # quiet days tolerated inside a single accumulation cluster

# Simulator
EPISODE_FLOOR_PROXIMITY = 0.03  # accumulation episodes start when close is within 3% of the floor

# Whale Accumulation Cluster Score (0-100)
DENSITY_WINDOW = 60             # trading days used for anomaly density
DENSITY_SATURATION = 0.20       # 20% of recent sessions flagged -> full density credit
SHARE_CEILING = 0.70            # off-exchange share that earns full intensity credit
VOLUME_SATURATION = 1.5         # session volume / 20d avg that earns full volume credit
PERSISTENCE_SATURATION = 5      # longest cluster (days) that earns full persistence credit
RECENCY_HALF_LIFE = 10.0        # trading days
SCORE_WEIGHTS = {"density": 0.30, "intensity": 0.25, "persistence": 0.20, "recency": 0.25}

# Conviction blend
CONVICTION_WHALE_WEIGHT = 0.75
SHARPE_FLOOR, SHARPE_CEILING = -1.0, 3.0  # Sharpe mapped linearly onto [0, 1] across this range


class DataUnavailableError(RuntimeError):
    """Market data could not be retrieved for any ticker."""


class InsufficientDataError(ValueError):
    """A single ticker has too little usable history to analyze."""


# --------------------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------------------

def _finite(value: object) -> float | None:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(value: object, digits: int = 4) -> float | None:
    v = _finite(value)
    return None if v is None else round(v, digits)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _stable_seed(ticker: str, as_of: pd.Timestamp) -> int:
    # hashlib (not hash()) so the tape is reproducible across processes for a given data date.
    digest = hashlib.sha256(f"{ticker}|{as_of:%Y-%m-%d}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# --------------------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------------------

def download_history(tickers: list[str], period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (close, volume) frames indexed by session date with one column per ticker."""
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise DataUnavailableError("yfinance returned no data")

    if isinstance(raw.columns, pd.MultiIndex):
        close, volume = raw["Close"], raw["Volume"]
    else:  # single-ticker flat layout
        close = raw[["Close"]].set_axis([tickers[0]], axis=1)
        volume = raw[["Volume"]].set_axis([tickers[0]], axis=1)

    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        index = pd.DatetimeIndex(frame.index)
        frame.index = index.tz_localize(None) if index.tz is not None else index
        return frame.sort_index()

    close, volume = _clean(close), _clean(volume)
    if close.dropna(how="all").empty:
        raise DataUnavailableError("yfinance returned no usable closing prices")
    return close, volume


# --------------------------------------------------------------------------------------
# Quant math engine
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskMetrics:
    last_close: float | None
    total_return: float | None
    cagr: float | None
    annualized_volatility: float | None
    sharpe_ratio: float | None
    observations: int


def compute_risk_metrics(close: pd.Series, risk_free_rate: float) -> RiskMetrics:
    prices = close[close > 0]  # also drops NaN
    n = len(prices)
    if n < 2:
        return RiskMetrics(_finite(prices.iloc[-1]) if n else None, None, None, None, None, n)

    first, last = float(prices.iloc[0]), float(prices.iloc[-1])
    total_return = last / first - 1.0
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = (last / first) ** (1.0 / years) - 1.0 if years > 0 else None

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volatility = sharpe = None
    if len(returns) >= 2:
        volatility = _finite(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
        annual_mean = _finite(returns.mean() * TRADING_DAYS_PER_YEAR)
        if volatility and volatility > 1e-12 and annual_mean is not None:
            sharpe = (annual_mean - risk_free_rate) / volatility

    return RiskMetrics(last, _finite(total_return), _finite(cagr), volatility, _finite(sharpe), n)


# --------------------------------------------------------------------------------------
# Synthetic dark pool transaction tape
# --------------------------------------------------------------------------------------

def simulate_dark_pool_tape(ticker: str, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    Build a per-session off-exchange tape on top of real close/volume.

    Baseline sessions draw an off-exchange share around 36% with loosely scattered block
    prints. A ticker-specific number of accumulation episodes (2-6 sessions each) is injected
    at sessions trading near their rolling floor: heavy off-exchange share, block prints
    pinned just above the floor, and tight print dispersion.
    """
    frame = pd.DataFrame({"close": close, "total_volume": volume})
    frame = frame[frame["close"] > 0]
    n = len(frame)
    if n < FLOOR_WINDOW:
        raise InsufficientDataError(f"only {n} valid sessions (need {FLOOR_WINDOW})")

    rng = np.random.default_rng(_stable_seed(ticker, frame.index[-1]))
    px = frame["close"].to_numpy(dtype=float)
    floor = frame["close"].rolling(FLOOR_WINDOW, min_periods=5).min().to_numpy(dtype=float)
    total = frame["total_volume"].to_numpy(dtype=float)
    valid_volume = np.isfinite(total) & (total > 0)

    share = rng.beta(18.0, 32.0, n)
    dark_vwap = px * (1.0 + rng.normal(0.0, 0.006, n))
    dispersion = rng.uniform(0.003, 0.012, n)

    candidates = np.flatnonzero(np.isfinite(floor) & (px <= floor * (1.0 + EPISODE_FLOOR_PROXIMITY)))
    institutional_interest = rng.uniform(0.0, 1.0)
    n_episodes = min(int(rng.poisson(1.0 + 9.0 * institutional_interest)), candidates.size)
    starts = rng.choice(candidates, size=n_episodes, replace=False) if n_episodes else np.empty(0, int)
    for start in starts:
        stop = min(int(start) + int(rng.integers(2, 7)), n)
        k = stop - int(start)
        sl = slice(int(start), stop)
        share[sl] = rng.uniform(0.47, 0.68, k)
        dark_vwap[sl] = np.minimum(px[sl], floor[sl] * (1.0 + rng.uniform(0.0, 0.012, k)))
        dispersion[sl] = rng.uniform(0.0008, 0.0035, k)

    share = np.where(valid_volume, share, np.nan)
    off_volume = np.where(valid_volume, np.floor(total * share), np.nan)

    return pd.DataFrame(
        {
            "close": px,
            "total_volume": np.where(valid_volume, total, np.nan),
            "off_exchange_volume": off_volume,
            "off_exchange_share": share,
            "dark_vwap": dark_vwap,
            "print_dispersion": dispersion,
            "price_floor": floor,
        },
        index=frame.index,
    )


def flag_anomalies(tape: pd.DataFrame) -> pd.Series:
    """Sessions where off-exchange share > threshold AND prints sit uniformly on the floor."""
    heavy = tape["off_exchange_share"] > OFF_EXCHANGE_THRESHOLD
    at_floor = tape["dark_vwap"] <= tape["price_floor"] * (1.0 + FLOOR_BAND)
    uniform = tape["print_dispersion"] <= UNIFORMITY_TOL
    return (heavy & at_floor & uniform).astype(bool)  # NaN comparisons evaluate False


@dataclass(frozen=True)
class AccumulationCluster:
    start: pd.Timestamp
    end: pd.Timestamp
    anomalous_days: int
    avg_off_exchange_share: float
    accumulation_price: float
    off_exchange_volume: float
    last_position: int


def build_clusters(tape: pd.DataFrame, flags: pd.Series) -> list[AccumulationCluster]:
    positions = np.flatnonzero(flags.to_numpy())
    if positions.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) > CLUSTER_MAX_GAP + 1) + 1
    clusters = []
    for group in np.split(positions, breaks):
        rows = tape.iloc[group]
        clusters.append(
            AccumulationCluster(
                start=rows.index[0],
                end=rows.index[-1],
                anomalous_days=len(group),
                avg_off_exchange_share=float(np.nanmean(rows["off_exchange_share"])),
                accumulation_price=float(np.nanmean(rows["dark_vwap"])),
                off_exchange_volume=float(np.nansum(rows["off_exchange_volume"])),
                last_position=int(group[-1]),
            )
        )
    return clusters


@dataclass(frozen=True)
class WhaleScore:
    score: float
    density: float
    intensity: float
    persistence: float
    recency: float


def score_whale_accumulation(
    tape: pd.DataFrame, flags: pd.Series, clusters: list[AccumulationCluster]
) -> WhaleScore:
    if not clusters:
        return WhaleScore(0.0, 0.0, 0.0, 0.0, 0.0)

    density = _clip01(float(flags.iloc[-DENSITY_WINDOW:].mean()) / DENSITY_SATURATION)

    avg_volume = tape["total_volume"].rolling(20, min_periods=5).mean()
    flagged = tape.loc[flags]
    excess = ((flagged["off_exchange_share"] - OFF_EXCHANGE_THRESHOLD)
              / (SHARE_CEILING - OFF_EXCHANGE_THRESHOLD)).clip(0.0, 1.0)
    volume_credit = (flagged["total_volume"] / avg_volume.loc[flags]).clip(0.0, VOLUME_SATURATION)
    volume_credit = (volume_credit / VOLUME_SATURATION).fillna(0.5)
    intensity = _clip01(_finite((excess * volume_credit).mean()) or 0.0)

    longest = max(c.anomalous_days for c in clusters)
    persistence = _clip01(longest / PERSISTENCE_SATURATION)

    sessions_since_last = len(tape) - 1 - clusters[-1].last_position
    recency = 0.5 ** (sessions_since_last / RECENCY_HALF_LIFE)

    score = 100.0 * (
        SCORE_WEIGHTS["density"] * density
        + SCORE_WEIGHTS["intensity"] * intensity
        + SCORE_WEIGHTS["persistence"] * persistence
        + SCORE_WEIGHTS["recency"] * recency
    )
    return WhaleScore(score, density, intensity, persistence, recency)


def conviction_score(whale: float, sharpe: float | None) -> float:
    sharpe_credit = 0.0 if sharpe is None else _clip01((sharpe - SHARPE_FLOOR) / (SHARPE_CEILING - SHARPE_FLOOR))
    return CONVICTION_WHALE_WEIGHT * whale + (1.0 - CONVICTION_WHALE_WEIGHT) * 100.0 * sharpe_credit


def classify_signal(whale: float) -> str:
    if whale >= 70:
        return "STRONG_ACCUMULATION"
    if whale >= 45:
        return "ACCUMULATION"
    if whale >= 20:
        return "WATCH"
    return "NEUTRAL"


@dataclass(frozen=True)
class TickerAnalysis:
    ticker: str
    risk: RiskMetrics
    whale: WhaleScore
    conviction: float
    clusters: list[AccumulationCluster]
    anomalous_days: int
    avg_off_exchange_share: float | None
    last_anomaly: pd.Timestamp | None
    sessions_since_last_anomaly: int | None


def analyze_ticker(ticker: str, close: pd.Series, volume: pd.Series) -> TickerAnalysis:
    risk = compute_risk_metrics(close, RISK_FREE_RATE)
    tape = simulate_dark_pool_tape(ticker, close, volume)
    flags = flag_anomalies(tape)
    clusters = build_clusters(tape, flags)
    whale = score_whale_accumulation(tape, flags, clusters)
    last = clusters[-1] if clusters else None
    return TickerAnalysis(
        ticker=ticker,
        risk=risk,
        whale=whale,
        conviction=conviction_score(whale.score, risk.sharpe_ratio),
        clusters=clusters,
        anomalous_days=int(flags.sum()),
        avg_off_exchange_share=_finite(tape["off_exchange_share"].mean()),
        last_anomaly=last.end if last else None,
        sessions_since_last_anomaly=(len(tape) - 1 - last.last_position) if last else None,
    )


# --------------------------------------------------------------------------------------
# API schema
# --------------------------------------------------------------------------------------

class ClusterOut(BaseModel):
    start_date: str
    end_date: str
    anomalous_days: int
    avg_off_exchange_share: float | None
    accumulation_price: float | None
    off_exchange_volume: int | None


class DarkPoolOut(BaseModel):
    avg_off_exchange_share: float | None
    anomalous_days: int
    cluster_count: int
    longest_cluster_days: int
    last_anomaly_date: str | None
    sessions_since_last_anomaly: int | None
    accumulation_floor_price: float | None
    recent_clusters: list[ClusterOut]


class ScoreComponentsOut(BaseModel):
    density: float
    intensity: float
    persistence: float
    recency: float


class TargetOut(BaseModel):
    rank: int
    ticker: str
    signal: str
    conviction_score: float
    whale_accumulation_cluster_score: float
    score_components: ScoreComponentsOut
    last_close: float | None
    total_return: float | None
    cagr: float | None
    annualized_volatility: float | None
    sharpe_ratio: float | None
    observations: int
    dark_pool: DarkPoolOut


class ExcludedOut(BaseModel):
    ticker: str
    reason: str


class ShadowFlowResponse(BaseModel):
    generated_at: str
    as_of: str | None
    lookback: str
    universe_size: int
    analyzed: int
    risk_free_rate: float
    off_exchange_threshold: float
    data_provenance: dict[str, str]
    targets: list[TargetOut]
    excluded: list[ExcludedOut]


def _serialize_cluster(c: AccumulationCluster) -> ClusterOut:
    volume = _finite(c.off_exchange_volume)
    return ClusterOut(
        start_date=c.start.strftime("%Y-%m-%d"),
        end_date=c.end.strftime("%Y-%m-%d"),
        anomalous_days=c.anomalous_days,
        avg_off_exchange_share=_round(c.avg_off_exchange_share),
        accumulation_price=_round(c.accumulation_price, 2),
        off_exchange_volume=int(volume) if volume is not None else None,
    )


def _serialize_target(rank: int, a: TickerAnalysis) -> TargetOut:
    last = a.clusters[-1] if a.clusters else None
    return TargetOut(
        rank=rank,
        ticker=a.ticker,
        signal=classify_signal(a.whale.score),
        conviction_score=round(a.conviction, 2),
        whale_accumulation_cluster_score=round(a.whale.score, 2),
        score_components=ScoreComponentsOut(
            density=round(a.whale.density, 4),
            intensity=round(a.whale.intensity, 4),
            persistence=round(a.whale.persistence, 4),
            recency=round(a.whale.recency, 4),
        ),
        last_close=_round(a.risk.last_close, 2),
        total_return=_round(a.risk.total_return),
        cagr=_round(a.risk.cagr),
        annualized_volatility=_round(a.risk.annualized_volatility),
        sharpe_ratio=_round(a.risk.sharpe_ratio, 3),
        observations=a.risk.observations,
        dark_pool=DarkPoolOut(
            avg_off_exchange_share=_round(a.avg_off_exchange_share),
            anomalous_days=a.anomalous_days,
            cluster_count=len(a.clusters),
            longest_cluster_days=max((c.anomalous_days for c in a.clusters), default=0),
            last_anomaly_date=a.last_anomaly.strftime("%Y-%m-%d") if a.last_anomaly is not None else None,
            sessions_since_last_anomaly=a.sessions_since_last_anomaly,
            accumulation_floor_price=_round(last.accumulation_price, 2) if last else None,
            recent_clusters=[_serialize_cluster(c) for c in reversed(a.clusters[-3:])],
        ),
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

class ShadowFlowEngine:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cached: tuple[float, ShadowFlowResponse] | None = None

    async def get_metrics(self, force_refresh: bool = False) -> ShadowFlowResponse:
        async with self._lock:  # one download in flight; concurrent callers share its result
            if not force_refresh and self._cached and time.monotonic() - self._cached[0] < CACHE_TTL_SECONDS:
                return self._cached[1]
            payload = await self._compute()
            self._cached = (time.monotonic(), payload)
            return payload

    async def _compute(self) -> ShadowFlowResponse:
        started = time.perf_counter()
        close, volume = await asyncio.to_thread(download_history, TICKERS, LOOKBACK_PERIOD)

        results = await asyncio.gather(
            *(asyncio.to_thread(analyze_ticker, t, close[t], volume[t]) for t in TICKERS),
            return_exceptions=True,
        )

        analyses: list[TickerAnalysis] = []
        excluded: list[ExcludedOut] = []
        for ticker, result in zip(TICKERS, results):
            if isinstance(result, TickerAnalysis):
                analyses.append(result)
            else:
                logger.warning("Excluding %s: %r", ticker, result)
                excluded.append(ExcludedOut(ticker=ticker, reason=str(result) or type(result).__name__))

        if not analyses:
            raise DataUnavailableError("no ticker produced usable history")

        analyses.sort(key=lambda a: (a.conviction, a.whale.score), reverse=True)
        valid_dates = close.dropna(how="all").index
        logger.info("Computed shadow-flow metrics for %d tickers in %.2fs", len(analyses), time.perf_counter() - started)

        return ShadowFlowResponse(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            as_of=valid_dates[-1].strftime("%Y-%m-%d") if len(valid_dates) else None,
            lookback=LOOKBACK_PERIOD,
            universe_size=len(TICKERS),
            analyzed=len(analyses),
            risk_free_rate=RISK_FREE_RATE,
            off_exchange_threshold=OFF_EXCHANGE_THRESHOLD,
            data_provenance={
                "prices_and_volume": "Yahoo Finance via yfinance (split/dividend adjusted daily bars)",
                "off_exchange_tape": "SYNTHETIC - seeded simulator; not real dark pool data, not a trading signal",
            },
            targets=[_serialize_target(i, a) for i, a in enumerate(analyses[:TOP_N_TARGETS], start=1)],
            excluded=excluded,
        )


engine = ShadowFlowEngine()


async def _warm_cache() -> None:
    try:
        await engine.get_metrics()
    except Exception:  # startup must not fail because the data vendor is down
        logger.exception("Cache warm-up failed; metrics will be computed on first request")


@asynccontextmanager
async def lifespan(_: FastAPI):
    warmup = asyncio.create_task(_warm_cache())
    yield
    warmup.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await warmup


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

app = FastAPI(
    title="ShadowFlow Intelligence API",
    version="1.0.0",
    description="Institutional order-flow monitoring. Off-exchange tape is simulated.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.state.shadowflow_engine = engine
app.include_router(portfolio_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/shadow-flow-metrics", response_model=ShadowFlowResponse)
async def shadow_flow_metrics(
    refresh: bool = Query(False, description="Bypass the cache and recompute from fresh market data."),
) -> ShadowFlowResponse:
    try:
        return await engine.get_metrics(force_refresh=refresh)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Shadow-flow computation failed")
        raise HTTPException(status_code=500, detail="Shadow-flow computation failed") from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
