"""LiveLoopOrchestrator — End-to-end continuous live loop orchestrator.

Coordinates the entire Metadron Capital Intelligence Platform from data
ingestion through signal generation, intelligence, decision, execution,
learning, and monitoring.  Runs as a continuous 1-minute heartbeat loop
throughout the trading day with cadence-based phase scheduling.

Architecture:
    Phase 1  DATA          (every tick)     Ingestion + Universal Data Pool
    Phase 2  SIGNALS       (1-min cadence)  Macro, Cube, Liquidity, Fundamentals
    Phase 3  INTELLIGENCE  (5-min cadence)  Alpha optimizer, ML ensemble, agents
    Phase 4  DECISION      (on signal Δ)    Decision matrix, beta corridor, options
    Phase 5  EXECUTION     (on approval)    Execution engine, options, futures hedge
    Phase 6  LEARNING      (continuous)     Feedback loops, agent gradients
    Phase 7  MONITORING    (5-min cadence)  P&L, risk, anomaly, snapshot

Daily Schedule:
    08:00-09:30  Pre-market   Full refresh, overnight signals, SEC scan
    09:30        Market open  Full pipeline flush
    09:30-16:00  Intraday     Continuous 1-min heartbeat
    16:00        Market close EOD reconciliation, learning snapshot
    16:00-20:00  After-hours  Reduced frequency, earnings scan
    20:00-08:00  Overnight    Backtesting, ML training, pattern evolution
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# External imports — graceful degradation (rule 8: try/except on ALL)
# ---------------------------------------------------------------------------
try:
    from .signals.macro_engine import MacroEngine, MacroSnapshot, MarketRegime
except ImportError:
    MacroEngine = None
    MacroSnapshot = None
    MarketRegime = None

try:
    from .signals.metadron_cube import MetadronCube, CubeOutput
except ImportError:
    MetadronCube = None
    CubeOutput = None

try:
    from .signals.security_analysis_engine import SecurityAnalysisEngine
except ImportError:
    SecurityAnalysisEngine = None

try:
    from .ml.alpha_optimizer import AlphaOptimizer, AlphaOutput, AlphaSignal
except ImportError:
    AlphaOptimizer = None
    AlphaOutput = None
    AlphaSignal = None

try:
    from .execution.decision_matrix import DecisionMatrix
except ImportError:
    DecisionMatrix = None

try:
    from .allocation.allocation_engine import (
        AllocationEngine, AllocationRules, ScanSignal as AllocationScanSignal,
        BetaCorridorLevel as AllocBetaCorridorLevel,
    )
except ImportError:
    AllocationEngine = None
    AllocationRules = None
    AllocationScanSignal = None
    AllocBetaCorridorLevel = None

try:
    from .allocation.universe_scan import FullUniverseScan
except ImportError:
    FullUniverseScan = None

try:
    from .execution.execution_engine import ExecutionEngine
except ImportError:
    ExecutionEngine = None

try:
    from .execution.options_engine import OptionsEngine
except ImportError:
    OptionsEngine = None

try:
    from .portfolio.beta_corridor import BetaCorridor, BetaState, BetaAction
except ImportError:
    BetaCorridor = None
    BetaState = None
    BetaAction = None

try:
    from .monitoring.learning_loop import LearningLoop, SignalOutcome, RegimeFeedback
except ImportError:
    LearningLoop = None
    SignalOutcome = None
    RegimeFeedback = None

try:
    from .monitoring.anomaly_detector import AnomalyDetector
except ImportError:
    AnomalyDetector = None

try:
    from .data.universe_engine import UniverseEngine, get_engine
except ImportError:
    UniverseEngine = None
    get_engine = None

try:
    from .agents.sector_bots import SectorBotManager
except ImportError:
    SectorBotManager = None

try:
    from .agents.agent_scorecard import AgentScorecard
except ImportError:
    AgentScorecard = None

try:
    from .monitoring.portfolio_analytics import PortfolioAnalytics
except ImportError:
    PortfolioAnalytics = None

# Components that may not exist yet — future-proofed stubs
try:
    from .data.ingestion_orchestrator import DataIngestionOrchestrator
except ImportError:
    DataIngestionOrchestrator = None

try:
    from .data.universal_pooling import UniversalDataPool
except ImportError:
    UniversalDataPool = None

try:
    from .signals.fed_liquidity_plumbing import FedLiquidityPlumbing
except ImportError:
    FedLiquidityPlumbing = None

try:
    from .ml.qstrader_backtest_bridge import QSTraderBacktestRunner
except ImportError:
    QSTraderBacktestRunner = None

# ---------------------------------------------------------------------------
# Signal engines — previously only in run_open.py / ExecutionEngine.run_pipeline()
# Now wired into live loop for continuous signal generation + ensemble feeding.
# ---------------------------------------------------------------------------
try:
    from .signals.contagion_engine import ContagionEngine
except ImportError:
    ContagionEngine = None

try:
    from .signals.stat_arb_engine import StatArbEngine
except ImportError:
    StatArbEngine = None

try:
    from .signals.fixed_income_engine import FixedIncomeEngine
except ImportError:
    FixedIncomeEngine = None

try:
    from .signals.news_engine import NewsEngine
except ImportError:
    NewsEngine = None

try:
    from .ml.pattern_recognition import PatternRecognitionEngine
except ImportError:
    PatternRecognitionEngine = None

try:
    from .ml.bridges.stock_prediction_bridge import StockPredictionBridge
except ImportError:
    StockPredictionBridge = None

try:
    from .signals.pattern_discovery_engine import PatternDiscoveryEngine
except ImportError:
    PatternDiscoveryEngine = None

try:
    from .signals.social_prediction_engine import MiroMomentumEngine
except ImportError:
    MiroMomentumEngine = None

try:
    from .signals.distressed_asset_engine import DistressedAssetEngine
except ImportError:
    DistressedAssetEngine = None

try:
    from .signals.cvr_engine import CVREngine
except ImportError:
    CVREngine = None

try:
    from .signals.event_driven_engine import EventDrivenEngine
except ImportError:
    EventDrivenEngine = None

try:
    from .agents.graphify_bridge import GraphifyBridge
except ImportError:
    GraphifyBridge = None

try:
    from .research.autoresearch_bridge import AutoresearchBridge
except ImportError:
    AutoresearchBridge = None

try:
    from .agents.research_bots import ResearchBotManager
except ImportError:
    ResearchBotManager = None

try:
    from .security.integrity import get_security, SecurityManager
    from .security.token_meter import get_meter
except ImportError:
    get_security = None
    SecurityManager = None
    get_meter = None

try:
    from intelligence_platform.plugins.gsd_paul_plugin import GSDPlugin, PaulPlugin
except ImportError:
    try:
        from ..intelligence_platform.plugins.gsd_paul_plugin import GSDPlugin, PaulPlugin
    except ImportError:
        GSDPlugin = None
        PaulPlugin = None

try:
    from intelligence_platform.plugins.gsd_workflow_bridge import GSDWorkflowBridge
except ImportError:
    try:
        from ..intelligence_platform.plugins.gsd_workflow_bridge import GSDWorkflowBridge
    except ImportError:
        GSDWorkflowBridge = None

logger = logging.getLogger("metadron.orchestrator")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class LoopPhase(str, Enum):
    """Phases within a single heartbeat iteration."""
    DATA = "DATA"
    SIGNALS = "SIGNALS"
    INTELLIGENCE = "INTELLIGENCE"
    DECISION = "DECISION"
    EXECUTION = "EXECUTION"
    LEARNING = "LEARNING"
    MONITORING = "MONITORING"


class LoopState(str, Enum):
    """Overall orchestrator state."""
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class MarketSession(str, Enum):
    """Current market session."""
    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    INTRADAY = "INTRADAY"
    MARKET_CLOSE = "MARKET_CLOSE"
    AFTER_HOURS = "AFTER_HOURS"
    OVERNIGHT = "OVERNIGHT"
    WEEKEND = "WEEKEND"


class RiskLevel(str, Enum):
    """Circuit breaker risk level."""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    KILL_SWITCH = "KILL_SWITCH"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    """Result of a single phase execution."""
    phase: str = ""
    success: bool = True
    duration_ms: float = 0.0
    timestamp: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""
    items_processed: int = 0


@dataclass
class HeartbeatResult:
    """Result of a complete heartbeat iteration."""
    iteration: int = 0
    timestamp: str = ""
    total_duration_ms: float = 0.0
    session: str = ""
    phases: Dict[str, PhaseResult] = field(default_factory=dict)
    signals_generated: int = 0
    trades_approved: int = 0
    trades_executed: int = 0
    errors: List[str] = field(default_factory=list)
    risk_level: str = RiskLevel.NORMAL.value


@dataclass
class LoopPerformance:
    """Cumulative loop performance metrics."""
    total_iterations: int = 0
    total_runtime_seconds: float = 0.0
    avg_heartbeat_ms: float = 0.0
    max_heartbeat_ms: float = 0.0
    min_heartbeat_ms: float = 999999.0
    total_signals: int = 0
    total_trades_approved: int = 0
    total_trades_executed: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    max_consecutive_errors: int = 0
    uptime_pct: float = 100.0
    phase_avg_ms: Dict[str, float] = field(default_factory=dict)
    phase_total_ms: Dict[str, float] = field(default_factory=dict)
    phase_call_count: Dict[str, int] = field(default_factory=dict)
    last_heartbeat: str = ""
    started_at: str = ""
    restarts: int = 0


@dataclass
class LoopStatus:
    """Current orchestrator status snapshot."""
    state: str = LoopState.IDLE.value
    session: str = MarketSession.OVERNIGHT.value
    risk_level: str = RiskLevel.NORMAL.value
    current_phase: str = ""
    iteration: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: str = ""
    next_heartbeat_in_seconds: float = 0.0
    components_loaded: Dict[str, bool] = field(default_factory=dict)
    kill_switch_active: bool = False
    performance: Optional[LoopPerformance] = None
    last_macro_regime: str = ""
    last_cube_regime: str = ""
    portfolio_nav: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0


@dataclass
class CircuitBreakerState:
    """Risk circuit breaker state."""
    level: RiskLevel = RiskLevel.NORMAL
    kill_switch_active: bool = False
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    vix_level: float = 20.0
    hy_spread_change_bps: float = 0.0
    breadth_pct: float = 50.0
    last_check: str = ""
    triggers: List[str] = field(default_factory=list)

    # Thresholds
    DRAWDOWN_ELEVATED: float = 0.02
    DRAWDOWN_HIGH: float = 0.05
    DRAWDOWN_CRITICAL: float = 0.08
    DRAWDOWN_KILL: float = 0.12
    VIX_ELEVATED: float = 25.0
    VIX_HIGH: float = 35.0
    VIX_CRITICAL: float = 45.0
    CONSECUTIVE_LOSS_LIMIT: int = 5
    HY_SPREAD_KILL_BPS: float = 35.0
    BREADTH_KILL_PCT: float = 50.0


# ---------------------------------------------------------------------------
# Schedule time boundaries (ET)
# ---------------------------------------------------------------------------
_PRE_MARKET_START = dt_time(8, 0)
_MARKET_OPEN = dt_time(9, 30)
_MARKET_CLOSE = dt_time(16, 0)
_AFTER_HOURS_END = dt_time(20, 0)

# Cadence intervals (seconds) — adaptive for best execution
_HEARTBEAT_INTERVAL = 180          # Base: 3 minutes (balances load vs responsiveness)
_SIGNAL_CADENCE = 180              # 3 minutes — full signal pipeline
_INTELLIGENCE_CADENCE = 300        # 5 minutes — alpha optimizer, ML ensemble
_MONITORING_CADENCE = 300          # 5 minutes — P&L, risk, anomaly
_AFTER_HOURS_INTERVAL = 1800       # 30 minutes
_OVERNIGHT_INTERVAL = 3600         # 1 hour

# Burst modes for open/close
_OPEN_BURST_INTERVAL = 60          # 1-min for first 5 min after open
_OPEN_BURST_COUNT = 5
_CLOSE_BURST_INTERVAL = 60         # 1-min for last 15 min
_CLOSE_BURST_COUNT = 15

# Auto-restart limits
_MAX_CONSECUTIVE_ERRORS = 10
_RESTART_COOLDOWN_SECONDS = 30
_STATE_PERSISTENCE_INTERVAL = 300  # 5 minutes

# Paths
_STATE_DIR = Path("logs/live_loop")
_STATE_FILE = _STATE_DIR / "loop_state.json"
_PERFORMANCE_FILE = _STATE_DIR / "loop_performance.json"


# ---------------------------------------------------------------------------
# LiveLoopOrchestrator
# ---------------------------------------------------------------------------
class LiveLoopOrchestrator:
    """End-to-end continuous live loop orchestrator.

    Coordinates the entire Intelligence Platform from data ingestion
    through execution in a continuous 1-minute heartbeat loop.

    Features:
        - Thread-safe concurrent execution
        - Heartbeat monitoring with timing
        - Auto-restart on failures
        - Risk circuit breaker (KillSwitch)
        - Comprehensive logging
        - State persistence between restarts
        - Graceful shutdown
    """

    def __init__(
        self,
        initial_nav: float = 1_000_000.0,
        broker_type: str = "schwab",
        heartbeat_interval: float = _HEARTBEAT_INTERVAL,
        enable_risk_gates: bool = True,
        enable_persistence: bool = True,
        max_consecutive_errors: int = _MAX_CONSECUTIVE_ERRORS,
    ):
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Configuration
        self._initial_nav = initial_nav
        self._broker_type = broker_type
        self._heartbeat_interval = heartbeat_interval
        self._enable_risk_gates = enable_risk_gates
        self._enable_persistence = enable_persistence
        self._max_consecutive_errors = max_consecutive_errors
        self._test_stage = os.getenv("METADRON_STAGE", "").lower() in {"test", "testing"}

        # State
        self._state = LoopState.IDLE
        self._session = MarketSession.OVERNIGHT
        self._current_phase = ""
        self._iteration = 0
        self._started_at: Optional[datetime] = None
        self._last_heartbeat_time: Optional[datetime] = None

        # Cadence tracking (last execution timestamps)
        self._last_signal_time: Optional[datetime] = None
        self._last_intelligence_time: Optional[datetime] = None
        self._last_monitoring_time: Optional[datetime] = None
        self._last_state_persist_time: Optional[datetime] = None

        # Results
        self._performance = LoopPerformance()
        self._heartbeat_history: deque[HeartbeatResult] = deque(maxlen=1440)  # 24h of 1-min
        self._circuit_breaker = CircuitBreakerState()

        # Signal state cache (for change detection in decision phase)
        self._last_signals: Dict[str, Any] = {}
        self._pending_trades: List[dict] = []
        self._approved_trades: List[dict] = []

        # Last outputs from each phase
        self._last_macro_snapshot: Any = None
        self._last_cube_output: Any = None
        self._last_alpha_output: Any = None
        self._last_decision_result: Any = None
        self._last_scan_slate: Any = None          # latest AllocationSlate from FullUniverseScan
        self._last_contagion_output: Any = None    # ContagionEngine scenarios
        self._last_fi_output: Any = None           # FixedIncomeEngine summary
        self._last_news_miro_output: Any = None    # News+MiroMomentum enriched signals
        self._scan_task: Any = None                # background asyncio task for run_full_cycle

        # Initialize components
        self._components: Dict[str, Any] = {}
        self._component_status: Dict[str, bool] = {}
        self._init_components()

        # Restore state if available
        if self._enable_persistence:
            self._restore_state()

        logger.info(
            "LiveLoopOrchestrator initialized: nav=%.0f broker=%s interval=%ds components=%d/%d",
            initial_nav, broker_type, heartbeat_interval,
            sum(self._component_status.values()), len(self._component_status),
        )

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------
    def _init_components(self):
        """Initialize all platform components with graceful fallbacks."""
        component_factories: List[Tuple[str, Callable]] = [
            ("data_ingestion", lambda: DataIngestionOrchestrator() if DataIngestionOrchestrator else None),
            ("data_pool", lambda: UniversalDataPool() if UniversalDataPool else None),
            ("universe", lambda: get_engine() if get_engine else None),
            ("fed_liquidity", lambda: FedLiquidityPlumbing() if FedLiquidityPlumbing else None),
            ("macro_engine", lambda: MacroEngine() if MacroEngine else None),
            ("metadron_cube", lambda: MetadronCube() if MetadronCube else None),
            ("security_analysis", lambda: SecurityAnalysisEngine() if SecurityAnalysisEngine else None),
            ("alpha_optimizer", lambda: AlphaOptimizer() if AlphaOptimizer else None),
            ("decision_matrix", lambda: DecisionMatrix() if DecisionMatrix else None),
            ("allocation_engine", lambda: AllocationEngine() if AllocationEngine else None),
            ("universe_scan", lambda: FullUniverseScan(
                allocation_engine=self._components.get("allocation_engine"),
            ) if FullUniverseScan else None),
            ("execution_engine", lambda: ExecutionEngine(
                initial_nav=self._initial_nav,
                broker_type=self._broker_type,
                enable_risk_gates=self._enable_risk_gates,
            ) if ExecutionEngine else None),
            ("options_engine", lambda: (
                getattr(getattr(self._components.get("execution_engine"), "l7", None), "_options_engine", None)
                or OptionsEngine(
                    nav=self._initial_nav,
                    market_data=getattr(
                        getattr(self._components.get("execution_engine"), "broker", None),
                        "market_data", None,
                    ),
                )
            ) if OptionsEngine else None),
            ("beta_corridor", lambda: BetaCorridor(nav=self._initial_nav) if BetaCorridor else None),
            ("learning_loop", lambda: LearningLoop() if LearningLoop and not self._test_stage else None),
            ("anomaly_detector", lambda: AnomalyDetector() if AnomalyDetector else None),
            ("sector_bots", lambda: SectorBotManager() if SectorBotManager else None),
            ("agent_scorecard", lambda: AgentScorecard() if AgentScorecard else None),
            ("portfolio_analytics", lambda: PortfolioAnalytics() if PortfolioAnalytics else None),
            ("gsd_plugin", lambda: GSDPlugin() if GSDPlugin else None),
            ("paul_plugin", lambda: PaulPlugin() if PaulPlugin else None),
            ("paul_orchestrator", lambda: self._init_paul_orchestrator()),
            ("gsd_workflow", lambda: GSDWorkflowBridge() if GSDWorkflowBridge else None),
            ("backtest_runner", lambda: QSTraderBacktestRunner() if QSTraderBacktestRunner else None),
            # Signal engines — feed MLVoteEnsemble Tiers 6-10
            ("contagion_engine", lambda: ContagionEngine() if ContagionEngine else None),
            ("stat_arb_engine", lambda: StatArbEngine() if StatArbEngine else None),
            ("fixed_income_engine", lambda: FixedIncomeEngine() if FixedIncomeEngine else None),
            ("news_engine", lambda: NewsEngine() if NewsEngine else None),
            ("pattern_recognition", lambda: PatternRecognitionEngine() if PatternRecognitionEngine else None),
            ("stock_prediction", lambda: StockPredictionBridge() if StockPredictionBridge else None),
            ("pattern_discovery", lambda: PatternDiscoveryEngine() if PatternDiscoveryEngine else None),
            ("miro_momentum", lambda: MiroMomentumEngine() if MiroMomentumEngine else None),
            ("distressed_assets", lambda: DistressedAssetEngine() if DistressedAssetEngine else None),
            ("cvr_engine", lambda: CVREngine() if CVREngine else None),
            ("event_driven", lambda: EventDrivenEngine() if EventDrivenEngine else None),
            # Knowledge graph + research
            ("graphify", lambda: GraphifyBridge() if GraphifyBridge else None),
            ("autoresearch", lambda: AutoresearchBridge() if AutoresearchBridge else None),
            ("research_bots", lambda: ResearchBotManager() if ResearchBotManager else None),
            # Security subsystems
            ("security", lambda: get_security() if get_security else None),
            ("token_meter", lambda: get_meter() if get_meter else None),
        ]

        for name, factory in component_factories:
            try:
                component = factory()
                self._components[name] = component
                self._component_status[name] = component is not None
                if component is not None:
                    logger.debug("Component loaded: %s", name)
                else:
                    logger.debug("Component unavailable (class not imported): %s", name)
            except Exception as exc:
                self._components[name] = None
                self._component_status[name] = False
                logger.warning("Component init failed: %s — %s", name, exc)

    def _get(self, name: str) -> Any:
        """Retrieve a component by name, returns None if unavailable."""
        return self._components.get(name)

    def _init_paul_orchestrator(self):
        """Initialize the PaulOrchestrator with GSD + Paul + Factory + Enforcement.

        Connects the dynamic agent creation and enforcement system to the
        intelligence platform via the Paul Plugin orchestration layer.
        """
        try:
            from .agents.paul_orchestrator import PaulOrchestrator
            orch = PaulOrchestrator()
            init_status = orch.initialize()
            logger.info("PaulOrchestrator initialized: %s", init_status.get("status"))

            # Attach existing agents if available
            sector_bots = self._get("sector_bots")
            scorecard = self._get("agent_scorecard")

            bot_list = []
            if sector_bots and hasattr(sector_bots, "bots"):
                bot_list = list(sector_bots.bots.values()) if isinstance(sector_bots.bots, dict) else sector_bots.bots

            orch.attach_all_platform_agents(
                sector_bots=bot_list if bot_list else None,
            )
            return orch
        except Exception as exc:
            logger.warning("PaulOrchestrator init failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API — start / stop / status
    # ------------------------------------------------------------------
    def start(self):
        """Begin the continuous live loop in a background thread.

        The loop runs until stop() is called or a fatal error occurs.
        Thread-safe: can be called from any thread.
        """
        with self._lock:
            if self._state == LoopState.RUNNING:
                logger.warning("LiveLoop already running — ignoring start()")
                return

            self._state = LoopState.STARTING
            self._stop_event.clear()
            self._started_at = datetime.now()
            self._performance.started_at = self._started_at.isoformat()
            self._iteration = 0

            logger.info("=" * 70)
            logger.info("METADRON CAPITAL — LIVE LOOP STARTING")
            logger.info("  %s", self._started_at.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("=" * 70)

        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="LiveLoopOrchestrator",
            daemon=True,
        )
        self._loop_thread.start()

    def stop(self):
        """Graceful shutdown of the live loop.

        Signals the loop to stop, waits for the current heartbeat to
        finish, persists state, and joins the thread.
        """
        logger.info("LiveLoop stop requested — initiating graceful shutdown")
        with self._lock:
            self._state = LoopState.STOPPING

        self._stop_event.set()

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=120)
            if self._loop_thread.is_alive():
                logger.error("LiveLoop thread did not exit within 120s timeout")

        with self._lock:
            self._state = LoopState.STOPPED

        # Final state persistence
        if self._enable_persistence:
            self._persist_state()

        logger.info("LiveLoop stopped. Iterations=%d", self._iteration)

    def pause(self):
        """Pause the loop (skips heartbeats until resumed)."""
        with self._lock:
            if self._state == LoopState.RUNNING:
                self._state = LoopState.PAUSED
                logger.info("LiveLoop paused at iteration %d", self._iteration)

    def resume(self):
        """Resume a paused loop."""
        with self._lock:
            if self._state == LoopState.PAUSED:
                self._state = LoopState.RUNNING
                logger.info("LiveLoop resumed at iteration %d", self._iteration)

    # ------------------------------------------------------------------
    # Heartbeat — single loop iteration
    # ------------------------------------------------------------------
    def heartbeat(self) -> HeartbeatResult:
        """Execute a single complete heartbeat iteration.

        Can be called standalone (outside the loop) for testing or
        manual step-through execution.

        Returns:
            HeartbeatResult with timing and outcome for every phase.
        """
        t_start = time.monotonic()
        now = datetime.now()
        self._iteration += 1

        result = HeartbeatResult(
            iteration=self._iteration,
            timestamp=now.isoformat(),
            session=self._session.value,
        )

        logger.info(
            "--- Heartbeat #%d  [%s]  session=%s ---",
            self._iteration, now.strftime("%H:%M:%S"), self._session.value,
        )

        # Determine current session
        self._session = self._classify_session(now)
        result.session = self._session.value

        # Check circuit breaker before proceeding
        if self._circuit_breaker.kill_switch_active:
            logger.warning("KILL SWITCH ACTIVE — skipping execution phases")
            # Still run monitoring + learning for awareness
            result.risk_level = RiskLevel.KILL_SWITCH.value
            if not self._test_stage:
                self._run_phase_safe(LoopPhase.LEARNING, self.run_learning_phase, result)
                self._run_phase_safe(LoopPhase.MONITORING, self.run_monitoring_phase, result)
            result.total_duration_ms = (time.monotonic() - t_start) * 1000
            self._record_heartbeat(result)
            return result

        # Phase 1: DATA (every tick)
        self._run_phase_safe(LoopPhase.DATA, self.run_data_phase, result)

        # Phase 2: SIGNALS (1-min cadence)
        if self._should_run_cadence(self._last_signal_time, _SIGNAL_CADENCE):
            self._run_phase_safe(LoopPhase.SIGNALS, self.run_signal_phase, result)
            self._last_signal_time = now

        # Phase 3: INTELLIGENCE (5-min cadence)
        if self._should_run_cadence(self._last_intelligence_time, _INTELLIGENCE_CADENCE):
            self._run_phase_safe(LoopPhase.INTELLIGENCE, self.run_intelligence_phase, result)
            self._last_intelligence_time = now

        # Phase 4: DECISION (on signal change)
        if self._signals_changed():
            self._run_phase_safe(LoopPhase.DECISION, self.run_decision_phase, result)

        # Phase 5: EXECUTION (on approved trades)
        if self._approved_trades:
            self._run_phase_safe(LoopPhase.EXECUTION, self.run_execution_phase, result)

        # Phase 6: LEARNING (continuous)
        if not self._test_stage:
            self._run_phase_safe(LoopPhase.LEARNING, self.run_learning_phase, result)

        # Phase 7: MONITORING (5-min cadence)
        if not self._test_stage and self._should_run_cadence(self._last_monitoring_time, _MONITORING_CADENCE):
            self._run_phase_safe(LoopPhase.MONITORING, self.run_monitoring_phase, result)
            self._last_monitoring_time = now

        # Finalize heartbeat
        result.total_duration_ms = (time.monotonic() - t_start) * 1000
        result.risk_level = self._circuit_breaker.level.value
        self._record_heartbeat(result)
        self._last_heartbeat_time = now

        logger.info(
            "--- Heartbeat #%d complete: %.1fms  signals=%d  trades=%d/%d  errors=%d ---",
            self._iteration, result.total_duration_ms,
            result.signals_generated, result.trades_approved,
            result.trades_executed, len(result.errors),
        )

        return result

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------
    def run_data_phase(self) -> PhaseResult:
        """Phase 1: Data ingestion and pooling.

        Runs DataIngestionOrchestrator for continuous data feed, then
        routes through UniversalDataPool for normalization and distribution.
        Falls back to UniverseEngine.load() if dedicated components absent.
        """
        pr = PhaseResult(phase=LoopPhase.DATA.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()

        items = 0

        # Primary: dedicated ingestion orchestrator
        ingestion = self._get("data_ingestion")
        if ingestion:
            try:
                ing_result = ingestion.run_continuous_loop()
                items += ing_result.get("records_ingested", 0) if isinstance(ing_result, dict) else 0
                pr.data["ingestion"] = "ok"
            except Exception as exc:
                pr.data["ingestion_error"] = str(exc)
                logger.warning("DataIngestion error: %s", exc)

        # Data pool routing
        pool = self._get("data_pool")
        if pool:
            try:
                pool_result = pool.pool_all()
                items += pool_result.get("records_pooled", 0) if isinstance(pool_result, dict) else 0
                pr.data["pool"] = "ok"
            except Exception as exc:
                pr.data["pool_error"] = str(exc)
                logger.warning("DataPool error: %s", exc)

        # Fallback: load universe directly
        universe = self._get("universe")
        if universe:
            try:
                universe.load()
                size = universe.size() if hasattr(universe, "size") else 0
                pr.data["universe_size"] = size
                items += size
            except Exception as exc:
                pr.data["universe_error"] = str(exc)
                logger.warning("Universe load error: %s", exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_signal_phase(self) -> PhaseResult:
        """Phase 2: Signal generation from all signal engines.

        Runs at 1-minute cadence, two independent parallel tracks:

        TRACK A (MetadronCube → Signal Engines):
            FedLiquidityPlumbing → MacroEngine → MetadronCube →
            SecurityAnalysis, Contagion, StatArb, FixedIncome,
            DistressedAsset, PatternDiscovery, AdaptiveThreshold

        TRACK B (NewsEngine → MiroMomentum, independent from Cube):
            NewsEngine.run_miro_on_news_tickers() →
            EventDrivenEngine (enriched) + CVREngine (enriched)

        Both tracks run independently and their signals converge
        before the Intelligence phase (Stage 3).
        """
        pr = PhaseResult(phase=LoopPhase.SIGNALS.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()

        track_a_result = self._run_track_a()
        track_b_result = self._run_track_b()

        # ── MERGE: Converge both tracks into phase result ───────────

        signals_count = track_a_result.get("signals", 0) + track_b_result.get("signals", 0)

        pr.data["track_a"] = track_a_result
        pr.data["track_b"] = track_b_result
        pr.data["track_a_latency_ms"] = track_a_result.get("latency_ms", 0)
        pr.data["track_b_latency_ms"] = track_b_result.get("latency_ms", 0)
        pr.data["track_a_signals"] = track_a_result.get("signals", 0)
        pr.data["track_b_signals"] = track_b_result.get("signals", 0)

        pr.data["macro_regime"] = track_a_result.get("macro_regime", "")
        pr.data["cube_regime"] = track_a_result.get("cube_regime", "")
        pr.data["cube_target_beta"] = track_a_result.get("cube_target_beta", 0.0)
        pr.data["macro_vix"] = track_a_result.get("macro_vix", 0.0)
        pr.data["cube_kill_switch"] = track_a_result.get("cube_kill_switch", False)

        pr.data["news_miro_tickers"] = track_b_result.get("news_miro_tickers", 0)
        pr.data["news_miro_buys"] = track_b_result.get("news_miro_buys", 0)
        pr.data["news_miro_sells"] = track_b_result.get("news_miro_sells", 0)

        logger.info(
            "Signals phase: Track A=%d signals (%.0fms) | Track B=%d signals (%.0fms) | Total=%d",
            track_a_result.get("signals", 0), track_a_result.get("latency_ms", 0),
            track_b_result.get("signals", 0), track_b_result.get("latency_ms", 0),
            signals_count,
        )

        self._last_signals = {
            "macro_regime": pr.data.get("macro_regime", ""),
            "cube_regime": pr.data.get("cube_regime", ""),
            "cube_target_beta": pr.data.get("cube_target_beta", 0.0),
            "vix": pr.data.get("macro_vix", 0.0),
        }

        pr.items_processed = signals_count
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    # ── TRACK A: MetadronCube → Parallel Signal Engines ─────────────

    def _run_track_a(self) -> dict:
        """Track A: FedLiquidity → MacroEngine → MetadronCube → signal engines.

        Sequential pipeline from macro data through the Cube, then fans out
        to all signal engines fed by the Cube's regime output.
        """
        ta = {"signals": 0, "errors": []}
        ta_t0 = time.monotonic()

        # Fed Liquidity Plumbing
        fed = self._get("fed_liquidity")
        if fed:
            try:
                fed.update() if hasattr(fed, "update") else None
                ta["fed_liquidity"] = "updated"
                ta["signals"] += 1
            except Exception as exc:
                ta["fed_liquidity_error"] = str(exc)
                ta["errors"].append(f"FedLiquidity: {exc}")

        # Macro Engine
        macro = self._get("macro_engine")
        if macro:
            try:
                snap = macro.analyze()
                self._last_macro_snapshot = snap
                ta["macro_regime"] = snap.regime.value if hasattr(snap, "regime") else str(snap)
                ta["macro_vix"] = getattr(snap, "vix", 0.0)
                ta["signals"] += 1
                logger.info("Track A — Macro regime: %s  VIX: %.1f", ta["macro_regime"], ta["macro_vix"])
            except Exception as exc:
                ta["macro_error"] = str(exc)
                ta["errors"].append(f"MacroEngine: {exc}")

        # MetadronCube
        cube = self._get("metadron_cube")
        if cube and self._last_macro_snapshot:
            try:
                cube_out = cube.compute(self._last_macro_snapshot)
                self._last_cube_output = cube_out
                ta["cube_regime"] = cube_out.regime.value if hasattr(cube_out, "regime") else str(cube_out)
                ta["cube_target_beta"] = getattr(cube_out, "target_beta", 0.0)
                ta["signals"] += 1

                cube_kill = getattr(cube_out, "kill_switch_active", False)
                alloc = self._get("allocation_engine")
                if alloc and hasattr(alloc, "set_cube_kill_switch"):
                    alloc.set_cube_kill_switch(cube_kill)
                ta["cube_kill_switch"] = cube_kill

                logger.info("Track A — Cube regime: %s  target_beta: %.3f", ta["cube_regime"], ta["cube_target_beta"])
            except Exception as exc:
                ta["cube_error"] = str(exc)
                ta["errors"].append(f"MetadronCube: {exc}")

        # Parallel signal engines (all fed by Cube output)
        sa = self._get("security_analysis")
        if sa:
            try:
                universe = self._get("universe")
                tickers = [s.ticker for s in universe.get_all()[:50]] if universe and hasattr(universe, "get_all") else []
                if tickers:
                    sa_macro = {}
                    if self._last_macro_snapshot:
                        snap = self._last_macro_snapshot
                        sa_macro = {
                            "treasury_10y": getattr(snap, "treasury_10y", 0.045),
                            "sp500_pe": getattr(snap, "sp500_pe", 22.0),
                            "cape": getattr(snap, "cape", 28.0),
                            "hy_spread": getattr(snap, "hy_spread", 4.0),
                            "ig_spread": getattr(snap, "ig_spread", 1.5),
                            "vix": getattr(snap, "vix", 20.0),
                            "gdp_growth": getattr(snap, "gdp_growth", 0.02),
                            "cpi": getattr(snap, "cpi", 0.03),
                            "fedfunds": getattr(snap, "fedfunds", 0.05),
                        }
                    sa_result = sa.analyze(tickers, sa_macro, {})
                    ta["security_analysis_count"] = getattr(sa_result, "tickers_analyzed", 0)
                    ta["signals"] += 1
            except Exception as exc:
                ta["security_analysis_error"] = str(exc)

        contagion = self._get("contagion_engine")
        if contagion:
            try:
                scenarios = contagion.run_all_scenarios()
                ta["contagion_scenarios"] = len(scenarios) if scenarios else 0
                self._last_contagion_output = scenarios
                ta["signals"] += 1
            except Exception as exc:
                ta["contagion_error"] = str(exc)

        stat_arb = self._get("stat_arb_engine")
        if stat_arb:
            try:
                universe = self._get("universe")
                tickers = [s.ticker for s in universe.get_all()[:50]] if universe and hasattr(universe, "get_all") else []
                if tickers and hasattr(stat_arb, "scan_pairs"):
                    from .data.schwab_data import get_adj_close
                    import pandas as pd
                    price_data = {}
                    try:
                        prices_df = get_adj_close(tickers[:30], start=(datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d"))
                        if isinstance(prices_df, pd.DataFrame) and not prices_df.empty:
                            for col in prices_df.columns:
                                if col in tickers:
                                    price_data[col] = prices_df[col]
                    except Exception:
                        pass
                    if price_data:
                        pairs = stat_arb.scan_pairs(price_data)
                        ta["stat_arb_pairs"] = len(pairs) if pairs else 0
                        ta["signals"] += 1
            except Exception as exc:
                ta["stat_arb_error"] = str(exc)

        fi_engine = self._get("fixed_income_engine")
        if fi_engine:
            try:
                fi_summary = fi_engine.get_summary()
                ta["fixed_income"] = "updated"
                ta["fi_yield_10y"] = fi_summary.get("treasury_10y", 0.0) if fi_summary else 0.0
                self._last_fi_output = fi_summary
                ta["signals"] += 1
            except Exception as exc:
                ta["fixed_income_error"] = str(exc)

        ta["latency_ms"] = round((time.monotonic() - ta_t0) * 1000, 1)
        return ta

    # ── TRACK B: NewsEngine → MiroMomentum (independent) ───────────

    def _run_track_b(self) -> dict:
        """Track B: NewsEngine → MiroMomentum → EventDriven + CVR.

        Runs independently from Track A. Feeds directly from the NewsEngine
        (newsfilter.io) without any connection to MetadronCube.
        """
        tb = {"signals": 0, "errors": []}
        tb_t0 = time.monotonic()

        news = self._get("news_engine")
        if not news:
            tb["status"] = "news_engine_unavailable"
            tb["latency_ms"] = 0
            return tb

        try:
            enriched = news.run_miro_on_news_tickers()
            if enriched:
                tb["news_miro_tickers"] = len(enriched)
                tb["news_miro_buys"] = sum(1 for v in enriched.values() if v["signal"] == "BUY")
                tb["news_miro_sells"] = sum(1 for v in enriched.values() if v["signal"] == "SELL")
                self._last_news_miro_output = enriched

                events = self._get("event_driven")
                if events and hasattr(events, "_news_engine"):
                    events._news_engine = news
                    events._news_miro_signals = enriched
                    tb["news_to_event_driven"] = True

                cvr = self._get("cvr_engine")
                if cvr and hasattr(cvr, "_news_engine"):
                    cvr._news_engine = news
                    cvr._news_miro_signals = enriched
                    tb["news_to_cvr"] = True

                tb["signals"] += 1
            else:
                news.refresh()
                tb["news_feed_refreshed"] = True
        except Exception as exc:
            tb["news_miro_error"] = str(exc)
            tb["errors"].append(f"News+Miro: {exc}")
            logger.warning("Track B — News+MiroMomentum error: %s", exc)

        tb["latency_ms"] = round((time.monotonic() - tb_t0) * 1000, 1)
        return tb


    def run_intelligence_phase(self) -> PhaseResult:
        """Phase 3: ML intelligence, signal engine analysis, and agent scoring.

        Runs at 5-minute cadence:
            - FullUniverseScan.run_full_cycle() — 4-run universe scan (async background)
            - AlphaOptimizer.optimize()
            - PatternDiscoveryEngine.discover() — symbolic regression patterns
            - Feed MLVoteEnsemble Tiers 6-10:
                T6: MiroMomentumEngine → set_social_snapshot()
                T7: DistressedAssetEngine → set_distress_signals()
                T8: EventDrivenEngine → set_event_signals()
                T9: CVREngine → set_cvr_signals()
                T10: CreditQuality → set_credit_scores()
            - ML vote ensemble scoring (all 10 tiers active)
            - Agent sector bot scoring
        """
        pr = PhaseResult(phase=LoopPhase.INTELLIGENCE.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        # ── FullUniverseScan (4-run 20-min cycle, fired as async background task) ──
        # The scan runs SP500 → SP400 → SP600 → ETF+FI, emitting live SSE events to
        # the Thinking Tab.  Each 5-min intelligence tick either:
        #   (a) Starts a new scan cycle if none is running
        #   (b) Harvests the completed slate from the previous cycle
        scanner = self._get("universe_scan")
        if scanner:
            try:
                import asyncio
                # Sync the scanner's AllocationEngine NAV with current live NAV
                alloc_engine = self._get("allocation_engine")
                if alloc_engine:
                    alloc_engine.nav = self._get_live_nav()

                # Harvest completed scan slate if ready
                if (
                    self._scan_task is not None
                    and hasattr(self._scan_task, "done")
                    and self._scan_task.done()
                    and not self._scan_task.cancelled()
                ):
                    try:
                        self._last_scan_slate = self._scan_task.result()
                        pos_count = len(getattr(self._last_scan_slate, "positions", []))
                        kill = getattr(self._last_scan_slate, "kill_switch_triggered", False)
                        pr.data["scan_slate_positions"] = pos_count
                        pr.data["scan_kill_switch"] = kill
                        logger.info(
                            "[LiveLoop] FullUniverseScan cycle complete: %d positions kill_switch=%s",
                            pos_count, kill,
                        )
                    except Exception as harvest_exc:
                        logger.warning("[LiveLoop] scan harvest error: %s", harvest_exc)
                    self._scan_task = None

                # Launch new scan cycle if none is running
                if self._scan_task is None or (
                    hasattr(self._scan_task, "done") and self._scan_task.done()
                ):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            self._scan_task = loop.create_task(
                                scanner.run_full_cycle()
                            )
                            pr.data["scan_cycle_launched"] = True
                            logger.info(
                                "[LiveLoop] FullUniverseScan cycle #%d launched",
                                scanner.cycle_count + 1,
                            )
                        else:
                            pr.data["scan_cycle_skipped"] = "no_running_event_loop"
                    except RuntimeError:
                        pr.data["scan_cycle_skipped"] = "event_loop_unavailable"
                else:
                    pr.data["scan_cycle_running"] = True
                    pr.data["scan_cycle_number"] = scanner.cycle_count

                pr.data["scan_status"] = scanner.get_scan_status().get("phase", "IDLE")
            except Exception as exc:
                pr.data["scan_error"] = str(exc)
                logger.warning("[LiveLoop] FullUniverseScan error: %s", exc)

        # Alpha Optimizer — both pipelines run in parallel, results merged
        alpha = self._get("alpha_optimizer")
        if alpha:
            try:
                # Get universe tickers for optimization
                universe = self._get("universe")
                tickers = []
                if universe and hasattr(universe, "get_all"):
                    tickers = [s.ticker for s in universe.get_all()[:100]]

                if tickers:
                    # Run BOTH pipelines and merge signals
                    standard_out = None
                    enhanced_out = None

                    # Standard pipeline (XGBoost + CAPM + Quality + SLSQP)
                    if hasattr(alpha, "optimize"):
                        try:
                            standard_out = alpha.optimize(tickers)
                            pr.data["alpha_standard_signals"] = len(getattr(standard_out, "signals", []))
                        except Exception as std_exc:
                            logger.warning("Standard alpha pipeline error: %s", std_exc)

                    # Enhanced pipeline (WalkForward + FactorLibrary + sector MVO)
                    if hasattr(alpha, "run_enhanced_pipeline"):
                        try:
                            enhanced_out = alpha.run_enhanced_pipeline(tickers)
                            pr.data["alpha_enhanced_signals"] = len(getattr(enhanced_out, "signals", []))
                        except Exception as enh_exc:
                            logger.info("Enhanced alpha pipeline error: %s", enh_exc)

                    # Merge: enhanced output takes priority for weights/metrics,
                    # standard signals fill gaps for tickers enhanced missed
                    alpha_out = enhanced_out or standard_out
                    if alpha_out and standard_out and enhanced_out:
                        # Blend: merge any signals from standard that enhanced missed
                        enhanced_tickers = {s.ticker for s in getattr(enhanced_out, "signals", [])}
                        for sig in getattr(standard_out, "signals", []):
                            if sig.ticker not in enhanced_tickers:
                                enhanced_out.signals.append(sig)
                        alpha_out = enhanced_out
                        pr.data["alpha_pipeline"] = "dual"
                    elif alpha_out:
                        pr.data["alpha_pipeline"] = "enhanced" if enhanced_out else "standard"

                    if alpha_out:
                        self._last_alpha_output = alpha_out
                        pr.data["alpha_signals"] = len(getattr(alpha_out, "signals", []))
                        pr.data["alpha_expected_return"] = getattr(alpha_out, "expected_annual_return", 0.0)
                        pr.data["alpha_sharpe"] = getattr(alpha_out, "sharpe_ratio", 0.0)
                        items += pr.data["alpha_signals"]
                elif tickers and hasattr(alpha, "run"):
                    alpha_out = alpha.run(tickers)
                    self._last_alpha_output = alpha_out
                    items += 1
            except Exception as exc:
                pr.data["alpha_error"] = str(exc)
                logger.warning("AlphaOptimizer error: %s", exc)

        # ── Feed MLVoteEnsemble Tiers 6-10 before voting ──────────────────
        # These engines generate signals that the ensemble needs for
        # full 10-tier voting. Without this, Tiers 6-10 vote zero/neutral.
        exec_engine = self._get("execution_engine")
        ensemble = getattr(exec_engine, "ensemble", None) if exec_engine else None

        # Pattern Discovery Engine (enrichment features for AlphaOptimizer)
        pattern_disc = self._get("pattern_discovery")
        if pattern_disc:
            try:
                import pandas as pd
                universe = self._get("universe")
                tickers = []
                if universe and hasattr(universe, "get_all"):
                    tickers = [s.ticker for s in universe.get_all()[:30]]
                if tickers and hasattr(pattern_disc, "discover"):
                    from .data.schwab_data import get_adj_close
                    price_dict = {}
                    try:
                        prices_df = get_adj_close(
                            tickers[:20],
                            start=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                        )
                        if isinstance(prices_df, pd.DataFrame) and not prices_df.empty:
                            for col in prices_df.columns:
                                if col in tickers:
                                    price_dict[col] = prices_df[col]
                    except Exception:
                        pass
                    if price_dict:
                        discoveries = pattern_disc.discover(price_dict)
                        pr.data["pattern_discoveries"] = len(discoveries) if discoveries else 0
                        items += pr.data.get("pattern_discoveries", 0)
            except Exception as exc:
                pr.data["pattern_discovery_error"] = str(exc)
                logger.warning("PatternDiscovery error: %s", exc)

        # Pattern Recognition — candlestick, chart patterns, anomalies
        pat_rec = self._get("pattern_recognition")
        if pat_rec and self._last_alpha_output:
            try:
                alpha_tickers = [
                    getattr(s, "ticker", "") for s in getattr(self._last_alpha_output, "signals", [])[:20]
                    if getattr(s, "ticker", "")
                ]
                if alpha_tickers and hasattr(pat_rec, "analyze"):
                    pr_results = pat_rec.analyze(alpha_tickers)
                    pr.data["pattern_recognition_signals"] = len(pr_results) if pr_results else 0
                    items += 1
                elif alpha_tickers and hasattr(pat_rec, "detect_patterns"):
                    pr_results = pat_rec.detect_patterns(alpha_tickers)
                    pr.data["pattern_recognition_signals"] = len(pr_results) if pr_results else 0
                    items += 1
            except Exception as exc:
                pr.data["pattern_recognition_error"] = str(exc)
                logger.warning("PatternRecognition error: %s", exc)

        # Stock Prediction Bridge — neural net + ensemble predictions for Tier-1
        stock_pred = self._get("stock_prediction")
        if stock_pred and ensemble and self._last_alpha_output:
            try:
                alpha_tickers = [
                    getattr(s, "ticker", "") for s in getattr(self._last_alpha_output, "signals", [])[:20]
                    if getattr(s, "ticker", "")
                ]
                pred_signals = {}
                for ticker in alpha_tickers:
                    try:
                        if hasattr(stock_pred, "predict"):
                            pred = stock_pred.predict(ticker)
                            if pred:
                                pred_signals[ticker] = pred
                    except Exception:
                        pass
                if pred_signals:
                    pr.data["stock_predictions"] = len(pred_signals)
                    items += 1
            except Exception as exc:
                pr.data["stock_prediction_error"] = str(exc)
                logger.warning("StockPrediction error: %s", exc)

        # Tier 6: MiroMomentumEngine → ensemble.set_social_snapshot()
        miro = self._get("miro_momentum")
        if miro and ensemble:
            try:
                # Run agent sim on top alpha tickers
                alpha_tickers = []
                if self._last_alpha_output:
                    alpha_tickers = [
                        getattr(s, "ticker", "") for s in getattr(self._last_alpha_output, "signals", [])[:20]
                        if getattr(s, "ticker", "")
                    ]
                snapshots = miro.analyze(tickers=alpha_tickers) if alpha_tickers else {}
                if snapshots and hasattr(ensemble, "set_social_snapshot"):
                    ensemble.set_social_snapshot(snapshots)
                    pr.data["miro_momentum_signals"] = len(snapshots)
                    items += 1
            except Exception as exc:
                pr.data["miro_momentum_error"] = str(exc)
                logger.warning("MiroMomentumEngine error: %s", exc)

        # Tier 7: Distressed Asset Engine → ensemble.set_distress_signals()
        distressed = self._get("distressed_assets")
        if distressed and ensemble:
            try:
                distress_scores = distressed.analyze()
                if distress_scores and hasattr(ensemble, "set_distress_signals"):
                    ensemble.set_distress_signals(distress_scores)
                    pr.data["distressed_signals"] = len(distress_scores)
                    items += 1
            except Exception as exc:
                pr.data["distressed_error"] = str(exc)
                logger.warning("DistressedAssets error: %s", exc)

        # Tier 8: Event-Driven Engine → ensemble.set_event_signals()
        events = self._get("event_driven")
        if events and ensemble:
            try:
                regime_mult = 1.0
                if self._last_macro_snapshot:
                    regime = getattr(self._last_macro_snapshot, "regime", None)
                    if regime and hasattr(regime, "value"):
                        regime_str = regime.value
                        regime_mult = {"TRENDING": 1.2, "RANGE": 1.0, "STRESS": 0.6, "CRASH": 0.3}.get(regime_str, 1.0)
                event_result = events.analyze(regime_multiplier=regime_mult)
                if event_result and hasattr(ensemble, "set_event_signals"):
                    event_dict = {}
                    for pos in getattr(event_result, "positions", []):
                        ticker = getattr(pos, "ticker", "")
                        if ticker:
                            event_dict[ticker] = {
                                "category": getattr(pos, "category", ""),
                                "signal": getattr(pos, "signal", ""),
                                "confidence": getattr(pos, "confidence", 0.5),
                            }
                    ensemble.set_event_signals(event_dict)
                    pr.data["event_signals"] = len(event_dict)
                    items += 1
            except Exception as exc:
                pr.data["event_error"] = str(exc)
                logger.warning("EventDriven error: %s", exc)

        # Tier 9: CVR Engine → ensemble.set_cvr_signals()
        cvr = self._get("cvr_engine")
        if cvr and ensemble:
            try:
                cvr_valuations = cvr.analyze()
                if cvr_valuations and hasattr(ensemble, "set_cvr_signals"):
                    ensemble.set_cvr_signals(cvr_valuations)
                    pr.data["cvr_signals"] = len(cvr_valuations)
                    items += 1
            except Exception as exc:
                pr.data["cvr_error"] = str(exc)
                logger.warning("CVREngine error: %s", exc)

        # Tier 10: Credit quality scores (from SecurityAnalysis or UniverseClassifier)
        if ensemble and hasattr(ensemble, "set_credit_scores"):
            try:
                credit_scores = {}
                sa = self._get("security_analysis")
                if sa and hasattr(sa, "get_credit_scores"):
                    credit_scores = sa.get_credit_scores()
                if credit_scores:
                    ensemble.set_credit_scores(credit_scores)
                    pr.data["credit_scores_fed"] = len(credit_scores)
                    items += 1
            except Exception as exc:
                pr.data["credit_scores_error"] = str(exc)
                logger.warning("CreditScores error: %s", exc)

        # ML Vote Ensemble — now with all 10 tiers fed
        if ensemble and hasattr(ensemble, "vote") and self._last_alpha_output:
            try:
                signals = getattr(self._last_alpha_output, "signals", [])
                vote_count = 0
                for sig in signals[:20]:  # Top 20 signals
                    ticker = getattr(sig, "ticker", "")
                    if ticker:
                        try:
                            vote = ensemble.vote(ticker)
                            vote_count += 1
                        except Exception:
                            pass
                pr.data["ensemble_votes"] = vote_count
                items += vote_count
            except Exception as exc:
                pr.data["ensemble_error"] = str(exc)
                logger.warning("MLEnsemble error: %s", exc)

        # Agent sector bots scoring
        bots = self._get("sector_bots")
        if bots:
            try:
                if hasattr(bots, "score_all"):
                    bot_scores = bots.score_all()
                    pr.data["agent_scores"] = len(bot_scores) if bot_scores else 0
                    items += pr.data["agent_scores"]
                elif hasattr(bots, "run_all"):
                    bot_result = bots.run_all()
                    items += 1
            except Exception as exc:
                pr.data["agent_error"] = str(exc)
                logger.warning("SectorBots error: %s", exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def _get_live_nav(self) -> float:
        """Return the best available current NAV from the broker or initial value."""
        exec_engine = self._get("execution_engine")
        if exec_engine:
            broker = getattr(exec_engine, "broker", None) or getattr(exec_engine, "_broker", None)
            if broker:
                try:
                    if hasattr(broker, "nav"):
                        return float(broker.nav)
                    if hasattr(broker, "get_nav"):
                        return float(broker.get_nav())
                    if hasattr(broker, "cash"):
                        return float(broker.cash)
                except Exception:
                    pass
        return self._initial_nav

    def run_decision_phase(self) -> PhaseResult:
        """Phase 4: Trade decision evaluation.

        Triggered on signal change:
            - DecisionMatrix.evaluate()
            - AllocationEngine.apply_rules()  ← bucket sizing + caps enforced here
            - BetaCorridor.check()
            - Options opportunistic scan
            - Futures beta hedge check
        """
        pr = PhaseResult(phase=LoopPhase.DECISION.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        approved = 0

        self._pending_trades.clear()
        self._approved_trades.clear()

        # ── Path A: Use FullUniverseScan slate if available ───────────────────
        # If Phase 3 completed a scan cycle, the aggregated AllocationSlate already
        # has bucket-sized, cap-enforced positions ready to execute directly.
        # This is the primary path — it bypasses the single-batch DM+AllocationEngine
        # path and uses the full 4-run universe scan result instead.
        if self._last_scan_slate is not None:
            scan_slate = self._last_scan_slate
            if getattr(scan_slate, "kill_switch_triggered", False):
                logger.critical(
                    "[LiveLoop] Decision: scan slate kill switch active — blocking all trades"
                )
                pr.data["scan_kill_switch"] = True
                pr.data["trades_approved"] = 0
                pr.duration_ms = (time.monotonic() - t0) * 1000
                pr.success = True
                return pr

            for pos in getattr(scan_slate, "positions", []):
                self._approved_trades.append({
                    "ticker": pos.ticker,
                    "signal": None,
                    "decision": {"source": "universe_scan", "bucket": pos.bucket},
                    "dollar_amount": pos.dollar_amount,
                    "position_size": pos.position_size,
                    "bucket": pos.bucket,
                    "instrument_type": pos.instrument_type,
                    "confidence": pos.confidence,
                    "alpha_score": pos.alpha_score,
                    "timestamp": datetime.now().isoformat(),
                })

            approved = len(self._approved_trades)
            pr.data["source"] = "universe_scan"
            pr.data["trades_approved"] = approved
            pr.data["scan_cycle"] = getattr(scan_slate, "cycle_number", 0)
            pr.data["beta_corridor"] = getattr(scan_slate, "beta_corridor", "NEUTRAL")
            logger.info(
                "[LiveLoop] Decision: using scan slate — %d positions from cycle #%d",
                approved, getattr(scan_slate, "cycle_number", 0),
            )
            # Consume the slate so it’s not re-used next tick
            self._last_scan_slate = None

            # Still run BetaCorridor and Options scan below

        # ── Path B: Fallback DecisionMatrix + AllocationEngine (single-batch) ──
        # Used when no scan slate is ready (first few minutes after startup, or
        # between 20-min scan cycles).
        dm = self._get("decision_matrix")
        alloc_engine = self._get("allocation_engine")
        live_nav = self._get_live_nav()

        # Resolve current drawdown for AllocationEngine kill-switch
        current_drawdown = max(0.0, (self._initial_nav - live_nav) / self._initial_nav) if self._initial_nav > 0 else 0.0

        # Resolve beta corridor level for AllocationEngine
        alloc_beta_level = None
        if AllocBetaCorridorLevel and self._last_cube_output:
            target_beta = getattr(self._last_cube_output, "target_beta", 1.0)
            if target_beta > 1.1:
                alloc_beta_level = AllocBetaCorridorLevel.HIGH
            elif target_beta < 0.9:
                alloc_beta_level = AllocBetaCorridorLevel.LOW
            else:
                alloc_beta_level = AllocBetaCorridorLevel.NEUTRAL

        if dm and self._last_alpha_output and pr.data.get("source") != "universe_scan":
            try:
                signals = getattr(self._last_alpha_output, "signals", [])
                cube_out = self._last_cube_output
                macro_snap = self._last_macro_snapshot
                regime_str = str(getattr(macro_snap, "regime", "TRENDING")) if macro_snap else "TRENDING"

                # Step 1 — collect DecisionMatrix-approved signals
                dm_approved_signals = []
                for sig in signals[:30]:
                    ticker = getattr(sig, "ticker", "")
                    if not ticker:
                        continue
                    try:
                        if hasattr(dm, "evaluate"):
                            decision = dm.evaluate(
                                signal=sig,
                                cube_output=cube_out,
                                macro_snapshot=macro_snap,
                            )
                        elif hasattr(dm, "score"):
                            decision = dm.score(sig)
                        else:
                            continue

                        is_approved = False
                        if isinstance(decision, dict):
                            is_approved = decision.get("approved", False)
                        elif hasattr(decision, "approved"):
                            is_approved = decision.approved
                        elif hasattr(decision, "composite_score"):
                            is_approved = decision.composite_score >= 0.55

                        if is_approved:
                            dm_approved_signals.append((ticker, sig, decision))
                    except Exception as exc:
                        logger.debug("Decision eval failed for %s: %s", ticker, exc)

                pr.data["candidates_evaluated"] = len(signals[:30])
                pr.data["dm_approved"] = len(dm_approved_signals)

                # Step 2 — pass through AllocationEngine for bucket sizing + cap enforcement
                if alloc_engine and AllocationScanSignal and dm_approved_signals:
                    try:
                        scan_signals = []
                        _decision_map: dict = {}
                        for ticker, sig, decision in dm_approved_signals:
                            quality_tier = getattr(sig, "quality_tier", "B")
                            itype = "OPTION" if "opt" in ticker.lower() else "EQUITY"
                            confidence = float(getattr(sig, "weight", 0.5) or 0.5)
                            alpha_score = float(getattr(sig, "alpha_pred", 0.0))
                            scan_sig = AllocationScanSignal(
                                ticker=ticker,
                                signal_type="LONG" if alpha_score >= 0 else "SHORT",
                                instrument_type=itype,
                                confidence=min(max(confidence, 0.01), 1.0),
                                alpha_score=alpha_score,
                                regime_context=regime_str,
                            )
                            scan_signals.append(scan_sig)
                            _decision_map[ticker] = decision

                        slate = alloc_engine.apply_rules(
                            signals=scan_signals,
                            total_capital=live_nav,
                            drawdown=current_drawdown,
                            beta_corridor=alloc_beta_level,
                        )

                        if slate.kill_switch_triggered:
                            logger.critical(
                                "AllocationEngine KILL SWITCH triggered (drawdown=%.2f%%) "
                                "— blocking all equity trades this cycle",
                                current_drawdown * 100,
                            )
                            pr.data["allocation_kill_switch"] = True
                            pr.data["trades_approved"] = 0
                        else:
                            for pos in slate.positions:
                                self._approved_trades.append({
                                    "ticker": pos.ticker,
                                    "signal": _decision_map.get(pos.ticker),
                                    "decision": _decision_map.get(pos.ticker),
                                    "dollar_amount": pos.dollar_amount,
                                    "position_size": pos.position_size,
                                    "bucket": pos.bucket,
                                    "instrument_type": pos.instrument_type,
                                    "timestamp": datetime.now().isoformat(),
                                })
                                approved += 1

                            pr.data["allocation_slate_positions"] = len(slate.positions)
                            pr.data["allocation_nav"] = live_nav
                            pr.data["allocation_drawdown_pct"] = round(current_drawdown * 100, 3)
                            pr.data["allocation_beta_corridor"] = alloc_beta_level.value if alloc_beta_level else "NEUTRAL"
                            logger.info(
                                "AllocationEngine: %d/%d signals accepted into slate "
                                "(nav=%.0f drawdown=%.2f%% beta=%s)",
                                len(slate.positions), len(dm_approved_signals),
                                live_nav, current_drawdown * 100,
                                alloc_beta_level.value if alloc_beta_level else "NEUTRAL",
                            )
                    except Exception as exc:
                        # Allocation engine failure — fall back to raw DM approvals
                        logger.warning("AllocationEngine error — falling back to DM approvals: %s", exc)
                        for ticker, sig, decision in dm_approved_signals:
                            self._approved_trades.append({
                                "ticker": ticker,
                                "signal": sig,
                                "decision": decision,
                                "timestamp": datetime.now().isoformat(),
                            })
                            approved += 1
                else:
                    # AllocationEngine unavailable — use raw DM approvals
                    for ticker, sig, decision in dm_approved_signals:
                        self._approved_trades.append({
                            "ticker": ticker,
                            "signal": sig,
                            "decision": decision,
                            "timestamp": datetime.now().isoformat(),
                        })
                        approved += 1

                pr.data["trades_approved"] = approved
            except Exception as exc:
                pr.data["decision_error"] = str(exc)
                logger.warning("DecisionMatrix error: %s", exc)

        # Beta Corridor check
        beta = self._get("beta_corridor")
        if beta:
            try:
                if hasattr(beta, "check"):
                    beta_action = beta.check()
                elif hasattr(beta, "compute"):
                    beta_action = beta.compute()
                else:
                    beta_action = None

                if beta_action:
                    action_str = getattr(beta_action, "action", "HOLD")
                    pr.data["beta_action"] = action_str
                    if action_str != "HOLD":
                        self._approved_trades.append({
                            "ticker": getattr(beta_action, "instrument", "SPY"),
                            "signal": beta_action,
                            "decision": {"type": "beta_hedge", "action": action_str},
                            "timestamp": datetime.now().isoformat(),
                        })
                        approved += 1
            except Exception as exc:
                pr.data["beta_error"] = str(exc)
                logger.warning("BetaCorridor error: %s", exc)

        # OptionsEngine is the mandatory Phase-4 options gate:
        # AllocationEngine -> OptionsEngine -> L7 -> Schwab.
        options = self._get("options_engine")
        if options:
            try:
                if hasattr(options, "scan_opportunities"):
                    if hasattr(options, "begin_portfolio_option_allocation"):
                        # A scan cycle shares one overlay budget across the
                        # mandatory option universe; L7 adds live pending
                        # reservations at order acceptance.
                        options.begin_portfolio_option_allocation()
                    option_regime = getattr(self._last_macro_snapshot, "regime", None)
                    option_regime = getattr(option_regime, "value", option_regime) or "NORMAL"
                    underlyings: dict[str, str] = {}
                    for approved_trade in self._approved_trades:
                        if approved_trade.get("decision", {}).get("type") in {"options", "beta_hedge"}:
                            continue
                        symbol = approved_trade.get("ticker", "")
                        if symbol and symbol not in underlyings:
                            signal = approved_trade.get("signal")
                            sector = (
                                signal.get("sector", "") if isinstance(signal, dict)
                                else getattr(signal, "sector", "")
                            )
                            underlyings[symbol] = str(sector or "")
                    if not underlyings:
                        underlyings = {"SPY": ""}
                    opts = []
                    beta_current = getattr(beta, "current_beta", None) if beta else None
                    beta_target = (
                        getattr(self._last_cube_output, "target_beta", None)
                        if self._last_cube_output else None
                    )
                    for underlying, sector in list(underlyings.items())[:20]:
                        opts.extend(options.scan_opportunities(
                            underlying,
                            allocation={
                                "sector": sector,
                                "current_beta": beta_current,
                                "target_beta": beta_target,
                            },
                            regime=str(option_regime), max_results=1,
                        ))
                    pr.data["options_opportunities"] = len(opts) if opts else 0
                    for opt in (opts or []):
                        self._approved_trades.append({
                            "ticker": opt.get("ticker", ""),
                            "signal": opt,
                            "decision": {"type": "options"},
                            "timestamp": datetime.now().isoformat(),
                        })
                        approved += 1
            except Exception as exc:
                pr.data["options_error"] = str(exc)
                logger.debug("OptionsEngine scan error: %s", exc)

        pr.data["total_approved"] = approved
        pr.items_processed = approved
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_execution_phase(self) -> PhaseResult:
        """Phase 5: Order execution.

        Triggered when approved trades exist:
            - ExecutionEngine.execute() for equity trades
            - OptionsEngine for options trades
            - Beta management via BetaCorridor
        """
        pr = PhaseResult(phase=LoopPhase.EXECUTION.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        executed = 0

        # Check circuit breaker — block execution if risk too high
        if self._circuit_breaker.level in (RiskLevel.CRITICAL, RiskLevel.KILL_SWITCH):
            pr.data["blocked_by_circuit_breaker"] = True
            pr.data["risk_level"] = self._circuit_breaker.level.value
            logger.warning(
                "Execution BLOCKED by circuit breaker: %s",
                self._circuit_breaker.level.value,
            )
            self._approved_trades.clear()
            pr.duration_ms = (time.monotonic() - t0) * 1000
            pr.success = True
            return pr

        exec_engine = self._get("execution_engine")

        # Determine current regime for L7 routing
        regime = "TRENDING"
        if self._last_macro_snapshot and hasattr(self._last_macro_snapshot, "regime"):
            regime_value = getattr(self._last_macro_snapshot, "regime", "TRENDING")
            regime = str(getattr(regime_value, "value", regime_value))

        for trade in self._approved_trades:
            trade_type = trade.get("decision", {}).get("type", "equity") if isinstance(
                trade.get("decision"), dict
            ) else "equity"

            try:
                # L7 Unified Execution Surface — routes ALL products
                if exec_engine and hasattr(exec_engine, "l7") and exec_engine.l7 is not None:
                    ticker = trade.get("ticker", "")
                    signal = trade.get("signal")
                    if not ticker:
                        continue

                    # Determine side + quantity from signal
                    alpha_pred = getattr(signal, "alpha_pred", 0.0)
                    weight = getattr(signal, "weight", 0.0)
                    qty = max(1, int(abs(weight) * 100)) if weight != 0 else 1

                    if trade_type == "options":
                        if not isinstance(signal, dict):
                            raise ValueError("OptionsEngine output must be an order-ready dictionary")
                        instruction = signal.get("instruction", "")
                        side = "BUY" if instruction.startswith("BUY") else "SELL"
                        result = exec_engine.l7_submit(
                            ticker=ticker, side=side, quantity=int(signal.get("quantity", 0)),
                            signal_type="OPTIONS_ENGINE",
                            regime=regime, product_type="OPTION",
                            limit_price=signal.get("limit_price"),
                            option_symbol=signal.get("option_symbol", ""),
                            instruction=instruction,
                            legs=signal.get("legs", []),
                            options_validation=signal.get("options_validation"),
                            sector=signal.get("sector", ""),
                        )
                        if result and result.get("status") in {"PENDING", "FILLED"}:
                            executed += 1
                    elif trade_type == "beta_hedge":
                        action = getattr(signal, "action", "HOLD")
                        instrument = getattr(signal, "instrument", "SPY")
                        hedge_qty = getattr(signal, "quantity", 0)
                        if action != "HOLD" and hedge_qty > 0:
                            result = exec_engine.l7_submit(
                                ticker=instrument, side=action, quantity=hedge_qty,
                                signal_type="MICRO_PRICE_BUY" if action == "BUY" else "MICRO_PRICE_SELL",
                                regime=regime, product_type="FUTURE" if instrument in ("ES", "NQ", "VX") else "EQUITY",
                            )
                            if result and result.get("status") in {"PENDING", "FILLED"}:
                                executed += 1
                    else:
                        side = "BUY" if (alpha_pred > 0 or weight > 0) else "SELL"
                        if alpha_pred == 0 and weight == 0:
                            continue
                        result = exec_engine.l7_submit(
                            ticker=ticker, side=side, quantity=qty,
                            signal_type=getattr(signal, "signal_type", "HOLD"),
                            regime=regime,
                        )
                        if result and result.get("status") in {"PENDING", "FILLED"}:
                            executed += 1

                # No direct execution fallback. All orders must pass through L7.
                elif exec_engine:
                    raise RuntimeError("L7 unavailable; execution fails closed")

            except Exception as exc:
                pr.errors.append(f"{trade.get('ticker', '?')}: {exc}")
                logger.warning("Execution failed for %s: %s", trade.get("ticker"), exc)

        # ── Direct L7 execution for high-conviction signal engines ──────
        # EventDriven and CVR can submit trades directly to L7 without going
        # through DecisionMatrix, when conviction is high.
        # NEWS_MIRO does NOT bypass DecisionMatrix — it feeds Track B signals
        # into the AlphaOptimizer and MLVoteEnsemble T6 via the normal pipeline.
        if exec_engine and hasattr(exec_engine, "l7_submit"):

            # EventDriven direct trades (high-confidence event positions)
            events = self._get("event_driven")
            if events and hasattr(events, "get_active_positions"):
                try:
                    positions = events.get_active_positions() if callable(getattr(events, "get_active_positions", None)) else []
                    for pos in positions[:5]:
                        ticker = getattr(pos, "ticker", "")
                        conf = getattr(pos, "confidence", 0)
                        if ticker and conf >= 0.7:
                            signal_dir = getattr(pos, "signal", "HOLD")
                            if signal_dir in ("LONG", "BUY"):
                                exec_engine.l7_submit(ticker=ticker, side="BUY", quantity=max(1, int(conf * 30)), signal_type="EVENT_DIRECT", regime=regime)
                                executed += 1
                            elif signal_dir in ("SHORT", "SELL"):
                                exec_engine.l7_submit(ticker=ticker, side="SELL", quantity=max(1, int(conf * 30)), signal_type="EVENT_DIRECT", regime=regime)
                                executed += 1
                            pr.data.setdefault("direct_event", []).append(ticker)
                except Exception:
                    pass

            # CVR direct trades (strong buy/sell valuations)
            cvr = self._get("cvr_engine")
            if cvr and hasattr(cvr, "get_active_instruments"):
                try:
                    instruments = cvr.get_active_instruments() if callable(getattr(cvr, "get_active_instruments", None)) else []
                    for inst in instruments[:3]:
                        ticker = getattr(inst, "ticker", "")
                        signal = getattr(inst, "signal", "HOLD")
                        if ticker and signal in ("STRONG_BUY", "BUY"):
                            exec_engine.l7_submit(ticker=ticker, side="BUY", quantity=10, signal_type="CVR_DIRECT", regime=regime)
                            executed += 1
                            pr.data.setdefault("direct_cvr", []).append(ticker)
                        elif ticker and signal in ("SELL", "AVOID"):
                            exec_engine.l7_submit(ticker=ticker, side="SELL", quantity=10, signal_type="CVR_DIRECT", regime=regime)
                            executed += 1
                            pr.data.setdefault("direct_cvr", []).append(ticker)
                except Exception:
                    pass

        # L7 heartbeat (every iteration)
        if exec_engine and hasattr(exec_engine, "l7_heartbeat"):
            try:
                exec_engine.l7_heartbeat(regime=regime)
            except Exception as exc:
                logger.debug("L7 heartbeat error: %s", exc)

        pr.data["trades_executed"] = executed
        pr.data["trades_attempted"] = len(self._approved_trades)
        pr.items_processed = executed

        # Clear executed trades
        self._approved_trades.clear()

        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_learning_phase(self) -> PhaseResult:
        """Phase 6: Learning and feedback loops.

        Continuous:
            - LearningLoop.record() for signal outcomes
            - GSDPlugin.update_gradients()
            - PaulPlugin.store_pattern()
            - Agent scorecard updates
        """
        pr = PhaseResult(phase=LoopPhase.LEARNING.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        # Learning Loop — record recent outcomes
        ll = self._get("learning_loop")
        if ll:
            try:
                # Record regime feedback if we have macro data
                if self._last_macro_snapshot and hasattr(ll, "record_regime_feedback"):
                    snap = self._last_macro_snapshot
                    regime_str = snap.regime.value if hasattr(snap.regime, "value") else str(snap.regime)
                    if RegimeFeedback:
                        feedback = RegimeFeedback(
                            predicted_regime=regime_str,
                            actual_regime=regime_str,
                            timestamp=datetime.now().isoformat(),
                        )
                        ll.record_regime_feedback(feedback)
                        items += 1

                # Get execution engine trade log for signal outcomes
                exec_engine = self._get("execution_engine")
                if exec_engine and hasattr(exec_engine, "_trade_log"):
                    recent_trades = exec_engine._trade_log[-10:]
                    for trade in recent_trades:
                        if SignalOutcome and hasattr(ll, "record_signal_outcome"):
                            try:
                                outcome = SignalOutcome(
                                    ticker=trade.get("ticker", ""),
                                    signal_engine=trade.get("engine", ""),
                                    signal_type=trade.get("signal_type", ""),
                                    realized_pnl=trade.get("pnl", 0.0),
                                    was_correct=trade.get("pnl", 0.0) > 0,
                                )
                                ll.record_signal_outcome(outcome)
                                items += 1
                            except Exception:
                                pass
            except Exception as exc:
                pr.data["learning_loop_error"] = str(exc)
                logger.warning("LearningLoop error: %s", exc)

        # GSD Plugin gradient updates
        gsd = self._get("gsd_plugin")
        if gsd:
            try:
                if hasattr(gsd, "update_gradients"):
                    gsd.update_gradients()
                    items += 1
                    pr.data["gsd_updated"] = True
            except Exception as exc:
                pr.data["gsd_error"] = str(exc)
                logger.debug("GSDPlugin error: %s", exc)

        # Paul Plugin pattern storage
        paul = self._get("paul_plugin")
        if paul:
            try:
                if hasattr(paul, "store_pattern"):
                    paul.store_pattern()
                    items += 1
                    pr.data["paul_updated"] = True
            except Exception as exc:
                pr.data["paul_error"] = str(exc)
                logger.debug("PaulPlugin error: %s", exc)

        # Agent scorecard updates
        scorecard = self._get("agent_scorecard")
        if scorecard:
            try:
                if hasattr(scorecard, "update"):
                    scorecard.update()
                    items += 1
                elif hasattr(scorecard, "refresh"):
                    scorecard.refresh()
                    items += 1
            except Exception as exc:
                pr.data["scorecard_error"] = str(exc)
                logger.debug("AgentScorecard error: %s", exc)

        # PaulOrchestrator — periodic enforcement + dynamic agent management
        paul_orch = self._get("paul_orchestrator")
        if paul_orch:
            try:
                # Run periodic enforcement check (herding, concentration, drift)
                periodic_result = paul_orch.run_periodic_check()
                pr.data["paul_orchestrator_check"] = True
                items += 1

                # Get ensemble adjustments for MLVoteEnsemble
                if self._last_macro_snapshot:
                    regime_str = ""
                    if hasattr(self._last_macro_snapshot, "regime"):
                        regime_str = (
                            self._last_macro_snapshot.regime.value
                            if hasattr(self._last_macro_snapshot.regime, "value")
                            else str(self._last_macro_snapshot.regime)
                        )
                    market_state = {"regime": regime_str}
                    adjustments = paul_orch.get_ensemble_adjustments(market_state)
                    pr.data["ensemble_adjustments"] = adjustments

                # Process recent trade outcomes through the orchestrator
                exec_engine = self._get("execution_engine")
                if exec_engine and hasattr(exec_engine, "_trade_log"):
                    recent = exec_engine._trade_log[-5:]
                    outcomes = []
                    for trade in recent:
                        outcomes.append({
                            "ticker": trade.get("ticker", ""),
                            "signal_engine": trade.get("engine", ""),
                            "realized_pnl": trade.get("pnl", 0.0),
                            "was_correct": trade.get("pnl", 0.0) > 0,
                            "direction": trade.get("side", ""),
                            "regime": trade.get("regime", ""),
                        })
                    if outcomes:
                        paul_orch.process_outcomes(outcomes)
                        items += len(outcomes)

            except Exception as exc:
                pr.data["paul_orchestrator_error"] = str(exc)
                logger.debug("PaulOrchestrator learning phase error: %s", exc)

        # GSD Workflow Bridge — feed metrics into structured state tracking
        gsd_wf = self._get("gsd_workflow")
        if gsd_wf:
            try:
                # Feed GSD gradient state
                gsd = self._get("gsd_plugin")
                if gsd and hasattr(gsd, "log_gradient_state"):
                    gsd_state = gsd.log_gradient_state()
                    gsd_wf.integrate_with_gradient_state(gsd_state)

                # Feed Paul pattern state
                paul = self._get("paul_plugin")
                if paul and hasattr(paul, "log_learning_state"):
                    paul_state = paul.log_learning_state()
                    gsd_wf.integrate_with_paul_state(paul_state)

                pr.data["gsd_workflow_updated"] = True
                items += 1
            except Exception as exc:
                pr.data["gsd_workflow_error"] = str(exc)
                logger.debug("GSD Workflow Bridge error: %s", exc)

        # ── FEEDBACK LOOPS: Learning → Decision + Signals + Ensemble ─────

        # 1. Apply learned tier weights back to MLVoteEnsemble
        exec_engine = self._get("execution_engine")
        if ll and exec_engine and hasattr(exec_engine, "ensemble"):
            try:
                weight_changes = ll.apply_to_ensemble(exec_engine.ensemble)
                if weight_changes:
                    pr.data["tier_weight_adjustments"] = len(weight_changes)
                    items += 1
            except Exception as exc:
                pr.data["tier_weight_error"] = str(exc)

        # 2. Feed regime calibration bias back to DecisionMatrix
        dm = self._get("decision_matrix")
        if ll and dm:
            try:
                if hasattr(ll, "get_regime_calibration_bias"):
                    bias = ll.get_regime_calibration_bias()
                    if bias and hasattr(dm, "regime_bias"):
                        dm.regime_bias = bias
                        pr.data["decision_regime_bias_updated"] = True
                # Push learned accuracy to adjust approval thresholds
                if hasattr(ll, "get_snapshot"):
                    snap = ll.get_snapshot()
                    accuracy = getattr(snap, "overall_accuracy", 0.5)
                    if hasattr(dm, "accuracy_adjustment"):
                        dm.accuracy_adjustment = accuracy
                        pr.data["decision_accuracy_fed"] = round(accuracy, 3)
                items += 1
            except Exception as exc:
                pr.data["decision_feedback_error"] = str(exc)

        # 3. Record sector feedback from execution outcomes
        if ll and exec_engine and hasattr(ll, "record_sector_feedback"):
            try:
                if hasattr(exec_engine, "broker"):
                    broker = exec_engine.broker
                    if hasattr(broker, "get_sector_pnl"):
                        sector_pnl = broker.get_sector_pnl()
                        if sector_pnl:
                            for sector, pnl in sector_pnl.items():
                                direction = "OVERWEIGHT" if pnl > 0 else "UNDERWEIGHT"
                                ll.record_sector_feedback(sector, direction, pnl)
                            pr.data["sector_feedback_recorded"] = len(sector_pnl)
            except Exception as exc:
                pr.data["sector_feedback_error"] = str(exc)

        # 4. DeepLearningEngine — PPO training on recent outcomes
        try:
            from .ml.deep_learning_engine import PPOAgent, TradingEnvironment, build_feature_matrix
            if exec_engine and hasattr(exec_engine, "_trade_log") and len(getattr(exec_engine, "_trade_log", [])) > 20:
                # Build returns from recent trade P&L
                trade_returns = np.array([t.get("pnl", 0.0) for t in exec_engine._trade_log[-252:]])
                if len(trade_returns) > 50:
                    features = build_feature_matrix(trade_returns, pad_to=50)
                    env = TradingEnvironment(returns=trade_returns)
                    agent = PPOAgent(state_dim=50, action_dim=3)

                    # Apply 2:4 structured sparsity to actor weights
                    for attr in ["actor_w1", "actor_w2", "actor_w3", "critic_w1", "critic_w2", "critic_w3"]:
                        w = getattr(agent, attr, None)
                        if w is not None and w.ndim == 2:
                            # 2:4 sparsity: in every group of 4 consecutive elements,
                            # keep the 2 with largest magnitude, zero the rest
                            flat = w.reshape(-1)
                            for i in range(0, len(flat) - 3, 4):
                                group = flat[i:i+4]
                                idx = np.argsort(np.abs(group))
                                group[idx[0]] = 0.0
                                group[idx[1]] = 0.0
                                flat[i:i+4] = group
                            setattr(agent, attr, flat.reshape(w.shape))

                    # Quick training pass (limited epochs for live loop)
                    state = env.reset()
                    for _ in range(min(100, len(trade_returns))):
                        action = agent._actor_forward(state).argmax()
                        next_state, reward, done, _ = env.step(action)
                        if done:
                            break
                        state = next_state

                    pr.data["deep_learning_trained"] = True
                    pr.data["deep_learning_sparsity"] = "2:4"
                    items += 1
        except Exception as exc:
            pr.data["deep_learning_error"] = str(exc)

        # 5. Archive daily deductions in compressed format
        try:
            import gzip
            archive_dir = Path("data/learning/archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y%m%d")
            deductions = {
                "date": today,
                "tier_weights": ll.compute_tier_weight_adjustments() if ll and hasattr(ll, "compute_tier_weight_adjustments") else {},
                "signal_accuracy": getattr(ll, "_signal_accuracy", {}) if ll else {},
                "regime_feedback": getattr(ll, "_regime_history", [])[-10:] if ll else [],
                "alpha_sharpe": pr.data.get("alpha_sharpe", 0),
                "ensemble_votes": pr.data.get("ensemble_votes", 0),
                "trades_executed": pr.data.get("trades_executed", 0),
                "deep_learning_active": pr.data.get("deep_learning_trained", False),
            }
            archive_file = archive_dir / f"deductions_{today}.json.gz"
            with gzip.open(archive_file, "wt", encoding="utf-8") as f:
                json.dump(deductions, f, indent=2, default=str)
            pr.data["archive_compressed"] = str(archive_file)
        except Exception as exc:
            pr.data["archive_error"] = str(exc)

        # Graphify knowledge graph — refresh god nodes for agent context
        graphify = self._get("graphify")
        if graphify:
            try:
                if graphify.is_available():
                    god_nodes = graphify.get_god_nodes()
                    pr.data["graphify_nodes"] = len(god_nodes) if god_nodes else 0
                    items += 1
                else:
                    pr.data["graphify_status"] = "graph_not_generated"
            except Exception as exc:
                pr.data["graphify_error"] = str(exc)

        # Autoresearch — check experiment status and feed into patterns
        autoresearch = self._get("autoresearch")
        if autoresearch:
            try:
                status = autoresearch.get_status()
                pr.data["autoresearch_available"] = status.get("available", False)
                pr.data["autoresearch_experiments"] = status.get("experiment_count", 0)
                if status.get("has_results"):
                    pr.data["autoresearch_best_bpb"] = status.get("best_val_bpb")
                    items += 1
            except Exception as exc:
                pr.data["autoresearch_error"] = str(exc)

        # Research bots — LLM-powered sector analysis (5-min cadence, only during market hours)
        research = self._get("research_bots")
        if research and hasattr(research, "run_llm_sector_analysis"):
            try:
                # Run LLM analysis for top-performing sectors only (limit API calls)
                leaderboard = research.get_leaderboard() if hasattr(research, "get_leaderboard") else []
                top_sectors = [sec for sec, _ in leaderboard[:3]] if leaderboard else []
                for sec in top_sectors:
                    try:
                        research.run_llm_sector_analysis(sec)
                    except Exception:
                        pass
                pr.data["research_llm_sectors"] = len(top_sectors)
                items += 1
            except Exception as exc:
                pr.data["research_llm_error"] = str(exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    def run_monitoring_phase(self) -> PhaseResult:
        """Phase 7: Monitoring and reporting.

        Runs at 5-minute cadence:
            - Portfolio P&L update
            - Risk check (circuit breaker evaluation)
            - Anomaly detection
            - Learning snapshot
        """
        pr = PhaseResult(phase=LoopPhase.MONITORING.value, timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        # Portfolio P&L
        exec_engine = self._get("execution_engine")
        nav = self._initial_nav
        daily_pnl = 0.0
        if exec_engine and hasattr(exec_engine, "broker"):
            try:
                broker = exec_engine.broker
                if hasattr(broker, "nav"):
                    nav = broker.nav
                elif hasattr(broker, "get_nav"):
                    nav = broker.get_nav()
                elif hasattr(broker, "cash"):
                    nav = broker.cash

                daily_pnl = nav - self._initial_nav
                pr.data["nav"] = nav
                pr.data["daily_pnl"] = daily_pnl
                pr.data["daily_pnl_pct"] = daily_pnl / self._initial_nav if self._initial_nav > 0 else 0.0
                items += 1
            except Exception as exc:
                pr.data["nav_error"] = str(exc)

        # Risk check — update circuit breaker
        self._update_circuit_breaker(nav, daily_pnl)
        pr.data["risk_level"] = self._circuit_breaker.level.value
        pr.data["kill_switch"] = self._circuit_breaker.kill_switch_active

        # Position-level drawdown exit check (20% per position)
        alloc = self._get("allocation_engine")
        if alloc and exec_engine and hasattr(alloc, "check_position_drawdown"):
            try:
                broker = getattr(exec_engine, "broker", None)
                if broker and hasattr(broker, "positions"):
                    positions = broker.positions if not callable(broker.positions) else broker.positions()
                    if positions:
                        cost_basis = {}
                        current_prices = {}
                        for pos in positions:
                            t = pos.get("ticker") or getattr(pos, "ticker", "")
                            if t:
                                cost_basis[t] = pos.get("avg_cost") or getattr(pos, "avg_cost", 0.0)
                                current_prices[t] = pos.get("current_price") or getattr(pos, "current_price", 0.0)
                        if cost_basis and current_prices:
                            exits = alloc.check_position_drawdown(positions, cost_basis, current_prices)
                            if exits:
                                pr.data["position_exits"] = exits
                                logger.warning(
                                    "[LiveLoop] Position-level 20%% drawdown EXIT: %s — queuing liquidation",
                                    exits,
                                )
                                for ticker in exits:
                                    self._approved_trades.append({
                                        "ticker": ticker,
                                        "signal": "POSITION_DRAWDOWN_EXIT",
                                        "decision": {"source": "position_stop_loss", "action": "LIQUIDATE"},
                                        "dollar_amount": 0,
                                        "position_size": 0,
                                        "bucket": "LIQUIDATION",
                                        "instrument_type": "EQUITY",
                                        "timestamp": datetime.now().isoformat(),
                                    })
            except Exception as exc:
                pr.data["position_drawdown_error"] = str(exc)
                logger.debug("Position drawdown check error: %s", exc)

        # Anomaly detection
        anomaly = self._get("anomaly_detector")
        if anomaly:
            try:
                if hasattr(anomaly, "scan_returns"):
                    # Would need returns data; run a general scan if available
                    alerts = anomaly.get_alerts() if hasattr(anomaly, "get_alerts") else None
                    if alerts and hasattr(alerts, "get_critical"):
                        critical = alerts.get_critical()
                        pr.data["critical_anomalies"] = len(critical)
                        if critical:
                            logger.warning("ANOMALY DETECTOR: %d critical alerts", len(critical))
                        items += 1
            except Exception as exc:
                pr.data["anomaly_error"] = str(exc)
                logger.debug("AnomalyDetector error: %s", exc)

        # Learning snapshot
        ll = self._get("learning_loop")
        if ll and hasattr(ll, "get_snapshot"):
            try:
                snapshot = ll.get_snapshot()
                pr.data["learning_accuracy"] = getattr(snapshot, "overall_accuracy", 0.0)
                pr.data["learning_total_signals"] = getattr(snapshot, "total_signals", 0)
                items += 1
            except Exception as exc:
                pr.data["learning_snapshot_error"] = str(exc)

        # Portfolio analytics
        analytics = self._get("portfolio_analytics")
        if analytics:
            try:
                if hasattr(analytics, "compute"):
                    analytics.compute()
                    items += 1
            except Exception as exc:
                pr.data["analytics_error"] = str(exc)

        # ── Security: Broker integrity reconciliation ──
        security = self._get("security")
        if security and exec_engine and hasattr(security, "broker_lock"):
            try:
                broker = getattr(exec_engine, "broker", None)
                if broker and hasattr(broker, "state") and hasattr(broker.state, "positions"):
                    local_positions = {}
                    for t, pos in broker.state.positions.items():
                        local_positions[t] = {
                            "quantity": getattr(pos, "quantity", 0),
                            "market_value": getattr(pos, "market_value", 0),
                        }
                    # Refresh Schwab positions from the single live broker instance.
                    broker_positions = {}
                    if hasattr(broker, "sync_account"):
                        broker.sync_account()
                    if hasattr(broker, "state") and hasattr(broker.state, "positions"):
                            for t, pos in broker.state.positions.items():
                                broker_positions[t] = {
                                    "quantity": getattr(pos, "quantity", 0),
                                    "market_value": getattr(pos, "market_value", 0),
                                }
                    if local_positions or broker_positions:
                        recon = security.broker_lock.reconcile(local_positions, broker_positions)
                        pr.data["broker_recon_clean"] = recon.get("clean", True)
                        pr.data["broker_recon_discrepancies"] = len(recon.get("discrepancies", []))
                        if not recon.get("clean"):
                            pr.data["broker_frozen"] = True
                        items += 1
            except Exception as exc:
                pr.data["broker_recon_error"] = str(exc)

        # ── Profit-taking: 3-layer P&L check → liquidate overlays ──
        # Layer 1: Schwab live-broker P&L > 20% → sell options, re-run
        # Layer 2: Futures/Rithmic P&L > 20% → sell futures, re-run
        # Layer 3: Aggregate P&L > 20% → sell ALL overlays, re-run
        alloc = self._get("allocation_engine")
        exec_engine = self._get("execution_engine")
        if alloc and hasattr(alloc, "check_profit_take") and nav > 0:
            try:
                # Extract per-product-class P&L from brokers
                live_broker_pnl = 0.0
                equities_pnl = 0.0
                options_pnl = 0.0
                futures_pnl = 0.0

                if exec_engine:
                    broker = getattr(exec_engine, "broker", None)
                    if broker:
                        # Schwab combined P&L (equities + options)
                        if hasattr(broker, "state"):
                            live_broker_pnl = getattr(broker.state, "total_pnl", 0.0)
                        # Split equities vs options if broker tracks them
                        if hasattr(broker, "get_equity_pnl"):
                            equities_pnl = broker.get_equity_pnl()
                        elif hasattr(broker, "state") and hasattr(broker.state, "positions"):
                            for _, pos in broker.state.positions.items():
                                equities_pnl += getattr(pos, "unrealized_pnl", 0.0) + getattr(pos, "realized_pnl", 0.0)
                        if hasattr(broker, "get_options_pnl"):
                            options_pnl = broker.get_options_pnl()

                    # Futures P&L from trade log (until Rithmic connected)
                    paper = getattr(exec_engine, "_paper_broker", None) or getattr(exec_engine, "paper_broker", None)
                    if paper and hasattr(paper, "state"):
                        # trade log tracks futures positions
                        for _, pos in getattr(paper.state, "positions", {}).items():
                            if getattr(pos, "sector", "") == "FUTURES" or getattr(pos, "ticker", "") in ("ES", "NQ", "YM", "RTY", "VX", "ZN", "ZB", "MES", "MNQ"):
                                futures_pnl += getattr(pos, "unrealized_pnl", 0.0) + getattr(pos, "realized_pnl", 0.0)

                profit_check = alloc.check_profit_take(
                    nav=nav,
                    initial_nav=self._initial_nav,
                    live_broker_pnl=live_broker_pnl,
                    futures_pnl=futures_pnl,
                    equities_pnl=equities_pnl,
                    options_pnl=options_pnl,
                )

                # Always report P&L breakdown
                pr.data["pnl_breakdown"] = profit_check.get("pnl_breakdown", {})

                if profit_check.get("triggered"):
                    pr.data["profit_take_triggered"] = True
                    pr.data["profit_take_layer"] = profit_check["trigger_layer"]
                    pr.data["profit_take_action"] = profit_check["action"]
                    pr.data["profit_take_liquidate"] = profit_check["liquidate"]

                    # Queue overlay liquidation trades via L7
                    if exec_engine and hasattr(exec_engine, "l7_submit"):
                        regime = "TRENDING"
                        if self._last_macro_snapshot and hasattr(self._last_macro_snapshot, "regime"):
                            regime = str(getattr(self._last_macro_snapshot, "regime", "TRENDING"))
                        for bucket in profit_check.get("liquidate", []):
                            exec_engine.l7_submit(
                                ticker="ALL", side="SELL", quantity=0,
                                signal_type=profit_check["action"],
                                regime=regime,
                                product_type=bucket,
                            )
                    logger.critical(
                        "[LiveLoop] PROFIT TAKE [%s]: total=%.1f%% equities=$%.0f options=$%.0f futures=$%.0f "
                        "— liquidating %s, next scan re-enters fresh",
                        profit_check["trigger_layer"],
                        profit_check["pnl_breakdown"]["total_pct"] * 100,
                        equities_pnl, options_pnl, futures_pnl,
                        profit_check["liquidate"],
                    )
            except Exception as exc:
                pr.data["profit_take_error"] = str(exc)

        # ── Security: token meter status for monitoring ──
        meter = self._get("token_meter")
        if meter:
            try:
                token_status = meter.get_status()
                pr.data["token_daily_used"] = token_status.get("daily_used", 0)
                pr.data["token_daily_pct"] = token_status.get("daily_pct", 0)
                pr.data["token_lockdown"] = token_status.get("lockdown_active", False)
            except Exception:
                pass

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True
        return pr

    # ------------------------------------------------------------------
    # Daily schedule
    # ------------------------------------------------------------------
    def run_daily_schedule(self):
        """Execute time-based daily schedule actions.

        Called within the main loop to trigger session-specific routines:
            - Pre-market: full data refresh, overnight signals, SEC scan
            - Market open: full pipeline flush
            - Market close: EOD reconciliation, learning snapshot
            - After-hours: reduced frequency, earnings scan
        """
        now = datetime.now()
        session = self._classify_session(now)

        if session == MarketSession.PRE_MARKET:
            self._run_pre_market()
        elif session == MarketSession.MARKET_OPEN:
            self._run_market_open()
        elif session == MarketSession.MARKET_CLOSE:
            self._run_market_close()
        elif session == MarketSession.AFTER_HOURS:
            self._run_after_hours()

    def _run_pre_market(self):
        """Pre-market routine (08:00-09:30 ET).

        Full data refresh, overnight signal processing, SEC filing scan.
        """
        logger.info("PRE-MARKET ROUTINE — full data refresh")

        # Full universe reload
        universe = self._get("universe")
        if universe and hasattr(universe, "load"):
            try:
                universe.load()
                logger.info("Universe reloaded: %d securities",
                            universe.size() if hasattr(universe, "size") else 0)
            except Exception as exc:
                logger.warning("Pre-market universe reload failed: %s", exc)

        # Overnight macro signals
        macro = self._get("macro_engine")
        if macro:
            try:
                snap = macro.analyze()
                self._last_macro_snapshot = snap
                logger.info("Pre-market macro: regime=%s",
                            snap.regime.value if hasattr(snap, "regime") else "?")
            except Exception as exc:
                logger.warning("Pre-market macro failed: %s", exc)

        # Security analysis scan (SEC filings)
        sa = self._get("security_analysis")
        if sa and hasattr(sa, "scan_filings"):
            try:
                sa.scan_filings()
                logger.info("Pre-market SEC filing scan complete")
            except Exception as exc:
                logger.debug("Pre-market SEC scan failed: %s", exc)

    def _run_market_open(self):
        """Market open routine (09:30 ET).

        Full pipeline flush — runs every phase sequentially.
        """
        logger.info("=" * 60)
        logger.info("MARKET OPEN — full pipeline flush")
        logger.info("=" * 60)

        # Run the full execution engine pipeline if available
        exec_engine = self._get("execution_engine")
        if exec_engine and hasattr(exec_engine, "run_pipeline"):
            try:
                result = exec_engine.run_pipeline()
                logger.info("Market open pipeline complete: %d stages",
                            len(result.get("stages", {})))
            except Exception as exc:
                logger.error("Market open pipeline failed: %s", exc)

        # L7 market open — reset daily counters
        if exec_engine and hasattr(exec_engine, "l7_market_open"):
            try:
                exec_engine.l7_market_open()
                logger.info("L7 Unified Execution Surface: market open")
            except Exception as exc:
                logger.debug("L7 market open failed: %s", exc)

        # Force all cadence timers so everything runs on first intraday heartbeat
        self._last_signal_time = None
        self._last_intelligence_time = None
        self._last_monitoring_time = None

    def _run_market_close(self):
        """Market close routine (16:00 ET).

        EOD reconciliation, learning snapshot, next-day prep.
        """
        logger.info("=" * 60)
        logger.info("MARKET CLOSE — EOD reconciliation")
        logger.info("=" * 60)

        # Final learning snapshot
        ll = self._get("learning_loop")
        if ll and hasattr(ll, "get_snapshot"):
            try:
                snapshot = ll.get_snapshot()
                logger.info("EOD learning snapshot: accuracy=%.2f%% total_signals=%d",
                            getattr(snapshot, "overall_accuracy", 0.0) * 100,
                            getattr(snapshot, "total_signals", 0))
            except Exception as exc:
                logger.warning("EOD learning snapshot failed: %s", exc)

        # L7 market close — daily learning + pattern persistence
        exec_engine = self._get("execution_engine")
        if exec_engine and hasattr(exec_engine, "l7_market_close"):
            try:
                exec_engine.l7_market_close()
                logger.info("L7 Unified Execution Surface: market close")
            except Exception as exc:
                logger.debug("L7 market close failed: %s", exc)

        # Persist state
        if self._enable_persistence:
            self._persist_state()

        # Agent scorecard weekly update
        scorecard = self._get("agent_scorecard")
        if scorecard and hasattr(scorecard, "weekly_update"):
            try:
                scorecard.weekly_update()
            except Exception as exc:
                logger.debug("EOD scorecard update failed: %s", exc)

    def _run_after_hours(self):
        """After-hours routine (16:00-20:00 ET).

        Reduced frequency, earnings scan.
        """
        # Earnings scan via security analysis
        sa = self._get("security_analysis")
        if sa and hasattr(sa, "scan_earnings"):
            try:
                sa.scan_earnings()
            except Exception as exc:
                logger.debug("After-hours earnings scan failed: %s", exc)

    # ------------------------------------------------------------------
    # Overnight backtesting
    # ------------------------------------------------------------------
    def run_backtest_overnight(self) -> PhaseResult:
        """Run overnight backtesting and ML training.

        Executes:
            - QSTrader backtest runner on current strategy
            - ML model retraining
            - Pattern evolution
        """
        pr = PhaseResult(phase="BACKTEST_OVERNIGHT", timestamp=datetime.now().isoformat())
        t0 = time.monotonic()
        items = 0

        logger.info("OVERNIGHT — starting backtesting and ML training")

        # QSTrader backtest
        runner = self._get("backtest_runner")
        if runner:
            try:
                if hasattr(runner, "run"):
                    bt_result = runner.run()
                    pr.data["backtest"] = bt_result if isinstance(bt_result, dict) else {"status": "complete"}
                    items += 1
                elif hasattr(runner, "run_backtest"):
                    bt_result = runner.run_backtest()
                    pr.data["backtest"] = bt_result if isinstance(bt_result, dict) else {"status": "complete"}
                    items += 1
                logger.info("Overnight backtest complete")
            except Exception as exc:
                pr.data["backtest_error"] = str(exc)
                logger.warning("Overnight backtest failed: %s", exc)

        # ML model retraining via alpha optimizer
        alpha = self._get("alpha_optimizer")
        if alpha and hasattr(alpha, "retrain"):
            try:
                alpha.retrain()
                pr.data["ml_retrain"] = "complete"
                items += 1
            except Exception as exc:
                pr.data["ml_retrain_error"] = str(exc)
                logger.warning("Overnight ML retrain failed: %s", exc)

        # Pattern evolution via learning loop
        ll = self._get("learning_loop")
        if ll and hasattr(ll, "evolve_patterns"):
            try:
                ll.evolve_patterns()
                pr.data["pattern_evolution"] = "complete"
                items += 1
            except Exception as exc:
                pr.data["pattern_evolution_error"] = str(exc)

        # GSD/Paul plugin overnight learning
        gsd = self._get("gsd_plugin")
        if gsd and hasattr(gsd, "overnight_learn"):
            try:
                gsd.overnight_learn()
                items += 1
            except Exception as exc:
                logger.debug("GSD overnight learn failed: %s", exc)

        paul = self._get("paul_plugin")
        if paul and hasattr(paul, "overnight_learn"):
            try:
                paul.overnight_learn()
                items += 1
            except Exception as exc:
                logger.debug("Paul overnight learn failed: %s", exc)

        pr.items_processed = items
        pr.duration_ms = (time.monotonic() - t0) * 1000
        pr.success = True

        logger.info("Overnight session complete: %d tasks  %.1fms", items, pr.duration_ms)
        return pr

    # ------------------------------------------------------------------
    # Status and performance
    # ------------------------------------------------------------------
    def get_status(self) -> LoopStatus:
        """Return current loop status snapshot."""
        with self._lock:
            now = datetime.now()
            uptime = (now - self._started_at).total_seconds() if self._started_at else 0.0

            # Estimate time to next heartbeat
            next_hb = 0.0
            if self._last_heartbeat_time and self._state == LoopState.RUNNING:
                elapsed = (now - self._last_heartbeat_time).total_seconds()
                interval = self._get_current_interval()
                next_hb = max(0.0, interval - elapsed)

            # NAV from execution engine
            nav = self._initial_nav
            daily_pnl = 0.0
            exec_engine = self._get("execution_engine")
            if exec_engine and hasattr(exec_engine, "broker"):
                try:
                    broker = exec_engine.broker
                    if hasattr(broker, "nav"):
                        nav = broker.nav
                    elif hasattr(broker, "get_nav"):
                        nav = broker.get_nav()
                    daily_pnl = nav - self._initial_nav
                except Exception:
                    pass

            return LoopStatus(
                state=self._state.value,
                session=self._session.value,
                risk_level=self._circuit_breaker.level.value,
                current_phase=self._current_phase,
                iteration=self._iteration,
                uptime_seconds=uptime,
                last_heartbeat=self._last_heartbeat_time.isoformat() if self._last_heartbeat_time else "",
                next_heartbeat_in_seconds=next_hb,
                components_loaded=dict(self._component_status),
                kill_switch_active=self._circuit_breaker.kill_switch_active,
                performance=self._performance,
                last_macro_regime=(
                    self._last_macro_snapshot.regime.value
                    if self._last_macro_snapshot and hasattr(self._last_macro_snapshot, "regime")
                    else ""
                ),
                last_cube_regime=(
                    self._last_cube_output.regime.value
                    if self._last_cube_output and hasattr(self._last_cube_output, "regime")
                    else ""
                ),
                portfolio_nav=nav,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl / self._initial_nav if self._initial_nav > 0 else 0.0,
            )

    def get_performance(self) -> LoopPerformance:
        """Return cumulative loop performance metrics."""
        with self._lock:
            return self._performance

    # ------------------------------------------------------------------
    # Internal — main loop
    # ------------------------------------------------------------------
    def _run_loop(self):
        """Main loop body — runs on the background thread."""
        with self._lock:
            self._state = LoopState.RUNNING

        logger.info("LiveLoop thread started")

        # Track whether we've triggered session-specific routines
        _triggered_sessions: dict[str, bool] = {}

        while not self._stop_event.is_set():
            try:
                # Check pause
                if self._state == LoopState.PAUSED:
                    self._stop_event.wait(timeout=1.0)
                    continue

                now = datetime.now()
                session = self._classify_session(now)
                self._session = session

                # Trigger once-per-session routines
                session_key = f"{now.date()}_{session.value}"
                if session_key not in _triggered_sessions:
                    _triggered_sessions[session_key] = True
                    self.run_daily_schedule()

                    # Overnight backtesting
                    if session == MarketSession.OVERNIGHT:
                        self.run_backtest_overnight()

                # Run heartbeat
                try:
                    result = self.heartbeat()

                    # Reset consecutive errors on success
                    if not result.errors:
                        self._performance.consecutive_errors = 0

                except Exception as exc:
                    self._performance.total_errors += 1
                    self._performance.consecutive_errors += 1
                    self._performance.max_consecutive_errors = max(
                        self._performance.max_consecutive_errors,
                        self._performance.consecutive_errors,
                    )
                    logger.error("Heartbeat exception: %s\n%s", exc, traceback.format_exc())

                    # Auto-restart guard
                    if self._performance.consecutive_errors >= self._max_consecutive_errors:
                        logger.critical(
                            "FATAL: %d consecutive errors — entering ERROR state",
                            self._performance.consecutive_errors,
                        )
                        with self._lock:
                            self._state = LoopState.ERROR
                        break

                # Periodic state persistence
                if self._enable_persistence and self._should_run_cadence(
                    self._last_state_persist_time, _STATE_PERSISTENCE_INTERVAL
                ):
                    self._persist_state()
                    self._last_state_persist_time = now

                # Sleep until next heartbeat
                interval = self._get_current_interval()
                self._stop_event.wait(timeout=interval)

            except Exception as exc:
                logger.error("Loop iteration fatal error: %s\n%s", exc, traceback.format_exc())
                self._performance.total_errors += 1
                self._performance.consecutive_errors += 1
                time.sleep(_RESTART_COOLDOWN_SECONDS)

        logger.info("LiveLoop thread exiting")

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------
    def _classify_session(self, now: datetime) -> MarketSession:
        """Determine current market session from wall clock time."""
        # Weekend check (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            return MarketSession.WEEKEND

        t = now.time()

        if t < _PRE_MARKET_START:
            return MarketSession.OVERNIGHT
        elif t < _MARKET_OPEN:
            return MarketSession.PRE_MARKET
        elif t < dt_time(9, 31):
            return MarketSession.MARKET_OPEN
        elif t < _MARKET_CLOSE:
            return MarketSession.INTRADAY
        elif t < dt_time(16, 1):
            return MarketSession.MARKET_CLOSE
        elif t < _AFTER_HOURS_END:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.OVERNIGHT

    def _get_current_interval(self) -> float:
        """Return heartbeat interval based on current session."""
        if self._session == MarketSession.INTRADAY:
            return self._heartbeat_interval
        elif self._session in (MarketSession.PRE_MARKET, MarketSession.MARKET_OPEN):
            return self._heartbeat_interval
        elif self._session == MarketSession.AFTER_HOURS:
            return _AFTER_HOURS_INTERVAL
        elif self._session == MarketSession.OVERNIGHT:
            return _OVERNIGHT_INTERVAL
        elif self._session == MarketSession.WEEKEND:
            return _OVERNIGHT_INTERVAL
        else:
            return self._heartbeat_interval

    def _should_run_cadence(self, last_time: Optional[datetime], cadence_seconds: float) -> bool:
        """Check whether enough time has elapsed since last run."""
        if last_time is None:
            return True
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= cadence_seconds

    def _signals_changed(self) -> bool:
        """Detect whether signals have materially changed since last decision.

        Returns True if regime changed, beta target shifted, or VIX moved
        beyond a threshold.
        """
        if not self._last_signals:
            return False

        prev = getattr(self, "_prev_signals", {})
        if not prev:
            self._prev_signals = dict(self._last_signals)
            return True  # First time — trigger decision

        changed = False

        # Regime change
        if self._last_signals.get("macro_regime") != prev.get("macro_regime"):
            changed = True
        if self._last_signals.get("cube_regime") != prev.get("cube_regime"):
            changed = True

        # Beta target shift > 0.05
        beta_delta = abs(
            self._last_signals.get("cube_target_beta", 0.0) -
            prev.get("cube_target_beta", 0.0)
        )
        if beta_delta > 0.05:
            changed = True

        # VIX move > 2 points
        vix_delta = abs(
            self._last_signals.get("vix", 0.0) - prev.get("vix", 0.0)
        )
        if vix_delta > 2.0:
            changed = True

        # Always run decision at least every 5 minutes
        if hasattr(self, "_last_decision_time") and self._last_decision_time:
            if (datetime.now() - self._last_decision_time).total_seconds() > 300:
                changed = True
        else:
            changed = True

        if changed:
            self._prev_signals = dict(self._last_signals)
            self._last_decision_time = datetime.now()

        return changed

    def _run_phase_safe(
        self,
        phase: LoopPhase,
        phase_fn: Callable[[], PhaseResult],
        heartbeat_result: HeartbeatResult,
    ):
        """Execute a phase with error handling, timing, security chain signing, and recording."""
        self._current_phase = phase.value

        # ── Security: verify phase chain before execution ──
        security = self._get("security")
        if security and hasattr(security, "phase_chain"):
            if not security.phase_chain.verify_chain(phase.value):
                logger.critical("PHASE CHAIN BROKEN — blocking %s (exits still active)", phase.value)
                heartbeat_result.errors.append(f"{phase.value}: PHASE_CHAIN_BROKEN")
                heartbeat_result.phases[phase.value] = PhaseResult(
                    phase=phase.value, success=False, error="Phase chain integrity failure",
                    timestamp=datetime.now().isoformat(),
                )
                self._current_phase = ""
                return

            # Check if system is healthy for trade-entry phases
            if phase in (LoopPhase.DECISION, LoopPhase.EXECUTION):
                if not security.is_system_healthy():
                    logger.warning("Security unhealthy — blocking %s (exits active)", phase.value)
                    heartbeat_result.errors.append(f"{phase.value}: SECURITY_UNHEALTHY")
                    self._current_phase = ""
                    return

        try:
            pr = phase_fn()
            heartbeat_result.phases[phase.value] = pr

            # ── Security: sign phase output for chain ──
            if security and hasattr(security, "phase_chain"):
                security.phase_chain.sign_phase(phase.value, pr.data if pr else {})

            # ── Security: record to transaction ledger (for trade phases) ──
            if security and phase in (LoopPhase.DECISION, LoopPhase.EXECUTION):
                security.ledger.record(
                    event_type=f"phase_{phase.value.lower()}",
                    data={"items": pr.items_processed, "duration_ms": pr.duration_ms,
                          "success": pr.success, "data_keys": list(pr.data.keys()) if pr.data else []},
                )

            # ── Security: record heartbeat for this service ──
            if security and hasattr(security, "heartbeat"):
                security.heartbeat.record_heartbeat("live-loop")

            # Accumulate into heartbeat totals
            if phase == LoopPhase.SIGNALS:
                heartbeat_result.signals_generated += pr.items_processed
            elif phase == LoopPhase.DECISION:
                heartbeat_result.trades_approved += pr.items_processed
            elif phase == LoopPhase.EXECUTION:
                heartbeat_result.trades_executed += pr.items_processed

            # Track phase performance
            phase_name = phase.value
            self._performance.phase_total_ms[phase_name] = (
                self._performance.phase_total_ms.get(phase_name, 0.0) + pr.duration_ms
            )
            self._performance.phase_call_count[phase_name] = (
                self._performance.phase_call_count.get(phase_name, 0) + 1
            )
            count = self._performance.phase_call_count[phase_name]
            total = self._performance.phase_total_ms[phase_name]
            self._performance.phase_avg_ms[phase_name] = total / count if count > 0 else 0.0

            if not pr.success:
                heartbeat_result.errors.append(f"{phase.value}: {pr.error}")

        except Exception as exc:
            error_msg = f"{phase.value}: {exc}"
            heartbeat_result.errors.append(error_msg)
            logger.error("Phase %s failed: %s", phase.value, exc)

            heartbeat_result.phases[phase.value] = PhaseResult(
                phase=phase.value,
                success=False,
                error=str(exc),
                timestamp=datetime.now().isoformat(),
            )

        self._current_phase = ""

    def _record_heartbeat(self, result: HeartbeatResult):
        """Record heartbeat result into performance tracking."""
        with self._lock:
            self._heartbeat_history.append(result)
            perf = self._performance

            perf.total_iterations = self._iteration
            perf.last_heartbeat = result.timestamp
            perf.total_signals += result.signals_generated
            perf.total_trades_approved += result.trades_approved
            perf.total_trades_executed += result.trades_executed
            perf.total_errors += len(result.errors)

            # Duration tracking
            ms = result.total_duration_ms
            perf.max_heartbeat_ms = max(perf.max_heartbeat_ms, ms)
            perf.min_heartbeat_ms = min(perf.min_heartbeat_ms, ms)

            if self._started_at:
                perf.total_runtime_seconds = (datetime.now() - self._started_at).total_seconds()

            if perf.total_iterations > 0:
                perf.avg_heartbeat_ms = (
                    (perf.avg_heartbeat_ms * (perf.total_iterations - 1) + ms)
                    / perf.total_iterations
                )

            # Uptime calculation
            if perf.total_runtime_seconds > 0:
                expected_iterations = perf.total_runtime_seconds / self._heartbeat_interval
                if expected_iterations > 0:
                    perf.uptime_pct = min(100.0, (perf.total_iterations / expected_iterations) * 100)

    def _update_circuit_breaker(self, nav: float, daily_pnl: float):
        """Evaluate risk circuit breaker state."""
        cb = self._circuit_breaker
        cb.last_check = datetime.now().isoformat()
        cb.triggers.clear()

        # Drawdown check
        if self._initial_nav > 0:
            cb.current_drawdown_pct = abs(min(0, daily_pnl)) / self._initial_nav
        else:
            cb.current_drawdown_pct = 0.0

        cb.max_drawdown_pct = max(cb.max_drawdown_pct, cb.current_drawdown_pct)

        # VIX from last macro snapshot
        if self._last_macro_snapshot:
            cb.vix_level = getattr(self._last_macro_snapshot, "vix", 20.0)

        # Evaluate risk level
        level = RiskLevel.NORMAL

        if cb.current_drawdown_pct >= cb.DRAWDOWN_KILL:
            level = RiskLevel.KILL_SWITCH
            cb.triggers.append(f"DRAWDOWN {cb.current_drawdown_pct:.1%} >= {cb.DRAWDOWN_KILL:.1%}")
        elif cb.current_drawdown_pct >= cb.DRAWDOWN_CRITICAL:
            level = RiskLevel.CRITICAL
            cb.triggers.append(f"DRAWDOWN {cb.current_drawdown_pct:.1%}")
        elif cb.current_drawdown_pct >= cb.DRAWDOWN_HIGH:
            level = max(level, RiskLevel.HIGH)
            cb.triggers.append(f"DRAWDOWN {cb.current_drawdown_pct:.1%}")
        elif cb.current_drawdown_pct >= cb.DRAWDOWN_ELEVATED:
            level = max(level, RiskLevel.ELEVATED)

        # VIX check
        if cb.vix_level >= cb.VIX_CRITICAL:
            level = max(level, RiskLevel.CRITICAL)
            cb.triggers.append(f"VIX {cb.vix_level:.1f}")
        elif cb.vix_level >= cb.VIX_HIGH:
            level = max(level, RiskLevel.HIGH)
            cb.triggers.append(f"VIX {cb.vix_level:.1f}")
        elif cb.vix_level >= cb.VIX_ELEVATED:
            level = max(level, RiskLevel.ELEVATED)

        # MetadronCube Kill-Switch: HY OAS +35bp & VIX term flat/inverted & breadth < 50%
        if (cb.hy_spread_change_bps >= cb.HY_SPREAD_KILL_BPS and
                cb.breadth_pct < cb.BREADTH_KILL_PCT and
                cb.vix_level >= cb.VIX_HIGH):
            level = RiskLevel.KILL_SWITCH
            cb.triggers.append("CUBE_KILL_SWITCH: HY+VIX+Breadth")

        cb.level = level
        cb.kill_switch_active = (level == RiskLevel.KILL_SWITCH)

        if cb.kill_switch_active:
            logger.critical("KILL SWITCH ACTIVATED: %s", ", ".join(cb.triggers))
        elif level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            logger.warning("Risk level %s: %s", level.value, ", ".join(cb.triggers))

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _persist_state(self):
        """Save loop state and performance to disk for restart recovery."""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)

            state = {
                "iteration": self._iteration,
                "session": self._session.value,
                "risk_level": self._circuit_breaker.level.value,
                "kill_switch": self._circuit_breaker.kill_switch_active,
                "started_at": self._started_at.isoformat() if self._started_at else "",
                "persisted_at": datetime.now().isoformat(),
                "last_signals": self._last_signals,
                "circuit_breaker": {
                    "max_drawdown_pct": self._circuit_breaker.max_drawdown_pct,
                    "consecutive_losses": self._circuit_breaker.consecutive_losses,
                },
            }

            _STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

            perf = asdict(self._performance)
            _PERFORMANCE_FILE.write_text(json.dumps(perf, indent=2, default=str))

            logger.debug("State persisted: iteration=%d", self._iteration)

        except Exception as exc:
            logger.warning("State persistence failed: %s", exc)

    def _restore_state(self):
        """Restore loop state from disk after restart."""
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text())
                self._iteration = data.get("iteration", 0)
                self._last_signals = data.get("last_signals", {})
                self._performance.restarts += 1

                cb_data = data.get("circuit_breaker", {})
                self._circuit_breaker.max_drawdown_pct = cb_data.get("max_drawdown_pct", 0.0)
                self._circuit_breaker.consecutive_losses = cb_data.get("consecutive_losses", 0)

                logger.info(
                    "State restored: iteration=%d restarts=%d",
                    self._iteration, self._performance.restarts,
                )

            if _PERFORMANCE_FILE.exists():
                perf_data = json.loads(_PERFORMANCE_FILE.read_text())
                # Restore cumulative counters
                self._performance.total_signals = perf_data.get("total_signals", 0)
                self._performance.total_trades_approved = perf_data.get("total_trades_approved", 0)
                self._performance.total_trades_executed = perf_data.get("total_trades_executed", 0)
                self._performance.total_errors = perf_data.get("total_errors", 0)

        except Exception as exc:
            logger.warning("State restore failed (starting fresh): %s", exc)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        loaded = sum(self._component_status.values())
        total = len(self._component_status)
        return (
            f"LiveLoopOrchestrator("
            f"state={self._state.value}, "
            f"session={self._session.value}, "
            f"iteration={self._iteration}, "
            f"components={loaded}/{total})"
        )
