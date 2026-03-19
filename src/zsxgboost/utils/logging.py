import sys
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box
from loguru import logger as _loguru

_loguru.remove()

# Loguru format: timestamp | level | message
_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "{message}"
)


class Logger:
    def __init__(self):
        self._loguru = _loguru
        # Dedicated Rich console on stderr for structured output (tables, rules, panels).
        self._console = Console(stderr=True, highlight=False)
        self._sink_id: Optional[int] = None
        self._verbose = False

    def set_verbosity(self, verbose: bool):
        self._verbose = verbose
        if verbose and self._sink_id is None:
            self._sink_id = self._loguru.add(
                sys.stderr,
                format=_FORMAT,
                colorize=True,
                level="DEBUG",
            )
        elif not verbose and self._sink_id is not None:
            try:
                self._loguru.remove(self._sink_id)
            except Exception:
                pass
            self._sink_id = None

    # ------------------------------------------------------------------
    # Plain-text log levels (always go through loguru → sink)
    # ------------------------------------------------------------------

    def debug(self, msg: str):
        self._loguru.debug(msg)

    def info(self, msg: str):
        self._loguru.info(msg)

    def warning(self, msg: str):
        self._loguru.warning(msg)

    def error(self, msg: str):
        self._loguru.error(msg)

    def success(self, msg: str):
        self._loguru.success(msg)

    # ------------------------------------------------------------------
    # Structured Rich output (only emitted when verbose=True)
    # ------------------------------------------------------------------

    def rule(self, title: str = "", style: str = "dim blue"):
        """Print a horizontal rule, optionally with a centred title."""
        if not self._verbose:
            return
        if title:
            self._console.rule(f"[bold cyan]{title}[/]", style=style)
        else:
            self._console.rule(style=style)

    def profile_summary(self, profile) -> None:
        """Print a compact one-line summary of the dataset profile."""
        if not self._verbose:
            return

        task_label = {
            "binary_classification": "Binary classification",
            "regression":            "Regression",
        }.get(profile.task, profile.task)

        parts = [
            f"n={profile.n_samples:,}",
            f"p={profile.n_features:,}",
            f"n/p={profile.n_p_ratio:.1f}",
        ]
        if profile.task == "binary_classification":
            parts.append(f"imbalance={profile.imbalance_ratio:.1f}:1")
        else:
            parts.append(f"skewness={profile.y_skewness:.2f}")
        if profile.is_sparse_counts:
            parts.append("sparse_counts=True")
        if profile.binary_feature_fraction > 0.5:
            parts.append(f"binary_frac={profile.binary_feature_fraction:.2f}")

        sep = "  [dim]|[/]  "
        body = sep.join(f"[cyan]{p}[/]" for p in parts)
        self._console.print(f"[bold]{task_label}[/bold]  {body}")

    def portfolio_table(
        self,
        fast_scores: dict,
        params_map: dict,
        winner: str,
        threshold: float,
        default_score: float,
        n_tr: int,
        n_splits: int,
        skipped: list,
    ) -> None:
        """
        Print a Rich table summarising Stage-1 portfolio scores.

        Columns: Preset | LR | max_depth | Stage-1 score | Gain vs default | Status
        """
        if not self._verbose:
            return

        table = Table(
            title="[bold]Portfolio — Stage 1 comparison[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
            title_style="",
        )
        table.add_column("Preset",          style="cyan", no_wrap=True, min_width=10)
        table.add_column("LR",              justify="right", width=9, no_wrap=True)
        table.add_column("Depth",           justify="right", width=8, no_wrap=True)
        table.add_column(f"Score ({n_splits} split{'s' if n_splits > 1 else ''})", justify="right", width=14)
        table.add_column("Gain vs default", justify="right", width=16)
        table.add_column("Decision",        no_wrap=True)

        preset_order = ["internal", "default", "flaml", "autogluon", "rf_like"]

        for name in preset_order:
            score  = fast_scores.get(name, float("nan"))
            params = params_map.get(name, {})
            lr     = params.get("learning_rate", float("nan"))
            # Prefer max_depth; FLAML presets use max_leaves instead.
            if params.get("grow_policy") == "lossguide":
                depth_val = f"{params.get('max_leaves', '?')}L"
            else:
                depth_val = str(params.get("max_depth", "?"))

            is_nan = score != score  # NaN check

            # Score cell
            score_str = f"{score:+.4f}" if not is_nan else "  —"

            # Gain cell
            if name == "default" or is_nan:
                gain_str   = "  —"
                gain_style = "dim"
                gain       = 0.0
            else:
                gain       = score - default_score
                gain_str   = f"{gain:+.4f}"
                gain_style = "green" if gain > 0 else "red"

            # Decision cell
            if is_nan:
                decision = "[dim]skipped (cost)[/dim]"
            elif name == winner:
                if name == "default":
                    decision = "[bold green]✓ default wins[/]"
                else:
                    decision = "[bold green]✓ selected[/]"
            elif name == "default":
                decision = "[dim]baseline[/dim]"
            else:
                if gain > 0 and gain < threshold:
                    decision = f"[yellow]↑ gain < thresh[/yellow]"
                elif gain <= 0:
                    decision = "[dim]worse than default[/dim]"
                else:
                    decision = "[dim]—[/dim]"

            # Highlight the winning row
            row_style = "bold" if name == winner else ""

            table.add_row(
                name,
                f"{lr:.4f}" if lr == lr else "—",
                depth_val,
                score_str,
                f"[{gain_style}]{gain_str}[/]",
                decision,
                style=row_style,
            )

        self._console.print(table)
        self._console.print(
            f"  [dim]threshold = {threshold:.4f}  "
            f"|  n_train = {n_tr:,}  "
            f"|  {n_splits} split(s) averaged[/dim]"
        )
        self._console.line()


logger = Logger()
