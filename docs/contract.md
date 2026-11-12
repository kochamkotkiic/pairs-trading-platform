# Kontrakty między modułami

> **Wersja:** 0.1 (projekt do akceptacji zespołu)
> **Ostatnia aktualizacja:** 12.11.2026
> **Lokalizacja w repo:** `docs/CONTRACTS.md`, implementacja w `src/ptp/core/`

Ten dokument określa, **jak moduły rozmawiają ze sobą**: jakie funkcje i klasy istnieją, co przyjmują, co zwracają i jakie zasady muszą spełniać. Dopóki każdy trzyma się kontraktu, każdy może pracować niezależnie.

---

## Spis treści

0. [Zasady zmiany kontraktów](#0-zasady-zmiany-kontraktów)
1. [Konwencje ogólne](#1-konwencje-ogólne)
2. [Warstwa danych](#2-warstwa-danych)
3. [Para (`Pair`) i selekcja par](#3-para-pair-i-selekcja-par)
4. [Strategia (`Strategy`)](#4-strategia-strategy)
5. [Filtr ML (`SignalFilter`) i cechy](#5-filtr-ml-signalfilter-i-cechy)
6. [Wielkość pozycji i ryzyko](#6-wielkość-pozycji-i-ryzyko)
7. [Konfiguracja symulacji (`BacktestConfig`)](#7-konfiguracja-symulacji-backtestconfig)
8. [Backtester i wynik (`BacktestResult`)](#8-backtester-i-wynik-backtestresult)
9. [Schemat dziennika transakcji (trade log)](#9-schemat-dziennika-transakcji-trade-log)
10. [Krzywa kapitału i metryki](#10-krzywa-kapitału-i-metryki)
11. [Rejestr wtyczek](#11-rejestr-wtyczek)
12. [API](#12-api)
13. [Baza danych](#13-baza-danych)
14. [Zasady przeciw wyciekowi danych](#14-zasady-przeciw-wyciekowi-danych)
15. [Mapa: kto implementuje, kto używa](#15-mapa-kto-implementuje-kto-używa)
16. [Historia zmian](#16-historia-zmian)

---

## 0. Zasady zmiany kontraktów

- Kontrakty zamrażamy na koniec fazy F0 (**29.11.2026**).
- Po zamrożeniu zmiana wymaga Pull Requesta zmieniającego **ten plik i `src/ptp/core/` jednocześnie** oraz akceptacji **wszystkich trzech osób**.
- Dodanie nowego **opcjonalnego** pola lub parametru jest zmianą niełamiącą (wystarczy jedna akceptacja).
- Usunięcie lub zmiana znaczenia pola jest zmianą łamiącą: podbijamy wersję i dopisujemy wpis w [historii zmian](#16-historia-zmian).

---

## 1. Konwencje ogólne

| Element | Konwencja | Przykład |
|---|---|---|
| Język kodu | Angielski (nazwy, docstringi, komentarze) | `generate_signals` |
| Daty | `pd.Timestamp` bez strefy czasowej, tylko dni sesyjne NYSE | `2021-03-04` |
| Indeks czasowy | Rosnący, bez duplikatów, nazwa `date` | |
| Tickery | Wielkie litery, format Yahoo (kropka → myślnik) | `AAPL`, `BRK-B` |
| Identyfikator pary | `"{ticker_a}__{ticker_b}"` (podwójne podkreślenie) | `KO__PEP` |
| Kolejność w parze | `ticker_a` to zmienna objaśniana w regresji (`A = α + β·B`) | |
| Ceny | Skorygowane o splity i dywidendy (`adj_*`) do sygnałów; `open` skorygowane do wykonania | |
| Kwoty | `float` w USD | `10000.0` |
| Ułamki | Zawsze jako ułamek dziesiętny, **nigdy procent** | `0.001` = 0,1% |
| Stopy zwrotu | Proste (nie logarytmiczne), chyba że nazwa mówi inaczej (`log_ret`) | |
| Kierunek pozycji | `+1` long spread (kup A, sprzedaj β·B), `-1` short spread, `0` brak pozycji | |
| Losowość | Każda funkcja losowa przyjmuje `seed: int` | |
| Typy | Wszystkie publiczne funkcje mają adnotacje typów | |

---

## 2. Warstwa danych

**Moduł:** `ptp.data` · **Właściciel:** O3 (+ O1)

### 2.1 Format przechowywania (Parquet)

Plik `data/prices/v{YYYYMMDD}/prices.parquet`, **format długi** (jeden wiersz = jeden ticker w jednym dniu):

| Kolumna | Typ | Opis |
|---|---|---|
| `date` | date | Dzień sesyjny |
| `ticker` | str | Ticker |
| `open`, `high`, `low`, `close` | float | Ceny surowe |
| `adj_open`, `adj_close` | float | Ceny skorygowane |
| `volume` | int | Wolumen |

Dodatkowo:
- `data/context/v{YYYYMMDD}/context.parquet`: kolumny `date`, `vix`, `spx`
- `data/meta/sectors.csv`: kolumny `ticker`, `sector`, `industry` (statyczne, opis ograniczenia w sprawozdaniu)

Wersja danych (`v20261120`) jest zapisywana w każdej symulacji, żeby wyniki były powtarzalne.

### 2.2 Interfejs

```python
# src/ptp/data/loader.py

def get_prices(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    field: Literal["adj_close", "adj_open", "close", "open", "volume"] = "adj_close",
    data_version: str | None = None,   # None = newest
) -> pd.DataFrame:
    """Wide format: index = date, columns = tickers. Inclusive [start, end].
    Missing values stay NaN (no forward fill here)."""

def get_context(start: pd.Timestamp, end: pd.Timestamp,
                data_version: str | None = None) -> pd.DataFrame:
    """index = date, columns = ['vix', 'spx']."""

def get_universe(name: str) -> list[str]:
    """Named universes: 'sp500', 'sector:Energy', 'custom:<file>'."""

def get_sectors() -> pd.Series:
    """index = ticker, value = sector."""

def trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex: ...
```

**Zasady:**
- Loader **nie** uzupełnia braków i **nie** filtruje tickerów: robi to wywołujący, świadomie.
- Loader zwraca tylko dane z zakresu `[start, end]`. Odpowiedzialność za to, by `end` nie wychodził w przyszłość względem decyzji, ma wywołujący (patrz [§14](#14-zasady-przeciw-wyciekowi-danych)).

---

## 3. Para (`Pair`) i selekcja par

**Moduł:** `ptp.core.types`, `ptp.pairs` · **Właściciel:** O1

### 3.1 Typ `Pair`

```python
# src/ptp/core/types.py
class Pair(BaseModel):
    pair_id: str                 # "KO__PEP"
    ticker_a: str
    ticker_b: str
    hedge_ratio: float           # beta from formation window
    intercept: float             # alpha
    half_life: float             # in trading days
    pvalue: float                # raw cointegration test p-value
    pvalue_adj: float | None     # after Benjamini-Hochberg
    hurst: float | None
    spread_mean: float           # mean of spread in formation window
    spread_std: float            # std of spread in formation window
    formation_start: date
    formation_end: date
    score: float | None = None   # ranking score (higher = better)
```

### 3.2 Interfejs selekcji

```python
# src/ptp/core/interfaces.py
class PairSelector(ABC):
    name: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def select(
        self,
        prices: pd.DataFrame,          # wide adj_close, ONLY formation window
        formation_start: pd.Timestamp,
        formation_end: pd.Timestamp,
        params: BaseModel,
    ) -> list[Pair]:
        """Returns selected pairs sorted by score (best first)."""
```

**Zasady:**
- `prices` zawiera **wyłącznie** dane z okna formowania. Wywołujący (backtester) przycina dane.
- Liczba zwróconych par ≤ `params.max_pairs`.

---

## 4. Strategia (`Strategy`)

**Moduł:** `ptp.strategies` · **Właściciel:** O1

### 4.1 Sygnał jako **docelowa pozycja**

Strategia nie mówi „kup” ani „sprzedaj”, tylko **jaką pozycję chce mieć na koniec dnia `t`** dla każdej pary. Backtester sam wylicza z tego zlecenia (zmiana docelowej pozycji = transakcja). Dzięki temu nie ma niejasności, co oznacza powtórzony sygnał.

### 4.2 Interfejs

```python
class Strategy(ABC):
    name: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,          # wide adj_close, from formation_start to trading_end
        pairs: list[Pair],
        trading_start: pd.Timestamp,
        trading_end: pd.Timestamp,
        params: BaseModel,
    ) -> pd.DataFrame:
        """One row per (date, pair_id) for every trading day in [trading_start, trading_end]."""
```

### 4.3 Format zwracany: `signals`

| Kolumna | Typ | Opis |
|---|---|---|
| `date` | date | Dzień, w którym **podjęto decyzję** (po zamknięciu sesji) |
| `pair_id` | str | |
| `target` | int | Docelowa pozycja: `+1`, `-1`, `0` |
| `zscore` | float | Z-score spreadu w dniu `date` |
| `spread` | float | Wartość spreadu |
| `hedge_ratio` | float | β używane w dniu `date` |
| `event` | str | `entry`, `exit_mean_revert`, `exit_stop_loss`, `exit_time_limit`, `hold`, `flat` |

**Zasady:**
- Wartości w wierszu z dnia `t` mogą zależeć **tylko od danych do `t` włącznie**.
- Backtester wykonuje zmianę pozycji po cenie otwarcia dnia `t+1`.
- **Zdarzenie wejścia** (`entry`) to wiersz, w którym `target` zmienia się z `0` na `±1`. Jego klucz to `(pair_id, date)`. Do tych zdarzeń odnosi się filtr ML.

### 4.4 Parametry (przykład)

```python
class ZScoreParams(BaseModel):
    lookback: int = 60            # rolling window for mean/std
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5
    max_holding_days: int | None = None   # None = 3 * half_life
```

`params_model` służy też frontendowi: na jego podstawie UI automatycznie generuje formularz.

---

## 5. Filtr ML (`SignalFilter`) i cechy

**Moduł:** `ptp.ml` · **Właściciel:** O2

### 5.1 Zdarzenia (events)

Tabela zdarzeń wejścia `events` to podzbiór `signals` z `event == "entry"`, z indeksem `event_id = f"{pair_id}|{date:%Y-%m-%d}"`.

### 5.2 Budowa cech

```python
class FeatureBuilder(ABC):
    name: ClassVar[str]

    @abstractmethod
    def build(
        self,
        events: pd.DataFrame,          # index = event_id; cols: date, pair_id, zscore, hedge_ratio, ...
        prices: pd.DataFrame,          # wide adj_close
        volume: pd.DataFrame,          # wide volume
        context: pd.DataFrame,         # vix, spx
        pairs: dict[str, Pair],
    ) -> pd.DataFrame:
        """index = event_id, columns = numeric features, no NaN."""
```

**Zasada:** cecha dla zdarzenia z dnia `t` używa wyłącznie danych do `t` włącznie.

### 5.3 Etykiety

```python
def make_labels(trade_log: pd.DataFrame, method: str = "triple_barrier", **kwargs) -> pd.Series:
    """index = event_id, values in {0, 1}. Also returns label end time (t1) for purging."""
```

Etykieta 1 = transakcja zakończona osiągnięciem bariery zysku. Dokładną definicję ustala O2 w sprawozdaniu.

### 5.4 Interfejs filtra

```python
class SignalFilter(ABC):
    name: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, t1: pd.Series) -> None:
        """t1 = time when each label is known (exit date); needed for purged CV."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """index = event_id, values = P(success) in [0, 1]."""

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "SignalFilter": ...
```

### 5.5 Jak backtester używa filtra

1. Dla każdego okna handlowego filtr jest trenowany **tylko na zdarzeniach, których etykieta była znana przed `trading_start`** (`t1 < trading_start`).
2. Dla każdego zdarzenia `entry` w oknie backtester wywołuje `predict_proba`.
3. Jeśli `proba < config.ml.threshold`, wejście jest **pomijane** i para zostaje płaska aż do kolejnego zdarzenia `entry`.
4. `proba` trafia do trade logu (kolumna `ml_proba`).

Wariant **filtr losowy** (eksperyment E3) implementuje ten sam interfejs i zwraca losowe wartości z ustalonym `seed`.

---

## 6. Wielkość pozycji i ryzyko

**Moduł:** `ptp.risk` · **Właściciel:** O1 (wywołuje O3)

```python
class PositionSizer(ABC):
    name: ClassVar[str]

    @abstractmethod
    def size(
        self,
        pair: Pair,
        direction: int,                 # +1 / -1
        price_a: float,                 # execution price (open t+1)
        price_b: float,
        portfolio: PortfolioState,
        params: RiskParams,
    ) -> tuple[float, float]:
        """Returns (qty_a, qty_b) in shares, signed. (0, 0) = reject."""
```

```python
class PortfolioState(BaseModel):
    date: date
    cash: float
    equity: float
    open_positions: dict[str, tuple[float, float]]   # pair_id -> (qty_a, qty_b)
    gross_exposure: float

class RiskParams(BaseModel):
    capital_per_pair: float = 0.1     # fraction of equity per pair
    max_open_pairs: int = 10
    max_gross_leverage: float = 2.0
    dollar_neutral: bool = True
    allow_fractional_shares: bool = True
```

---

## 7. Konfiguracja symulacji (`BacktestConfig`)

**Moduł:** `ptp.core.config` · **Właściciel:** O3

Jeden obiekt opisuje całą symulację. Ten sam schemat jest używany przez API, frontend, pliki eksperymentów (`experiments/*.yaml`) i bazę (kolumna JSON).

```python
class WalkForwardConfig(BaseModel):
    start: date
    end: date
    formation_months: int = 12
    trading_months: int = 6
    step_months: int = 6

class CostConfig(BaseModel):
    commission: float = 0.0005       # per side, fraction of notional
    slippage: float = 0.0005         # per side, fraction of price
    borrow_rate_annual: float = 0.01 # on short leg

class MLConfig(BaseModel):
    filter: str | None = None        # registry name, None = baseline
    threshold: float = 0.5
    params: dict = {}

class BacktestConfig(BaseModel):
    name: str
    universe: str = "sp500"
    data_version: str | None = None
    initial_capital: float = 1_000_000.0
    walk_forward: WalkForwardConfig
    pair_selector: str = "coint_bh"
    pair_selector_params: dict = {}
    strategy: str = "zscore"
    strategy_params: dict = {}
    sizer: str = "equal_dollar"
    risk: RiskParams = RiskParams()
    costs: CostConfig = CostConfig()
    ml: MLConfig = MLConfig()
    engine: Literal["vectorized", "event"] = "vectorized"
    seed: int = 42
```

Przykład w YAML (`experiments/e1_baseline.yaml`):

```yaml
name: e1_baseline
universe: sp500
walk_forward:
  start: 2012-01-01
  end: 2025-12-31
  formation_months: 12
  trading_months: 6
strategy: zscore
strategy_params: {lookback: 60, z_entry: 2.0, z_exit: 0.5, z_stop: 3.5}
costs: {commission: 0.0005, slippage: 0.0005}
ml: {filter: null}
engine: event
```

---

## 8. Backtester i wynik (`BacktestResult`)

**Moduł:** `ptp.backtest` · **Właściciel:** O3

```python
class Backtester(ABC):
    name: ClassVar[str]               # "vectorized" | "event"

    @abstractmethod
    def run(self, config: BacktestConfig,
            progress: Callable[[float], None] | None = None) -> BacktestResult: ...
```

```python
@dataclass
class BacktestResult:
    config: BacktestConfig
    data_version: str
    trades: pd.DataFrame          # schema §9
    equity: pd.DataFrame          # schema §10.1
    signals: pd.DataFrame         # schema §4.3 (all windows concatenated)
    pairs: list[Pair]             # all selected pairs (with formation dates)
    metrics: dict[str, float]     # schema §10.2
    duration_s: float
```

**Przebieg `run`** (wspólny dla obu silników):
1. Podział `[start, end]` na okna walk-forward.
2. Dla każdego okna: `PairSelector.select` na oknie formowania → `Strategy.generate_signals` → (opcjonalnie) trening i zapytanie `SignalFilter` → wykonanie zleceń (open `t+1`, koszty) → aktualizacja portfela.
3. Na koniec okna handlowego pozycje są zamykane (`exit_reason = "end_of_window"`).
4. `metrics = compute_metrics(equity, trades)`.

**Wymóg:** oba silniki dają ten sam trade log dla tej samej konfiguracji (z tolerancją numeryczną), co sprawdza test integracyjny.

---

## 9. Schemat dziennika transakcji (trade log)

**Najważniejszy kontrakt w projekcie:** łączy strategię (O1), ML (O2), backtester, bazę i frontend (O3).

| Kolumna | Typ | Opis |
|---|---|---|
| `trade_id` | int | Unikalny w ramach symulacji |
| `event_id` | str | `"{pair_id}\|{signal_date}"`, łączy z cechami ML |
| `pair_id` | str | |
| `ticker_a`, `ticker_b` | str | |
| `direction` | int | `+1` / `-1` |
| `signal_date` | date | Dzień decyzji o wejściu (`t`) |
| `entry_date` | date | Dzień wykonania wejścia (`t+1`) |
| `exit_signal_date` | date | Dzień decyzji o wyjściu |
| `exit_date` | date | Dzień wykonania wyjścia |
| `holding_days` | int | Dni sesyjne między `entry_date` i `exit_date` |
| `hedge_ratio` | float | β przy wejściu |
| `qty_a`, `qty_b` | float | Liczba akcji (ze znakiem) |
| `entry_price_a`, `entry_price_b` | float | Ceny wykonania wejścia (z poślizgiem) |
| `exit_price_a`, `exit_price_b` | float | Ceny wykonania wyjścia (z poślizgiem) |
| `entry_z`, `exit_z` | float | Z-score przy decyzji o wejściu i wyjściu |
| `gross_pnl` | float | Zysk przed kosztami (USD) |
| `costs` | float | Prowizje + poślizg + koszt pożyczki (USD, ≥ 0) |
| `pnl` | float | `gross_pnl - costs` |
| `ret` | float | `pnl / notional przy wejściu` |
| `exit_reason` | str | `mean_revert`, `stop_loss`, `time_limit`, `end_of_window` |
| `window_id` | int | Numer okna walk-forward |
| `ml_proba` | float \| null | Wynik filtra (null dla baseline) |

**Wersja dla Osoby 2 (wczesna):** zanim powstanie pełny backtester, O1 może dostarczyć trade log z backtestu wektorowego z tymi samymi kolumnami. Pola, których nie da się jeszcze policzyć, mogą być `NaN`.

---

## 10. Krzywa kapitału i metryki

### 10.1 `equity`

| Kolumna | Typ | Opis |
|---|---|---|
| `date` | date | Każdy dzień sesyjny symulacji |
| `equity` | float | Wartość portfela na zamknięcie |
| `cash` | float | Gotówka |
| `gross_exposure` | float | Suma wartości bezwzględnych pozycji |
| `n_open_pairs` | int | Liczba otwartych par |
| `daily_ret` | float | Dzienna stopa zwrotu portfela |

### 10.2 Metryki

```python
# src/ptp/metrics/core.py  (O1)
def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame,
                    periods_per_year: int = 252) -> dict[str, float]: ...
```

Stałe klucze (frontend i baza na nich polegają):

| Klucz | Opis |
|---|---|
| `total_return` | Całkowita stopa zwrotu |
| `cagr` | Średnioroczna stopa zwrotu |
| `volatility` | Roczna zmienność |
| `sharpe` | Sharpe (stopa wolna od ryzyka = 0) |
| `sortino` | Sortino |
| `max_drawdown` | Maks. obsunięcie (liczba dodatnia, ułamek) |
| `calmar` | `cagr / max_drawdown` |
| `n_trades` | Liczba transakcji |
| `win_rate` | Odsetek transakcji z `pnl > 0` |
| `avg_trade_ret` | Średni `ret` |
| `avg_holding_days` | Średni czas trwania |
| `turnover` | Roczny obrót / średni kapitał |
| `total_costs` | Suma kosztów (USD) |

Nowe metryki można dodawać, istniejących kluczy nie wolno zmieniać.

---

## 11. Rejestr wtyczek

**Moduł:** `ptp.core.registry` · **Właściciel:** O3

```python
# usage in a plugin file
@register("strategy", "zscore")
class ZScoreStrategy(Strategy):
    name = "zscore"
    params_model = ZScoreParams
    ...

# usage in backtester / API
strategy_cls = get("strategy", config.strategy)
params = strategy_cls.params_model(**config.strategy_params)
```

Kategorie rejestru: `pair_selector`, `strategy`, `signal_filter`, `feature_builder`, `sizer`, `backtester`.

Dodanie nowej strategii lub modelu = nowa klasa z dekoratorem `@register` + import w `__init__.py` pakietu. Nic więcej nie trzeba zmieniać.

---

## 12. API

**Moduł:** `ptp.api` · **Właściciel:** O3 · Dokumentacja na żywo: `http://localhost:8000/docs`

| Metoda | Ścieżka | Wejście | Wyjście |
|---|---|---|---|
| GET | `/health` | – | `{"status": "ok"}` |
| GET | `/universes` | – | `list[{name, n_tickers}]` |
| GET | `/plugins/{category}` | – | `list[{name, params_schema}]` (JSON Schema z `params_model`) |
| POST | `/pairs/discover` | `{universe, formation_start, formation_end, selector, params}` | `list[Pair]` |
| POST | `/backtests` | `BacktestConfig` | `{run_id, status}` (HTTP 202) |
| GET | `/backtests` | `?limit=&offset=` | `list[RunSummary]` |
| GET | `/backtests/{run_id}` | – | `{run_id, status, progress, config, metrics, error}` |
| GET | `/backtests/{run_id}/trades` | – | trade log (§9) jako lista JSON |
| GET | `/backtests/{run_id}/equity` | – | `equity` (§10.1) |
| GET | `/backtests/{run_id}/signals` | `?pair_id=` | `signals` dla pary (do wykresu spreadu) |
| GET | `/backtests/compare` | `?ids=1,2,3` | metryki + krzywe kapitału wielu przebiegów |
| DELETE | `/backtests/{run_id}` | – | 204 |

**Statusy przebiegu:** `pending` → `running` → `done` \| `failed`.

**Błędy:** zawsze `{"detail": "<opis>"}`, kod 422 dla złej konfiguracji, 404 dla brakującego `run_id`.

---

## 13. Baza danych

**Moduł:** `ptp.db` · **Właściciel:** O3 · PostgreSQL, migracje przez Alembic

| Tabela | Kluczowe kolumny |
|---|---|
| `backtest_runs` | `id` PK, `name`, `created_at`, `status`, `progress`, `config` JSONB, `data_version`, `duration_s`, `error` |
| `trades` | `id` PK, `run_id` FK, kolumny z §9 |
| `equity_curve` | `run_id` FK, `date`, kolumny z §10.1, PK (`run_id`, `date`) |
| `metrics` | `run_id` FK, `name`, `value`, PK (`run_id`, `name`) |
| `pairs` | `id` PK, `run_id` FK, `window_id`, kolumny z §3.1 |
| `signals` | `run_id` FK, `date`, `pair_id`, kolumny z §4.3 *(opcjonalnie, może być duża)* |

Moduły O1 i O2 **nie** używają bazy bezpośrednio: dostają i zwracają DataFrame'y. Zapisem zajmuje się warstwa `ptp.db.repository` (O3).

---

## 14. Zasady przeciw wyciekowi danych

To są zasady obowiązujące wszystkich, sprawdzane testami w `tests/leakage/`.

| # | Zasada | Kto pilnuje | Test |
|---|---|---|---|
| L1 | Selekcja par widzi tylko okno formowania | O3 (przycina dane) | Zmiana cen po `formation_end` nie zmienia wybranych par |
| L2 | Sygnał z dnia `t` zależy tylko od danych ≤ `t` | O1 | Zmiana cen po `t` nie zmienia sygnałów ≤ `t` |
| L3 | Wykonanie po otwarciu `t+1` | O3 | `entry_date > signal_date` dla każdej transakcji |
| L4 | Cechy ML z dnia `t` tylko z danych ≤ `t` | O2 | Zmiana danych po `t` nie zmienia cech zdarzeń ≤ `t` |
| L5 | Model trenowany tylko na etykietach znanych przed oknem handlowym | O2 + O3 | Wszystkie `t1` w zbiorze treningowym < `trading_start` |
| L6 | Normalizacja/skalowanie cech dopasowywane tylko na zbiorze treningowym | O2 | Przegląd kodu + test na sztucznych danych |
| L7 | Brak `bfill()` i `shift(-n)` w kodzie produkcyjnym | wszyscy | Prosty test grep w CI |

**Wzór testu L2** (przykład dla O1):

```python
def test_signals_do_not_look_ahead(sample_prices, pairs, strategy, params):
    t = sample_prices.index[200]
    base = strategy.generate_signals(sample_prices, pairs, ..., params)
    perturbed = sample_prices.copy()
    perturbed.loc[perturbed.index > t] *= 1.5       # change the future
    after = strategy.generate_signals(perturbed, pairs, ..., params)
    pd.testing.assert_frame_equal(base[base.date <= t], after[after.date <= t])
```

---

## 15. Mapa: kto implementuje, kto używa

| Kontrakt | Implementuje | Używa |
|---|---|---|
| `get_prices`, `get_context`, `get_universe` | O3 (+O1) | O1, O2, backtester |
| `Pair`, `PairSelector` | O1 | backtester, API, frontend |
| `Strategy`, format `signals` | O1 | backtester, O2, frontend |
| `FeatureBuilder`, `make_labels`, `SignalFilter` | O2 | backtester |
| `PositionSizer`, `RiskParams` | O1 | backtester |
| `compute_metrics` | O1 | backtester, API, frontend |
| `BacktestConfig` | O3 | wszyscy |
| `Backtester`, `BacktestResult` | O3 | API, skrypty eksperymentów (O1, O2) |
| Trade log (§9) | O3 (backtester), O1 (wersja wczesna) | O2, baza, frontend |
| Rejestr wtyczek | O3 | wszyscy |
| API, baza | O3 | frontend |

---

## 16. Historia zmian

| Wersja | Data | Autor | Zmiana |
|---|---|---|---|
| 0.1 | 12.11.2026 | O3 | Pierwsza wersja do przeglądu przez zespół |