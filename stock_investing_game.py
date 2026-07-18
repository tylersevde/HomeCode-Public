import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import pygame
except Exception as e:
    print("This game requires pygame. Install it with: python -m pip install pygame")
    raise

# yfinance is optional at runtime. If it is not installed or data fetch fails,
# the game falls back to deterministic offline market data.
try:
    import yfinance as yf
except Exception:
    yf = None


WIDTH, HEIGHT = 800, 480
FPS = 60
STARTING_CASH = 10000.0
MAX_VISIBLE_LOG = 5
CHART_LOOKBACK = 90
AUTO_ADVANCE_MS = 700
SYMBOLS = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet"),
    ("META", "Meta"),
    ("TSLA", "Tesla"),
    ("SPY", "S&P 500 ETF"),
]

BG = (13, 18, 27)
PANEL = (21, 29, 42)
PANEL_2 = (31, 41, 58)
TEXT = (230, 235, 245)
MUTED = (144, 156, 180)
ACCENT = (80, 170, 255)
POSITIVE = (77, 200, 120)
NEGATIVE = (240, 96, 96)
GRID = (45, 58, 82)
WARNING = (255, 191, 94)
WHITE = (250, 250, 252)
BLACK = (0, 0, 0)


@dataclass
class PriceSeries:
    symbol: str
    name: str
    dates: List[str]
    closes: List[float]
    is_fallback: bool = False


@dataclass
class GameState:
    market: Dict[str, PriceSeries]
    ordered_symbols: List[str]
    cash: float = STARTING_CASH
    holdings: Dict[str, int] = field(default_factory=dict)
    selected_idx: int = 0
    day_idx: int = 0
    auto_play: bool = False
    auto_advance_ms: int = AUTO_ADVANCE_MS
    last_advance_time: int = 0
    log: List[str] = field(default_factory=list)
    finished: bool = False
    start_equity: float = STARTING_CASH

    def __post_init__(self) -> None:
        for symbol in self.ordered_symbols:
            self.holdings.setdefault(symbol, 0)

    @property
    def selected_symbol(self) -> str:
        return self.ordered_symbols[self.selected_idx]

    @property
    def max_day_index(self) -> int:
        return min(len(self.market[s].closes) for s in self.ordered_symbols) - 1

    def price(self, symbol: str) -> float:
        series = self.market[symbol].closes
        idx = max(0, min(self.day_idx, len(series) - 1))
        return float(series[idx])

    def previous_price(self, symbol: str) -> float:
        series = self.market[symbol].closes
        idx = max(0, min(self.day_idx - 1, len(series) - 1))
        return float(series[idx])

    def day_label(self) -> str:
        base = self.market[self.ordered_symbols[0]].dates
        idx = max(0, min(self.day_idx, len(base) - 1))
        return base[idx]

    def equity(self) -> float:
        return self.cash + sum(self.holdings[s] * self.price(s) for s in self.ordered_symbols)

    def benchmark_return(self) -> float:
        spy = self.market.get("SPY")
        if not spy:
            return 0.0
        start = spy.closes[0]
        current = spy.closes[min(self.day_idx, len(spy.closes) - 1)]
        if start <= 0:
            return 0.0
        return (current / start) - 1.0

    def portfolio_return(self) -> float:
        return (self.equity() / self.start_equity) - 1.0

    def add_log(self, message: str) -> None:
        stamp = self.day_label()
        self.log.append(f"[{stamp}] {message}")
        self.log = self.log[-100:]

    def buy(self, qty: int) -> None:
        if self.finished:
            return
        symbol = self.selected_symbol
        price = self.price(symbol)
        cost = price * qty
        if qty <= 0:
            return
        if cost > self.cash + 1e-9:
            self.add_log(f"Not enough cash to buy {qty} {symbol} shares.")
            return
        self.cash -= cost
        self.holdings[symbol] += qty
        self.add_log(f"Bought {qty} {symbol} @ ${price:,.2f}.")

    def buy_max(self) -> None:
        symbol = self.selected_symbol
        price = self.price(symbol)
        if price <= 0:
            return
        qty = int(self.cash // price)
        if qty <= 0:
            self.add_log(f"Cannot afford any {symbol} shares.")
            return
        self.buy(qty)

    def sell(self, qty: int) -> None:
        if self.finished:
            return
        symbol = self.selected_symbol
        owned = self.holdings[symbol]
        if qty <= 0:
            return
        if qty > owned:
            self.add_log(f"You only own {owned} {symbol} shares.")
            return
        price = self.price(symbol)
        self.cash += price * qty
        self.holdings[symbol] -= qty
        self.add_log(f"Sold {qty} {symbol} @ ${price:,.2f}.")

    def sell_all(self) -> None:
        symbol = self.selected_symbol
        qty = self.holdings[symbol]
        if qty <= 0:
            self.add_log(f"No {symbol} shares to sell.")
            return
        self.sell(qty)

    def advance_day(self, steps: int = 1) -> None:
        if self.finished:
            return
        self.day_idx = min(self.day_idx + steps, self.max_day_index)
        if self.day_idx >= self.max_day_index:
            self.finished = True
            self.auto_play = False
            perf = self.portfolio_return() * 100.0
            bench = self.benchmark_return() * 100.0
            self.add_log(
                f"Simulation finished. Return: {perf:+.2f}% | SPY benchmark: {bench:+.2f}%"
            )


def fallback_series(symbol: str, name: str, days: int = 220) -> PriceSeries:
    seed = sum((i + 1) * ord(c) for i, c in enumerate(symbol))
    rng = random.Random(seed)
    base_price_map = {
        "AAPL": 185.0,
        "MSFT": 420.0,
        "NVDA": 120.0,
        "AMZN": 180.0,
        "GOOGL": 160.0,
        "META": 510.0,
        "TSLA": 220.0,
        "SPY": 520.0,
    }
    start = base_price_map.get(symbol, 100.0)
    drift = rng.uniform(-0.0002, 0.0012)
    volatility = rng.uniform(0.008, 0.03)
    dates: List[str] = []
    closes: List[float] = []
    price = start
    base_date = datetime(2025, 1, 2)

    for day in range(days):
        # Skip weekends in the synthetic calendar for a stock-market feel.
        while base_date.weekday() >= 5:
            base_date = base_date.replace(day=base_date.day)  # no-op to satisfy linters
            from datetime import timedelta
            base_date += timedelta(days=1)
        dates.append(base_date.strftime("%Y-%m-%d"))
        wave = math.sin(day / 11.0) * 0.003 + math.cos(day / 37.0) * 0.002
        shock = rng.gauss(0.0, volatility)
        price *= max(0.55, 1.0 + drift + wave + shock)
        closes.append(round(price, 2))
        from datetime import timedelta
        base_date += timedelta(days=1)

    return PriceSeries(symbol=symbol, name=name, dates=dates, closes=closes, is_fallback=True)


def fetch_yfinance_series(symbol: str, name: str, period: str = "1y") -> PriceSeries:
    if yf is None:
        raise RuntimeError("yfinance not available")

    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        raise RuntimeError(f"No historical close data for {symbol}")

    hist = hist.dropna(subset=["Close"])
    closes = [float(x) for x in hist["Close"].tolist()]
    if len(closes) < 80:
        raise RuntimeError(f"Not enough data for {symbol}")

    dates = []
    for idx in hist.index.tolist():
        try:
            dates.append(idx.strftime("%Y-%m-%d"))
        except Exception:
            dates.append(str(idx)[:10])

    return PriceSeries(symbol=symbol, name=name, dates=dates, closes=closes, is_fallback=False)


def load_market() -> Tuple[Dict[str, PriceSeries], bool]:
    market: Dict[str, PriceSeries] = {}
    used_fallback = False
    for symbol, name in SYMBOLS:
        try:
            series = fetch_yfinance_series(symbol, name)
        except Exception:
            series = fallback_series(symbol, name)
            used_fallback = True
        market[symbol] = series

    # Trim all series to the same trailing window so day indexes line up.
    min_len = min(len(series.closes) for series in market.values())
    target_len = min(240, min_len)
    for series in market.values():
        series.closes = series.closes[-target_len:]
        series.dates = series.dates[-target_len:]

    return market, used_fallback


def make_fonts() -> Dict[str, pygame.font.Font]:
    small_screen = WIDTH <= 800 or HEIGHT <= 480
    if small_screen:
        return {
            "tiny": pygame.font.SysFont("consolas", 12),
            "small": pygame.font.SysFont("consolas", 14),
            "body": pygame.font.SysFont("consolas", 18),
            "title": pygame.font.SysFont("consolas", 24, bold=True),
            "hero": pygame.font.SysFont("consolas", 34, bold=True),
        }
    return {
        "tiny": pygame.font.SysFont("consolas", 16),
        "small": pygame.font.SysFont("consolas", 20),
        "body": pygame.font.SysFont("consolas", 24),
        "title": pygame.font.SysFont("consolas", 34, bold=True),
        "hero": pygame.font.SysFont("consolas", 48, bold=True),
    }


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def draw_text(surface: pygame.Surface, text: str, font: pygame.font.Font, color, pos) -> pygame.Rect:
    img = font.render(text, True, color)
    return surface.blit(img, pos)


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, color=PANEL, border=PANEL_2) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=14)
    pygame.draw.rect(surface, border, rect, 2, border_radius=14)


def draw_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    series: PriceSeries,
    day_idx: int,
    fonts: Dict[str, pygame.font.Font],
) -> None:
    draw_panel(surface, rect)
    padding = 12 if (WIDTH <= 800 or HEIGHT <= 480) else 16
    inner = rect.inflate(-padding * 2, -padding * 2)
    prices = series.closes[max(0, day_idx - CHART_LOOKBACK + 1): day_idx + 1]
    if len(prices) < 2:
        return

    min_p = min(prices)
    max_p = max(prices)
    if abs(max_p - min_p) < 1e-9:
        max_p += 1.0
        min_p -= 1.0

    # Reserve space at the top for the chart labels.
    header_h = 44 if (WIDTH <= 800 or HEIGHT <= 480) else 64
    inner = pygame.Rect(inner.x, inner.y + header_h, inner.width, inner.height - header_h)

    # Grid
    for i in range(5):
        y = inner.top + i * inner.height / 4
        pygame.draw.line(surface, GRID, (inner.left, y), (inner.right, y), 1)
    for i in range(6):
        x = inner.left + i * inner.width / 5
        pygame.draw.line(surface, GRID, (x, inner.top), (x, inner.bottom), 1)

    points = []
    for i, price in enumerate(prices):
        x = inner.left + (i / (len(prices) - 1)) * inner.width
        y = inner.bottom - ((price - min_p) / (max_p - min_p)) * inner.height
        points.append((x, y))

    start_price = prices[0]
    end_price = prices[-1]
    line_color = POSITIVE if end_price >= start_price else NEGATIVE
    if len(points) >= 2:
        pygame.draw.lines(surface, line_color, False, points, 3)
    pygame.draw.circle(surface, WHITE, (int(points[-1][0]), int(points[-1][1])), 4)

    draw_text(surface, f"{series.name} ({series.symbol})", fonts["title"], TEXT, (rect.x + 14, rect.y + 10))
    change_pct = ((end_price / start_price) - 1.0) * 100.0
    draw_text(
        surface,
        f"{format_money(end_price)}   {change_pct:+.2f}%   {len(prices)}d",
        fonts["body"],
        line_color,
        (rect.x + 16, rect.y + 34),
    )
    draw_text(surface, f"H {format_money(max_p)}", fonts["small"], MUTED, (rect.right - 120, rect.y + 12))
    draw_text(surface, f"L {format_money(min_p)}", fonts["small"], MUTED, (rect.right - 120, rect.y + 28))


def draw_sidebar(surface: pygame.Surface, rect: pygame.Rect, state: GameState, fonts: Dict[str, pygame.font.Font]) -> None:
    draw_panel(surface, rect)
    draw_text(surface, "Market", fonts["title"], TEXT, (rect.x + 12, rect.y + 10))
    draw_text(surface, "Up/Down to select", fonts["small"], MUTED, (rect.x + 12, rect.y + 38))

    small_screen = WIDTH <= 800 or HEIGHT <= 480
    y = rect.y + (62 if small_screen else 90)
    row_h = 48 if small_screen else 66
    for idx, symbol in enumerate(state.ordered_symbols):
        row = pygame.Rect(rect.x + 8, y + idx * row_h, rect.width - 16, row_h - 6)
        selected = idx == state.selected_idx
        row_color = (36, 48, 68) if selected else (25, 34, 49)
        pygame.draw.rect(surface, row_color, row, border_radius=10)
        if selected:
            pygame.draw.rect(surface, ACCENT, row, 2, border_radius=10)

        p = state.price(symbol)
        pp = state.previous_price(symbol)
        daily = 0.0 if pp <= 0 else ((p / pp) - 1.0) * 100.0
        daily_color = POSITIVE if daily >= 0 else NEGATIVE
        held = state.holdings[symbol]
        name = state.market[symbol].name

        draw_text(surface, symbol, fonts["body"], WHITE, (row.x + 10, row.y + 4))
        draw_text(surface, format_money(p), fonts["small"], TEXT, (row.right - 80, row.y + 6))
        draw_text(surface, name, fonts["tiny"], MUTED, (row.x + 10, row.y + 24))
        draw_text(surface, f"{daily:+.2f}%", fonts["tiny"], daily_color, (row.right - 60, row.y + 24))
        if held > 0:
            draw_text(surface, f"H:{held}", fonts["tiny"], WARNING, (row.x + 72, row.y + 24))


def draw_status(surface: pygame.Surface, rect: pygame.Rect, state: GameState, fonts: Dict[str, pygame.font.Font], used_fallback: bool) -> None:
    draw_panel(surface, rect)
    equity = state.equity()
    port_pct = state.portfolio_return() * 100.0
    bench_pct = state.benchmark_return() * 100.0
    perf_color = POSITIVE if port_pct >= 0 else NEGATIVE
    spread = port_pct - bench_pct
    spread_color = POSITIVE if spread >= 0 else NEGATIVE
    small_screen = WIDTH <= 800 or HEIGHT <= 480

    if small_screen:
        x1 = rect.x + 12
        x2 = rect.x + 210
        x3 = rect.x + 380
        draw_text(surface, f"Date: {state.day_label()}", fonts["body"], TEXT, (x1, rect.y + 10))
        draw_text(surface, f"Equity: {format_money(equity)}", fonts["body"], TEXT, (x2, rect.y + 10))
        draw_text(surface, f"SPY: {bench_pct:+.2f}%", fonts["body"], MUTED, (x3, rect.y + 10))
        draw_text(surface, f"Cash: {format_money(state.cash)}", fonts["small"], TEXT, (x1, rect.y + 38))
        draw_text(surface, f"Return: {port_pct:+.2f}%", fonts["small"], perf_color, (x2, rect.y + 38))
        draw_text(surface, f"Alpha: {spread:+.2f}%", fonts["small"], spread_color, (x3, rect.y + 38))
        mode = "AUTO" if state.auto_play else "MANUAL"
        mode_color = WARNING if state.auto_play else ACCENT
        draw_text(surface, f"Mode: {mode}", fonts["small"], mode_color, (x1, rect.y + 60))
        src_text = "Data: mixed live/fallback" if used_fallback else "Data: yfinance history"
        draw_text(surface, src_text, fonts["tiny"], MUTED, (x2, rect.y + 62))
        draw_text(surface, "B/S trade  A max  D all  Space auto", fonts["tiny"], MUTED, (x2, rect.y + 10))
        return

    draw_text(surface, f"Date: {state.day_label()}", fonts["body"], TEXT, (rect.x + 18, rect.y + 16))
    draw_text(surface, f"Cash: {format_money(state.cash)}", fonts["body"], TEXT, (rect.x + 18, rect.y + 54))
    draw_text(surface, f"Equity: {format_money(equity)}", fonts["body"], TEXT, (rect.x + 280, rect.y + 16))
    draw_text(surface, f"Return: {port_pct:+.2f}%", fonts["body"], perf_color, (rect.x + 280, rect.y + 54))
    draw_text(surface, f"SPY: {bench_pct:+.2f}%", fonts["body"], MUTED, (rect.x + 520, rect.y + 16))
    draw_text(surface, f"Alpha vs SPY: {spread:+.2f}%", fonts["body"], spread_color, (rect.x + 520, rect.y + 54))

    mode = "AUTO" if state.auto_play else "MANUAL"
    mode_color = WARNING if state.auto_play else ACCENT
    draw_text(surface, f"Mode: {mode}", fonts["body"], mode_color, (rect.right - 240, rect.y + 16))
    src_text = "Data: mixed live/fallback" if used_fallback else "Data: yfinance history"
    draw_text(surface, src_text, fonts["small"], MUTED, (rect.right - 290, rect.y + 54))


def draw_positions(surface: pygame.Surface, rect: pygame.Rect, state: GameState, fonts: Dict[str, pygame.font.Font]) -> None:
    draw_panel(surface, rect)
    draw_text(surface, "Portfolio", fonts["title"], TEXT, (rect.x + 12, rect.y + 10))

    small_screen = WIDTH <= 800 or HEIGHT <= 480
    if small_screen:
        headers = [("Tk", 12), ("Sh", 78), ("Val", 138), ("Wt", 270)]
        header_y = rect.y + 38
        row_h = 24
        y = rect.y + 60
    else:
        headers = [("Ticker", 18), ("Shares", 150), ("Price", 260), ("Value", 390), ("Weight", 560)]
        header_y = rect.y + 54
        row_h = 40
        y = rect.y + 90

    for label, dx in headers:
        draw_text(surface, label, fonts["small"], MUTED, (rect.x + dx, header_y))

    total_equity = state.equity()
    visible_symbols = [s for s in state.ordered_symbols if state.holdings[s] > 0]
    if not visible_symbols:
        draw_text(surface, "No positions yet. Buy a stock to start.", fonts["body"], MUTED, (rect.x + 12, y + 6))
        return

    for i, symbol in enumerate(visible_symbols[:9]):
        row = pygame.Rect(rect.x + 8, y + i * row_h, rect.width - 16, row_h - 2)
        pygame.draw.rect(surface, (26, 36, 51), row, border_radius=8)
        shares = state.holdings[symbol]
        price = state.price(symbol)
        value = shares * price
        weight = 0.0 if total_equity <= 0 else (value / total_equity) * 100.0
        if small_screen:
            draw_text(surface, symbol, fonts["small"], TEXT, (row.x + 6, row.y + 4))
            draw_text(surface, str(shares), fonts["small"], TEXT, (row.x + 70, row.y + 4))
            draw_text(surface, format_money(value), fonts["small"], TEXT, (row.x + 130, row.y + 4))
            draw_text(surface, f"{weight:,.1f}%", fonts["small"], TEXT, (row.x + 262, row.y + 4))
        else:
            draw_text(surface, symbol, fonts["body"], TEXT, (row.x + 8, row.y + 6))
            draw_text(surface, str(shares), fonts["body"], TEXT, (row.x + 140, row.y + 6))
            draw_text(surface, format_money(price), fonts["body"], TEXT, (row.x + 250, row.y + 6))
            draw_text(surface, format_money(value), fonts["body"], TEXT, (row.x + 375, row.y + 6))
            draw_text(surface, f"{weight:,.1f}%", fonts["body"], TEXT, (row.x + 550, row.y + 6))


def draw_log(surface: pygame.Surface, rect: pygame.Rect, state: GameState, fonts: Dict[str, pygame.font.Font]) -> None:
    draw_panel(surface, rect)
    draw_text(surface, "Trade Log", fonts["title"], TEXT, (rect.x + 12, rect.y + 10))
    messages = state.log[-MAX_VISIBLE_LOG:]
    if not messages:
        messages = ["Use B/S to trade, Space to autoplay."]

    y = rect.y + (38 if (WIDTH <= 800 or HEIGHT <= 480) else 58)
    spacing = 24 if (WIDTH <= 800 or HEIGHT <= 480) else 30
    for i, message in enumerate(messages):
        color = MUTED if "Not enough cash" in message or "only own" in message else TEXT
        draw_text(surface, message, fonts["tiny" if (WIDTH <= 800 or HEIGHT <= 480) else "small"], color, (rect.x + 12, y + i * spacing))


def draw_help(surface: pygame.Surface, rect: pygame.Rect, fonts: Dict[str, pygame.font.Font]) -> None:
    draw_panel(surface, rect)
    lines = [
        "Controls",
        "Up/Down select  Left/Right move day",
        "B/S trade  Shift for 10  A max  D all",
        "Space autoplay  [ ] speed  R restart  Esc quit",
    ]
    draw_text(surface, lines[0], fonts["title"], TEXT, (rect.x + 12, rect.y + 8))
    for i, line in enumerate(lines[1:]):
        draw_text(surface, line, fonts["small"], MUTED, (rect.x + 12, rect.y + 34 + i * 18))


def draw_finish_overlay(surface: pygame.Surface, state: GameState, fonts: Dict[str, pygame.font.Font]) -> None:
    shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 150))
    surface.blit(shade, (0, 0))

    box_w, box_h = (460, 220) if (WIDTH <= 800 or HEIGHT <= 480) else (560, 280)
    box = pygame.Rect(WIDTH // 2 - box_w // 2, HEIGHT // 2 - box_h // 2, box_w, box_h)
    draw_panel(surface, box, color=(18, 26, 39), border=(86, 104, 138))
    port_pct = state.portfolio_return() * 100.0
    bench_pct = state.benchmark_return() * 100.0
    spread = port_pct - bench_pct
    draw_text(surface, "Simulation Complete", fonts["hero"], WHITE, (box.x + 24, box.y + 20))
    draw_text(surface, f"Final equity: {format_money(state.equity())}", fonts["body"], TEXT, (box.x + 26, box.y + 78))
    draw_text(surface, f"Your return: {port_pct:+.2f}%", fonts["body"], POSITIVE if port_pct >= 0 else NEGATIVE, (box.x + 26, box.y + 108))
    draw_text(surface, f"SPY benchmark: {bench_pct:+.2f}%", fonts["body"], MUTED, (box.x + 26, box.y + 138))
    draw_text(surface, f"Alpha vs SPY: {spread:+.2f}%", fonts["body"], POSITIVE if spread >= 0 else NEGATIVE, (box.x + 26, box.y + 168))
    draw_text(surface, "Press R to play again.", fonts["small"], WARNING, (box.x + 26, box.y + box_h - 26))


def reset_state(market: Dict[str, PriceSeries]) -> GameState:
    state = GameState(market=market, ordered_symbols=[s for s, _ in SYMBOLS])
    fallback_symbols = [s.symbol for s in market.values() if s.is_fallback]
    if fallback_symbols:
        joined = ", ".join(fallback_symbols)
        state.add_log(f"Fallback data used for: {joined}")
    else:
        state.add_log("Loaded historical data with yfinance.")
    state.add_log("Try to beat SPY by the end of the replay.")
    return state


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Stock Investing Game")
    clock = pygame.time.Clock()
    fonts = make_fonts()

    market, used_fallback = load_market()
    state = reset_state(market)

    while True:
        dt = clock.tick(FPS)
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                elif event.key == pygame.K_UP:
                    state.selected_idx = (state.selected_idx - 1) % len(state.ordered_symbols)
                elif event.key == pygame.K_DOWN:
                    state.selected_idx = (state.selected_idx + 1) % len(state.ordered_symbols)
                elif event.key == pygame.K_LEFT:
                    if not state.auto_play:
                        state.day_idx = max(0, state.day_idx - 1)
                        state.finished = False
                elif event.key == pygame.K_RIGHT:
                    if not state.auto_play and state.day_idx < state.max_day_index:
                        state.advance_day(1)
                elif event.key == pygame.K_SPACE:
                    state.auto_play = not state.auto_play
                elif event.key == pygame.K_b:
                    state.buy(10 if (mods & pygame.KMOD_SHIFT) else 1)
                elif event.key == pygame.K_a:
                    state.buy_max()
                elif event.key == pygame.K_s:
                    state.sell(10 if (mods & pygame.KMOD_SHIFT) else 1)
                elif event.key == pygame.K_d:
                    state.sell_all()
                elif event.key == pygame.K_LEFTBRACKET:
                    state.auto_advance_ms = min(2000, state.auto_advance_ms + 100)
                    state.add_log(f"Autoplay speed set to {state.auto_advance_ms} ms/day")
                elif event.key == pygame.K_RIGHTBRACKET:
                    state.auto_advance_ms = max(100, state.auto_advance_ms - 100)
                    state.add_log(f"Autoplay speed set to {state.auto_advance_ms} ms/day")
                elif event.key == pygame.K_r:
                    state = reset_state(market)

        if state.auto_play and not state.finished and now - state.last_advance_time >= state.auto_advance_ms:
            state.advance_day(1)
            state.last_advance_time = now

        screen.fill(BG)

        if WIDTH <= 800 or HEIGHT <= 480:
            sidebar_rect = pygame.Rect(12, 12, 200, HEIGHT - 24)
            status_rect = pygame.Rect(224, 12, WIDTH - 236, 84)
            chart_rect = pygame.Rect(224, 104, WIDTH - 236, 180)
            positions_rect = pygame.Rect(224, 292, 356, HEIGHT - 304)
            log_rect = pygame.Rect(588, 292, WIDTH - 600, HEIGHT - 304)

            draw_sidebar(screen, sidebar_rect, state, fonts)
            draw_status(screen, status_rect, state, fonts, used_fallback)
            draw_chart(screen, chart_rect, state.market[state.selected_symbol], state.day_idx, fonts)
            draw_positions(screen, positions_rect, state, fonts)
            draw_log(screen, log_rect, state, fonts)
        else:
            status_rect = pygame.Rect(300, 20, WIDTH - 320, 100)
            chart_rect = pygame.Rect(300, 132, WIDTH - 320, 330)
            positions_rect = pygame.Rect(300, 474, 700, 176)
            help_rect = pygame.Rect(300, 660, WIDTH - 320, 80)
            log_rect = pygame.Rect(1012, 474, 248, 266)
            sidebar_rect = pygame.Rect(20, 20, 260, 720)

            draw_sidebar(screen, sidebar_rect, state, fonts)
            draw_status(screen, status_rect, state, fonts, used_fallback)
            draw_chart(screen, chart_rect, state.market[state.selected_symbol], state.day_idx, fonts)
            draw_positions(screen, positions_rect, state, fonts)
            draw_log(screen, log_rect, state, fonts)
            draw_help(screen, help_rect, fonts)

        if state.finished:
            draw_finish_overlay(screen, state, fonts)

        pygame.display.flip()


if __name__ == "__main__":
    main()
