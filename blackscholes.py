# black_scholes.pypi
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Optional
from scipy.stats import norm


@dataclass(frozen=True)
class OptionParams:
    S: float      # spot
    K: float      # strike
    r: float      # risk-free annual rate (en décimal)
    sigma: float  # volatilité annuelle (en décimal)
    T: float      # maturité en années


# -------- core --------
def _d1_d2(p: OptionParams):
    if p.S <= 0 or p.K <= 0 or p.sigma <= 0 or p.T <= 0:
        raise ValueError("S, K, sigma, T doivent être > 0")
    sqrtT = math.sqrt(p.T)
    d1 = (math.log(p.S / p.K) + (p.r + 0.5 * p.sigma**2) * p.T) / (p.sigma * sqrtT)
    d2 = d1 - p.sigma * sqrtT
    return d1, d2


def price_call(p: OptionParams) -> float:
    d1, d2 = _d1_d2(p)
    return p.S * norm.cdf(d1) - p.K * math.exp(-p.r * p.T) * norm.cdf(d2)


def price_put(p: OptionParams) -> float:
    d1, d2 = _d1_d2(p)
    return p.K * math.exp(-p.r * p.T) * norm.cdf(-d2) - p.S * norm.cdf(-d1)


# -------- greeks (sans dividende) --------
def delta(p: OptionParams, kind: Literal["call", "put"] = "call") -> float:
    d1, _ = _d1_d2(p)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0


def gamma(p: OptionParams) -> float:
    d1, _ = _d1_d2(p)
    return norm.pdf(d1) / (p.S * p.sigma * math.sqrt(p.T))


def vega(p: OptionParams) -> float:
    d1, _ = _d1_d2(p)
    # vega par point (1.0 = 100% de vol)
    return p.S * norm.pdf(d1) * math.sqrt(p.T)


def theta(p: OptionParams, kind: Literal["call", "put"] = "call") -> float:
    d1, d2 = _d1_d2(p)
    first = -(p.S * norm.pdf(d1) * p.sigma) / (2.0 * math.sqrt(p.T))
    if kind == "call":
        return first - p.r * p.K * math.exp(-p.r * p.T) * norm.cdf(d2)
    else:
        return first + p.r * p.K * math.exp(-p.r * p.T) * norm.cdf(-d2)


def rho(p: OptionParams, kind: Literal["call", "put"] = "call") -> float:
    _, d2 = _d1_d2(p)
    disc = math.exp(-p.r * p.T)
    if kind == "call":
        return p.K * p.T * disc * norm.cdf(d2)
    else:
        return -p.K * p.T * disc * norm.cdf(-d2)


# -------- helper: pretty print --------
def summarize(p: OptionParams, kind: Literal["call", "put"] = "call") -> str:
    price = price_call(p) if kind == "call" else price_put(p)
    return (
        f"{kind.upper()} price: {price:.6f}\n"
        f"Delta: {delta(p, kind):.6f}\n"
        f"Gamma: {gamma(p):.6f}\n"
        f"Vega : {vega(p):.6f}\n"
        f"Theta: {theta(p, kind):.6f}\n"
        f"Rho  : {rho(p, kind):.6f}"
    )


# -------- CLI --------
def _parse_args(argv: Optional[list[str]] = None):
    import argparse
    ap = argparse.ArgumentParser(description="Black–Scholes pricer (call/put) avec Greeks.")
    ap.add_argument("--S", type=float, help="Spot")
    ap.add_argument("--K", type=float, help="Strike")
    ap.add_argument("--r", type=float, help="Taux sans risque (annuel)")
    ap.add_argument("--sigma", type=float, help="Volatilité (annuelle)")
    ap.add_argument("--T", type=float, help="Maturité (années)")
    ap.add_argument("--kind", choices=["call", "put"], default="call")
    ap.add_argument("--example", action="store_true",
                    help="Utiliser l'exemple S=100,K=105,r=0.05,sigma=0.2,T=1")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    args = _parse_args(argv)
    if args.example or any(v is None for v in [args.S, args.K, args.r, args.sigma, args.T]):
        # valeurs par défaut (cas d'usage rapide)
        p = OptionParams(S=100, K=105, r=0.05, sigma=0.2, T=1.0)
        kind = args.kind
        print(">> Exemple par défaut S=100, K=105, r=0.05, sigma=0.2, T=1.0")
    else:
        p = OptionParams(S=args.S, K=args.K, r=args.r, sigma=args.sigma, T=args.T)
        kind = args.kind

    print(summarize(p, kind))


if __name__ == "__main__":
    main()
