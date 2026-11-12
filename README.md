# Pairs Trading Platform

> **Projekt i implementacja modułowej platformy handlu algorytmicznego wykorzystującej strategię Pairs Trading oraz uczenie maszynowe do optymalizacji decyzji inwestycyjnych**

Projekt zespołowy realizowany przez 3 osoby w ciągu dwóch semestrów (październik 2026 – maj 2027).

---

## O projekcie

Budujemy platformę w Pythonie, która pozwala:

1. wybrać zestaw spółek z indeksu **S&P 500** do analizy,
2. **automatycznie znaleźć pary** spółek powiązanych statystycznie (kointegracja),
3. wybrać **strategię bazową** i opcjonalny **model uczenia maszynowego** filtrujący sygnały,
4. skonfigurować **parametry handlu i ryzyka** (kapitał, prowizje, progi wejścia/wyjścia, stop-loss),
5. przeprowadzić **symulację na danych historycznych** (własny silnik backtestingu),
6. przejrzeć **wyniki, wykresy i historię transakcji** oraz porównać warianty z ML i bez ML.

## Cel

Głównym celem jest zbudowanie **modułowej, rozszerzalnej platformy** oraz rzetelne sprawdzenie, czy **uczenie maszynowe poprawia wyniki klasycznej strategii Pairs Trading**. Wynik negatywny (ML nie pomaga) także jest pełnoprawnym wnioskiem, o ile został poprawnie zmierzony.

Projekt obejmuje trzy dziedziny:

| Dziedzina | Zakres |
|---|---|
| **Algorytmika i analiza danych** | selekcja par, kointegracja, generowanie sygnałów, zarządzanie ryzykiem, metryki |
| **Sztuczna inteligencja** | inżynieria cech, etykietowanie, trening i walidacja modeli, analiza wpływu ML |
| **Inżynieria oprogramowania** | architektura, baza danych, silnik backtestingu, API, frontend, testy, wdrożenie |

## Najważniejsze założenia

- **Dane:** dzienne notowania spółek z S&P 500 (źródło: yfinance), przechowywane lokalnie.
- **Brak podglądania przyszłości:** każda decyzja (wybór par, parametry, trening modelu) opiera się wyłącznie na danych z przeszłości (podejście *walk-forward*).
- **Realistyczna symulacja:** zlecenia realizowane po cenie otwarcia następnego dnia, z uwzględnieniem prowizji i poślizgu cenowego.
- **Modułowość:** strategie i modele ML to wymienne wtyczki implementujące wspólne interfejsy.
- **Projekt badawczo-symulacyjny:** platforma nie handluje na prawdziwym rynku.

## Architektura (w skrócie)

```
Dane (Parquet) → Selekcja par → Strategia (sygnały) → Filtr ML → Backtester → Baza danych
                                                                                   ↓
                                    Frontend (Streamlit)  ←──────  API (FastAPI)  ←┘
```

Szczegóły: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), interfejsy: [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Technologie

| Obszar | Technologia |
|---|---|
| Język | Python 3.11 |
| Dane i obliczenia | pandas, numpy, yfinance, Parquet |
| Statystyka | statsmodels, scipy |
| Uczenie maszynowe | scikit-learn, XGBoost, Optuna, SHAP |
| Backend / API | FastAPI, Pydantic |
| Baza danych | PostgreSQL, SQLAlchemy, Alembic |
| Frontend | Streamlit, Plotly |
| Testy i jakość | pytest, ruff, GitHub Actions |
| Uruchamianie | Docker, Docker Compose |


## Zespół i podział ról

| Osoba | Rola | Główne obszary |
|---|---|---|
| Osoba 1 | Strategia i analiza danych | selekcja par, kointegracja, sygnały, ryzyko, metryki |
| Osoba 2 | Uczenie maszynowe | cechy, etykietowanie, modele, walidacja czasowa, analiza wpływu ML |
| Osoba 3 | Inżynieria oprogramowania | architektura, baza, backtester, API, frontend, testy, wdrożenie |


## Zasady pracy

- Gałąź `main` jest zawsze działająca; nowe funkcje powstają na osobnych gałęziach (`feat/...`) i trafiają do `main` przez Pull Request.
- Każdy PR musi przejść CI (testy + linter) i zostać przejrzany przez co najmniej jedną osobę.
- Notebooki służą tylko do eksploracji; kod używany przez platformę trafia do `src/`.

## Status

| Kamień milowy | Termin | Status |
|---|---|---|
| M0: Fundamenty (repo, kontrakty, dane) | 29.11.2026 | ⏳ w toku |
| M1: Baseline end-to-end (pokaz) | 31.01.2027 | ⬜ |
| M2: Integracja ML | 28.03.2027 | ⬜ |
| M3: Platforma kompletna | 09.05.2027 | ⬜ |
| M4: Finał (raport, prezentacja) | 31.05.2027 | ⬜ |
