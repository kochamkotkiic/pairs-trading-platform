# Kwestie otwarte

Decyzje, których zespół jeszcze nie podjął. Nie rozstrzygamy ich w kodzie „po cichu”: najpierw ustalenie (zespół, w razie potrzeby prowadzący), potem zmiana kontraktu (`docs/contract.md`) i kodu.

| ID | Pytanie | Status | Od | Właściciel |
|---|---|---|---|---|
| OQ-1 | W czym przechowywać wyniki symulacji? | otwarte | 27.11.2026 | O3 |
| OQ-2 | Jak ma wyglądać dashboard? | otwarte | 27.11.2026 | O3 (+O1, O2) |
| OQ-3 | Jaka będzie finalna infrastruktura platformy? | otwarte | 27.11.2026 | O3 |

## OQ-1: Przechowywanie wyników symulacji

**Co trzeba przechowywać** (niezależnie od technologii, zakres w `docs/contract.md` §13): przebiegi symulacji z konfiguracją i statusem, dziennik transakcji, krzywą kapitału, metryki, wybrane pary, opcjonalnie sygnały.

**Możliwe kierunki (do oceny, bez rozstrzygnięcia):**

- relacyjna baza danych działająca jako osobna usługa,
- baza plikowa, działająca w procesie aplikacji,
- zwykłe pliki na dysku (np. osobny folder na każdy przebieg).

**Od tej decyzji zależą:** moduł `ptp.db`, endpointy API zwracające wyniki (`docs/contract.md` §12), `docker-compose.yml`, konfiguracja CI.

## OQ-2: Wygląd dashboardu

**Kontekst.** Wiemy, co dashboard ma umożliwiać (funkcje od prowadzącego): wybór instrumentów, strategii i modelu ML, ustawienie parametrów handlu i ryzyka, uruchomienie symulacji, przegląd wyników, wykresów i historii transakcji oraz porównanie wariantu bez ML i z ML. Nie ustaliliśmy jeszcze, jak ma to wyglądać.

**Do ustalenia:**

- podział na strony lub zakładki i przepływ użytkownika (od konfiguracji do wyników),
- jakie wykresy i tabele pokazujemy na każdej stronie (np. spread i Z-score pary, krzywa kapitału, porównanie przebiegów, ważność cech ML),
- kto przygotowuje które wykresy (szkielet O3, wykresy domenowe O1 i O2).

**Od tej decyzji zależą:** endpointy API potrzebne interfejsowi (`docs/contract.md` §12), zakres prac O3 w F1 i F3, wkład O1 i O2 w interfejs.

**Proponowany krok:** szkic ekranów (np. makieta) do omówienia w zespole, a potem z prowadzącym.

## OQ-3: Finalna infrastruktura

**Kontekst.** Na razie zakładamy uruchamianie lokalne jednym poleceniem (Docker Compose), bez docelowego serwera. 

**Do ustalenia:**

- z jakich usług składa się platforma (API, interfejs, ewentualne przechowywanie wyników),
- co uruchamiamy w CI (np. czy testy potrzebują dodatkowych usług).

**Od tej decyzji zależą:** `docker-compose.yml`, pliki Dockerfile, konfiguracja CI, instrukcja uruchomienia w `README.md`.

**Zależy od:** OQ-1 (przechowywanie wyników).
