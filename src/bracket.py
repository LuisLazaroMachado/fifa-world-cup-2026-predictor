from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
THIRD_TABLE = ROOT_DIR / "data" / "raw" / "third_place_assignment.csv"

# Ronda de 32 (reglamento FIFA 2026): partido -> (equipo A, equipo B)
# "1X" = ganador del grupo X, "2X" = segundo del grupo X, "3" = tercero (según la tabla)
ROUND_OF_32 = {
    73: ("2A", "2B"), 74: ("1E", "3"), 75: ("1F", "2C"), 76: ("1C", "2F"),
    77: ("1I", "3"), 78: ("2E", "2I"), 79: ("1A", "3"), 80: ("1L", "3"),
    81: ("1D", "3"), 82: ("1G", "3"), 83: ("2K", "2L"), 84: ("1H", "2J"),
    85: ("1B", "3"), 86: ("1J", "2H"), 87: ("1K", "3"), 88: ("2D", "2G"),
}

# Columna de la tabla de terceros -> partido de la ronda de 32 donde juega ese ganador
THIRD_SLOT_MATCH = {"1A": 79, "1B": 85, "1D": 81, "1E": 74,
                    "1G": 82, "1I": 77, "1K": 87, "1L": 80}

# Rondas siguientes: partido -> (ganador del partido X, ganador del partido Y)
KNOCKOUT = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),      # octavos
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),     # cuartos
    101: (97, 98), 102: (99, 100),                               # semifinales
    104: (101, 102),                                             # final
}

_third_table = pd.read_csv(THIRD_TABLE).set_index("third_groups")


def third_place_opponents(qualified_groups) -> dict:
    """{partido de ronda de 32: grupo del tercero que juega allí}."""
    row = _third_table.loc["".join(sorted(qualified_groups))]
    return {match: row[slot] for slot, match in THIRD_SLOT_MATCH.items()}


if __name__ == "__main__":
    # Clasificación real del Mundial 2026 (grupo -> (1º, 2º, 3º)) para validar la estructura
    real = {
        "A": ("Mexico", "South Africa", None), "B": ("Switzerland", "Canada", "Bosnia and Herzegovina"),
        "C": ("Brazil", "Morocco", None), "D": ("United States", "Australia", "Paraguay"),
        "E": ("Germany", "Ivory Coast", "Ecuador"), "F": ("Netherlands", "Japan", "Sweden"),
        "G": ("Belgium", "Egypt", None), "H": ("Spain", "Cape Verde", None),
        "I": ("France", "Norway", "Senegal"), "J": ("Argentina", "Austria", "Algeria"),
        "K": ("Colombia", "Portugal", "DR Congo"), "L": ("England", "Croatia", "Ghana"),
    }
    qualified = [g for g, t in real.items() if t[2]]
    thirds = third_place_opponents(qualified)
    pos = {"1": 0, "2": 1}

    def team(code, match):
        if code == "3":
            return real[thirds[match]][2]
        return real[code[1]][pos[code[0]]]

    for m, (a, b) in ROUND_OF_32.items():
        print(m, team(a, m), "vs", team(b, m))