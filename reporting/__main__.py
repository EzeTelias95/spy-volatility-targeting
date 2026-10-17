"""Write every numbered table and chart of the final document from the cache."""

import matplotlib.pyplot as plt
import pandas as pd

from data.download.config import DATASET_PATH
from reporting.figures import FIGURES, STYLE
from reporting.tables import TABLES, source

OUTPUT_DIR = DATASET_PATH.parent / "report"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for name, (title, build) in TABLES.items():
        build().to_csv(OUTPUT_DIR / f"{name}.csv")
        index.append(
            {"archivo": f"{name}.csv", "titulo": title, "fuente": source(name)}
        )
    with plt.rc_context(STYLE):
        for name, (title, build) in FIGURES.items():
            fig = build()
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
            plt.close(fig)
            index.append(
                {"archivo": f"{name}.png", "titulo": title, "fuente": source(name)}
            )
    pd.DataFrame(index).to_csv(OUTPUT_DIR / "indice.csv", index=False)


if __name__ == "__main__":
    main()
