import pandas as pd
from whitecanvas import new_canvas

def main():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url).dropna()

    canvas = new_canvas("matplotlib:qt")

    (
        canvas
        .cat(df, x="bill_length_mm", y="flipper_length_mm")
        .add_markers(color="species", size="bill_depth_mm")
    )

    canvas.add_legend()
    canvas.show(block=True)

if __name__ == "__main__":
    main()
