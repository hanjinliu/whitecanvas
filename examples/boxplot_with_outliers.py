from whitecanvas import new_canvas
import pandas as pd

def main():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    df = pd.read_csv(url)

    canvas = new_canvas("matplotlib:qt")

    layer = (
        canvas.cat_x(df, "smoker", "tip")
        .add_violinplot(color="sex")
        .as_edge_only()
        .with_outliers(symbol="D")
    )
    canvas.add_legend()
    canvas.show(block=True)

if __name__ == "__main__":
    main()
