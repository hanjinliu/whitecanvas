import pandas as pd

from whitecanvas import new_canvas


def main():
    canvas = new_canvas(
        backend="matplotlib:qt",
        palette=["#EE6363", "#7777FF", "#57A557"],
    )

    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    x = "species"
    y = "sepal_width"
    cat_plt = canvas.cat(df)
    cat_plt.add_stripplot(
        x, y, color=x, extent=0.3
    ).with_edge(color="#3F3F00").with_shift(-0.3)
    cat_plt.add_boxplot(x, y, color=x, extent=0.3)
    cat_plt.mean().add_markers(x, y, color="black", size=10, symbol="+")
    (
        cat_plt
        .add_violinplot(x, y, color=x, extent=0.3, shape="right")
        .with_edge(color="#3F3F00")
        .with_shift(0.2)
    )

    canvas.show(block=True)

if __name__ == "__main__":
    main()
