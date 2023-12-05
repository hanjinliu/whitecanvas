from whitecanvas import new_canvas
import pandas as pd

def main():
    canvas = new_canvas(
        backend="matplotlib:qt",
        palette=["#EE6363", "#7777FF", "#57A557"],
    )

    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    x = "species"
    y = "sepal_width"
    cat_plt = canvas.cat(df, by=x)
    cat_plt.with_offset(-0.3).add_stripplot(y).with_edge(color="#3F3F00")
    cat_plt.with_offset(0).add_boxplot(y)
    cat_plt.with_offset(0).mean().add_markers(y, size=10, symbol="+", color="black")
    cat_plt.with_offset(0.2).add_violinplot(y, extent=0.5, shape="right").with_edge(color="#3F3F00")

    canvas.show(block=True)

if __name__ == "__main__":
    main()
