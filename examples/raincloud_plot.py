from whitecanvas import new_canvas
import pandas as pd

def main():
    canvas = new_canvas(
        backend="matplotlib",
        palette=["#EE6363", "#7777FF", "#57A557"],
    )

    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    x = "species"
    y = "sepal_width"
    canvas.cat(df, by=x, offsets=-0.3).to_stripplot(y).with_edge(color="#3F3F00")
    canvas.cat(df, by=x, offsets=0).to_boxplot(y)
    canvas.cat(df, by=x, offsets=0).mean().to_markers(y, size=10, symbol="+", color="black")
    canvas.cat(df, by=x, offsets=0.2).to_violinplot(y, violin_width=0.5, shape="right").with_edge(color="#3F3F00")

    canvas.x.lim = (-0.7, 2.7)
    canvas.show()

if __name__ == "__main__":
    main()
