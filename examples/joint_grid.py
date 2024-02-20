import pandas as pd
from whitecanvas import new_jointgrid

def main():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url).dropna()

    joint = (
        new_jointgrid("matplotlib:qt")
        .with_hist_x(shape="step")  # show histogram as the x-marginal distribution
        .with_kde_y(width=2)  # show kde as the y-marginal distribution
        .with_rug(width=1)  # show rug plot for both marginal distributions
    )

    layer = (
        joint.cat(df, x="bill_length_mm", y="flipper_length_mm")
        .add_markers(color="species")
    )

    joint.add_legend()
    joint.show(block=True)

if __name__ == "__main__":
    main()
