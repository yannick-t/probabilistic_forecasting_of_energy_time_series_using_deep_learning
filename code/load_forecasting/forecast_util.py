
def dataset_df_to_np(df):
    y = df.filter(regex='target').to_numpy()
    offset = df.filter(regex='offset').to_numpy()
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
        offset = offset.reshape(-1, 1)
    df = df.drop(columns=list(df.filter(regex='target')))
    df = df.drop(columns=list(df.filter(regex='offset')))
    x = df.to_numpy()

    return x, y, offset