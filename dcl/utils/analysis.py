import pandas as pd

from dcl.datajoint.sweep import flatten_dict


def check_acc(col: str):
    return any('accuracy' in str(level).lower() for level in col) or any(
        'acc' in str(level).lower() for level in col)


def check_mse(col: str):
    return any('PredictiveMSE' in str(level) for level in col)


def check_r2(col: str):
    return any('R2' in str(level) for level in col)


def check_mean(col: str):
    return any('mean' in str(level).lower() for level in col)


def check_max(col: str):
    return any('max' in str(level).lower() for level in col)


def check_min(col: str):
    return any('min' in str(level).lower() for level in col)


def check_std(col: str):
    return any('std' in str(level).lower() for level in col)


def style_acc(styler):

    acc_columns = [
        col for col in styler.columns
        if check_acc(col) and (check_mean(col) or check_max(col))
    ]
    return styler.background_gradient(cmap='RdYlGn',
                                      subset=acc_columns,
                                      vmin=0.5,
                                      vmax=1)


def style_mse(styler):
    mse_columns = [
        col for col in styler.columns
        if check_mse(col) and (check_mean(col) or check_max(col))
    ]
    return styler.background_gradient(cmap='RdYlGn',
                                      subset=mse_columns,
                                      vmin=0,
                                      vmax=1)


def style_r2(styler):
    r2_columns = [
        col for col in styler.columns
        if check_r2(col) and (check_mean(col) or check_max(col))
    ]
    return styler.background_gradient(cmap='RdYlGn',
                                      subset=r2_columns,
                                      vmin=0,
                                      vmax=1)


def style_metrics(styler):
    styler_list = [style_acc, style_mse, style_r2]
    for style_fn in styler_list:
        styler = styler.pipe(style_fn)
    return styler


def unpack_results(results: pd.DataFrame):
    results.columns = [col.replace("__", ".") for col in results.columns]
    # Extract values from train.results dict into separate columns
    # Get first row to check types
    first_row = results.iloc[0]

    # Find columns containing dicts by checking first row
    dict_columns = []
    for col in results.columns:
        if isinstance(first_row[col], dict):
            dict_columns.append(col)
    print(dict_columns)

    # Process each dict column
    for col in dict_columns:
        # Flatten nested dicts
        results[col] = results[col].apply(lambda x: flatten_dict(x, sep="."))

        # Extract flattened keys into separate columns
        # Collect all unique keys across all rows
        all_keys = set()
        for d in results[col]:
            all_keys.update(d.keys())

        # Create new columns for each unique key
        for key in all_keys:
            results[f"{col}.{key}"] = results[col].apply(
                lambda x: x.get(key, None))
        del results[col]
        results = results.copy()
    return results
