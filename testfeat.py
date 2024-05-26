import pandas as pd
import numpy as np
import pandas._testing as tm
import pytest

@pytest.mark.parametrize(
    "method",
    [
        "sum",
        "min",
        "max",
        "mean",
        "median",
        "prod",
        "sem",
        "std",
        "var"
    ]
)
def test_skipna_cython(method):
    df = pd.DataFrame(
        {
            "l": ["A", "A", "A", "B", "B", "B"],
            "v": [-1, 1, -1, 1, np.nan, 1],
            "t": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-04"),
                pd.Timestamp("2020-01-06"),
                pd.NaT
            ],
            "td": [
                pd.Timedelta(days=1),
                pd.Timedelta(days=2),
                pd.Timedelta(days=3),
                pd.Timedelta(days=4),
                pd.Timedelta(days=6),
                pd.NaT
            ],
        }
    )
    result_cython = getattr(df.groupby("l").v, method)(skipna=False)
    expected = df.groupby("l").v.apply(lambda x: getattr(x, method)(skipna=False))
    tm.assert_series_equal(result_cython, expected, check_exact=False)
    if method in ["min", "max", "mean", "median", "std"]:
        result_ts = getattr(df.groupby("l").t, method)(skipna=False)
        expected_ts = df.groupby("l").t.apply(
            lambda x: getattr(x, method)(skipna=False)
        )
        tm.assert_series_equal(result_ts, expected_ts, check_exact=False)
        result_td = getattr(df.groupby("l").td, method)(skipna=False)
        expected_td = df.groupby("l").td.apply(
            lambda x: getattr(x, method)(skipna=False)
        )
        tm.assert_series_equal(result_td, expected_td, check_exact=False)
@pytest.mark.parametrize(
    "numba_method", [
        "sum", 
        "min",
        "max",  
        "std", 
        "var", 
        "mean"
    ]
)
def test_skipna_numba(numba_method):
    df = pd.DataFrame(
        {
            "l": ["A", "A", "A", "B", "B", "B"],
            "v": [-1, 1, -1, 1, 1, np.nan],
        }
    )
    result_numba = getattr(df.groupby("l").v, numba_method)(skipna=False, engine="numba")
    expected = df.groupby("l").v.apply(lambda x: getattr(x, numba_method)(skipna=False))
    tm.assert_series_equal(result_numba, expected, check_exact=False)
@pytest.mark.parametrize(
    "method",
    [
        "sum",
        "min",
        "max",
        "mean",
        "median",
        "prod",
        "sem",
        "std",
        "var"
    ]
)
def test_skipna_consistency(method):
    df = pd.DataFrame(
        {
            "l": ["A", "A", "A", "B", "B", "B"],
            "v": [-1, 1, -1, 1, np.nan, 1],
            "t": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-04"),
                pd.NaT,
                pd.Timestamp("2020-01-06")
            ],
            "td": [
                pd.Timedelta(days=1),
                pd.Timedelta(days=2),
                pd.Timedelta(days=3),
                pd.Timedelta(days=4),
                pd.NaT,
                pd.Timedelta(days=6)
            ],
        }
    )
    result_with_arg = getattr(df.groupby("l").v, method)(skipna=True)
    result_default = getattr(df.groupby("l").v, method)()
    tm.assert_series_equal(result_with_arg, result_default, check_exact=False)
    if method in ["min", "max", "mean", "median", "std"]:
        result_ts_with_arg = getattr(df.groupby("l").t, method)(skipna=True)
        result_ts_default = getattr(df.groupby("l").t, method)()
        tm.assert_series_equal(result_ts_with_arg, result_ts_default, check_exact=False)
        result_td_with_arg = getattr(df.groupby("l").td, method)(skipna=True)
        result_td_default = getattr(df.groupby("l").td, method)()
        tm.assert_series_equal(result_td_with_arg, result_td_default, check_exact=False)
@pytest.mark.parametrize(
    "numba_method", ["sum", "min", "max", "std", "var", "mean"]
)
def test_skipna_consistency_numba(numba_method):
    df = pd.DataFrame(
        {
            "l": ["A", "A", "A", "B", "B", "B"],
            "v": [-1, 1, -1, 1, np.nan, 1],
        }
    )
    result_with_arg = getattr(df.groupby("l").v, numba_method)(skipna=True, engine="numba")
    result_default = getattr(df.groupby("l").v, numba_method)(engine="numba")
    tm.assert_series_equal(result_with_arg, result_default, check_exact=False)
def test_skipna_string_sum():
    df = pd.DataFrame(
            {
                "l": ["A", "A", "A", "B", "B", "B"],
                "v": ["foo", "bar", "baz", "foo", pd.NA, "foo"],
            }
    )
    result_cython = df.groupby('l').v.sum(skipna=False)
    expected = df.groupby('l').v.apply(lambda x: x.sum(skipna=False))
    tm.assert_series_equal(result_cython, expected, check_exact=False)
def test_skipna_string_sum_consistency():
    df = pd.DataFrame(
            {
                "l": ["A", "A", "A", "B", "B", "B"],
                "v": ["foo", "bar", "baz", "foo", pd.NA, "foo"],
            }
    )
    result_cython = df.groupby('l').v.sum(skipna=True)
    expected = df.groupby('l').v.sum()
    tm.assert_series_equal(result_cython, expected, check_exact=False)

