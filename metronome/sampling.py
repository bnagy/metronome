import pandas as pd
import numpy as np


def subset(df: pd.DataFrame, frm: int, to: int, meter: str) -> pd.DataFrame:
    """
    Subsets the given dataframe (in Plechac format) to find records composed
    between `frm` and `to` in the given meter

    In:
        df (pd.DataFrame): dataframe to work on
        frm, to (int): the 'year' field to subset (poem composition year)
        meter (str): Meter code to find (J5, T4, etc.)

    Returns:
        pd.DataFrame: the results
    """
    return df.query(
        f'year <= {to} and year >= {frm} and meter == "{meter}" and metronome.str.count("\|") >= 4'
    )


def _take_four(m):
    # Break metronome into list, take four lines starting at line numbers that
    # are divisible by 4 (for alignment)
    lines = m.split("|")[:-1]
    if len(lines) < 4:
        print(m)
        raise ValueError("not long enough")
    idx = np.random.randint(len(lines) // 4) * 4
    while True:
        samp = lines[idx : idx + 4]
        if len(samp) == 4:
            return "|".join(samp) + "|"


def synthetic_metronome(ss: pd.DataFrame, len: int) -> str:
    """
    Extract one synthetic metronome of length `len` from the given subset
    dataframe. Designed to run on subsets created with `subset` above.

    In:
        ss (pd.Dataframe): dataframe to run on
        len (int): length in lines of returned metronome (must be divisible by 4)

    Returns:
        str: one metronome
    """
    if len % 4 != 0:
        raise ValueError("len must be divisible by 4")
    return "".join(ss.sample(len // 4, replace=True).metronome.apply(_take_four))
