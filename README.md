# Metronome [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


## About

This is the metronome package. It is not fully cooked.

## Installation

`pip install git+https://github.com/bnagy/metronome@main`

## Usage

Check out my hot documentation skills!

```python
def dist_matrix(df: pd.DataFrame, col: str = "metronome") -> pd.DataFrame:
    """
    Take a set of n metronomes and produce an nxn matrix of distances, suitable
    for passing to hclust in R. The 'distances' are constructed from the
    BioPython local alignment score (higher is better), normalised by the length
    of the shorter string at each pairwise comparison. Those scores yield 1 for
    a perfect match, so the final matrix is 1 - normalised_score_matrix (small
    distance is a closer match).

    In:
        df (pd.DataFrame): data frame containing the metronomes col (str =
        "metronome"): name of the metronome column

    Returns:
        pd.DataFrame: the matrix as a dataframe
    """

def pair_score(
    a, b: str, scale: bool = True, match_dict: dict = MATCH_DICT, oe: tuple = (-3, -3)
) -> float:
    """
    For two metronome strings, output a pairwise score (higher is a better
    match). This method allows for some lower level tuning, eg by supplying a
    custom match dict to set the match / mismatch scores for each pair of
    symbols in the alphabet.

    In:
        a, b (str): strings to compare scale (bool = True): whether to normalise
        by the length of the shorter
            string. Set to False to get the raw BioPython score
        match_dict (dict): match dict to use for alignment scoring See BioPython
            docs for details. Uses Bio.Align.PairwiseAligner internally.
        oe (tuple = (-3, -3)): Open and extend penalties for gaps

    Returns:
        float: the score
    """
```

## Citation


## License & Acknowledgements

Code: BSD style, see the [LICENSE](LICENSE.txt)

## Contributing

Fork and PR. Particularly welcome would be:
- improving the packaging structure
- tests
- general addition of pythonicity
