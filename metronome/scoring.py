import pandas as pd
import ray
import Bio.Align
from Bio.Align import substitution_matrices

MATCH_DICT = {
    ("|", "|"): 2,
    ("w", "w"): 2,
    ("S", "S"): 2,
    # TODO it's kind of better to decrease the space vs space bonus, along the
    # same lines as reducing the penalty for mismatches, but then when you scale
    # a perfect match turns out to be less than 1, which is bad.
    (".", "."): 2,
    # penalise non-alignment at line ends more, because they're rarer. The idea
    # is to stop hendecasyllables being routinely padded out to match hexameter
    # lines
    ("|", "S"): -4,
    ("|", "w"): -4,
    ("|", "."): -4,
    ("w", "S"): 0,
    ("w", "."): 0,  # not sure. In theory we don't care if spaces mismatch
    ("S", "."): 0,  # because metre isn't based on a fixed number of words
}


class Scorer:
    def __init__(
        self, mode: str = "local", match_dict: dict = MATCH_DICT, oe: tuple = (-6, -3)
    ):
        self.aligner = Bio.Align.PairwiseAligner()
        self.aligner.mode = mode
        mat = substitution_matrices.Array(data=match_dict)
        self.aligner.substitution_matrix = mat
        self.aligner.open_gap_score = oe[0]
        self.aligner.extend_gap_score = oe[1]

    @ray.remote
    def _row_compare(self, s: str, row: pd.Series) -> list[float]:
        return [self.pair_score(s, x) for x in row]

    def dist_matrix_parallel(
        self, df: pd.DataFrame, col: str = "metronome"
    ) -> pd.DataFrame:
        """
        Take a set of n metronomes and produce an nxn matrix of distances, suitable
        for passing to hclust in R. The 'distances' are constructed from the
        BioPython local alignment score (higher is better), normalised by the length
        of the shorter string at each pairwise comparison. Those scores yield 1 for
        a perfect match, so the final matrix is 1 - normalised_score_matrix (small
        distance is a closer match).

        This version runs in parallel using ray.

        In:
            df (pd.DataFrame): data frame containing the metronomes col (str =
            "metronome"): name of the metronome column

        Returns:
            pd.DataFrame: the matrix as a dataframe
        """
        if not col in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")
        df = df.copy().reset_index(drop=True)
        mtrx = []
        for x in df[col]:
            mtrx.append(self._row_compare.remote(self, x, df[col].copy()))
        return 1 - pd.DataFrame.from_records(ray.get(mtrx))

    def dist_matrix(self, df: pd.DataFrame, col: str = "metronome") -> pd.DataFrame:
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
        df = df.copy().reset_index(drop=True)
        if not col in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")
        mtrx = df[col].apply(lambda str: [self.pair_score(str, x) for x in df[col]])
        # the score is a match strength (1 for perfect) whereas we want a distance
        # here (0 for identical)
        return 1 - pd.DataFrame.from_records(mtrx)

    def pair_score(self, a, b: str, scale: bool = True) -> float:
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
        score = float(self.aligner.score(a, b))
        # normalize score by the length of the shorter work, so a 'perfect match'
        # would be 1
        if not scale:
            return score
        if len(a) >= len(b):
            return score / len(b) / 2
        else:
            return score / len(a) / 2
