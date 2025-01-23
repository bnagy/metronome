import pandas as pd
import ray
import Bio.Align
from Bio.Align import substitution_matrices

MATCH_DICT = {
    ("|", "|"): 3.5,
    ("w", "w"): 2.5,
    ("S", "S"): 3,
    (".", "."): 1,
    # penalise non-alignment at line ends more, because they're rarer. The idea
    # is to stop hendecasyllables being routinely padded out to match hexameter
    # lines
    ("|", "S"): -4,
    ("|", "w"): -4,
    ("|", "."): -4,
    ("w", "S"): -0.25,
    ("w", "."): 0,  # not sure. In theory we don't care if spaces mismatch
    ("S", "."): 0,  # because metre isn't based on a fixed number of words
}

OPEN_GAP_SCORE = -6
EXTEND_GAP_SCORE = -3
MODE = "local"


class Scorer:
    def __init__(
        self,
        mode: str = MODE,
        match_dict: dict[tuple[str, str], float] = MATCH_DICT,  # type: ignore
        open_gap_score: float = OPEN_GAP_SCORE,
        extend_gap_score: float = EXTEND_GAP_SCORE,
    ):
        self.aligner = Bio.Align.PairwiseAligner()
        self.aligner.mode = mode
        self._complete_dictionary(match_dict)
        mat = substitution_matrices.Array(data=match_dict)
        self.aligner.substitution_matrix = mat
        self.aligner.open_gap_score = open_gap_score
        self.aligner.extend_gap_score = extend_gap_score

    @staticmethod
    def _complete_dictionary(match_dict: dict[tuple[str, str], float]):
        additions = dict()
        for key in match_dict:
            reverse = key[::-1]
            if reverse not in match_dict:
                additions[reverse] = match_dict[key]
        match_dict |= additions

    @ray.remote
    def _row_compare_idx(self, idx: int, row: pd.Series) -> list[float]:
        r = [1.0] * len(row)
        for i, x in enumerate(row):
            if i > idx:
                break
            r[i] = self.pair_score(row[idx], x)
        return r

    def dist_matrix_parallel(
        self,
        df: pd.DataFrame,
        col: str = "metronome",
        mem_limit: int = 1 * 1024 * 1024 * 1024,
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
            df (pd.DataFrame): data frame containing the metronomes

            col (str ="metronome"): name of the metronome column

        Returns:
            pd.DataFrame: the matrix as a dataframe
        """
        if not col in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df = df.copy().reset_index(drop=True)
        mtrx = []

        # set up the ray futures, concurrency at the level of rows
        for i, _ in enumerate(df[col]):
            mtrx.append(
                self._row_compare_idx.options(memory=mem_limit).remote(
                    self, i, df[col].copy()
                )
            )
        # this blocks until all the rows are done
        lower_dm = 1 - pd.DataFrame.from_records(ray.get(mtrx))
        return lower_dm + lower_dm.T

    def dist_matrix(self, df: pd.DataFrame, col: str = "metronome") -> pd.DataFrame:
        """
        Take a set of n metronomes and produce an nxn matrix of distances, suitable
        for passing to hclust in R. The 'distances' are constructed from the
        BioPython local alignment score (higher is better), normalised by the length
        of the shorter string at each pairwise comparison. Those scores yield 1 for
        a perfect match, so the final matrix is 1 - normalised_score_matrix (small
        distance is a closer match).

        In:
            df (pd.DataFrame): data frame containing the metronomes

            col (str ="metronome"): name of the metronome column

        Returns:
            pd.DataFrame: the matrix as a dataframe
        """
        df = df.copy().reset_index(drop=True)
        mtrx = []
        if not col in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        for i, _ in enumerate(df[col]):
            r = [1.0] * len(df[col])
            for j, x in enumerate(df[col]):
                # calculate only the lower triangle
                if j > i:
                    break
                r[j] = self.pair_score(df[col][i], x)
            mtrx.append(r)
        lower_dm = 1 - pd.DataFrame.from_records(mtrx)
        # mirror across the diagonal
        return lower_dm + lower_dm.T

    def pair_score(
        self,
        a: str,
        b: str,
        scale: bool = True,
    ) -> float:
        """
        For two metronome strings, output a pairwise score in [0,1] (higher is a
        better match).

        In:
            a, b (str): strings to compare

            scale (bool = True): whether to normalise by the length of the
            shorter string. Set to False to get the raw BioPython score

        Returns:
            float: the score
        """

        score = float(self.aligner.score(a, b))
        # normalize score by the self-score of the shorter work, so a 'perfect match'
        # would be 1
        if not scale:
            return float(score)
        if len(a) >= len(b):
            return score / self.aligner.score(b, b)
        else:
            return score / self.aligner.score(a, a)
