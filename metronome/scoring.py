import pandas as pd
import numpy as np
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
        batch_size: int = 1000,  # Process results in batches
    ) -> pd.DataFrame:
        """
        Compute a pairwise distance matrix in parallel using Ray and NumPy.

        In:
            df (pd.DataFrame): data frame containing the metronomes

            col (str ="metronome"): name of the metronome column

            batch_size (optional int = 1000): Limit for simultaneous ray futures

        Returns:
            pd.DataFrame: The final distance matrix.
        """
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df = df.copy().reset_index(drop=True)
        col_values = df[col].to_numpy()
        col_ref = ray.put(col_values)
        # Pre-allocate result array
        n = len(col_values)
        results = np.zeros((n, n), dtype=np.float64)

        # Launch parallel tasks in batches
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            # doing this in the loop lets the futures be GCd one they have been fetched
            batch_futures = [
                self._row_compare_idx.remote(self, i, col_ref)
                for i in range(batch_start, batch_end)
            ]
            batch_results = ray.get(batch_futures)  # Fetch only this batch
            # update result array one batch at a time. We treat the nxn result array as 1d here
            results[batch_start:batch_end, :] = np.array(
                batch_results, dtype=np.float64
            )

        # Convert to distance matrix
        lower_dm = 1 - results
        df = pd.DataFrame(lower_dm + lower_dm.T)  # Mirror across diagonal
        return df

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
