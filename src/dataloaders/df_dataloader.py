# ypyright: ignore[reportUnknownMemberType]
# type: ignore
import dask.dataframe as dd
import pandas as pd # pyright: ignore[reportMissingTypeStubs] 
from torch import Tensor
from torch.utils.data import Dataset
from typing import cast, override

class DF_DataLoader(Dataset[tuple[Tensor, ...]]):
    def __init__(self, 
                 data_path: str,
                 X: list[str],
                 y: list[str] | None=None,
                 ):
        super().__init__()
        if y is None:
            y = []

        columns = list(set(X + y))
        self.df: pd.DataFrame = cast(pd.DataFrame, dd.read_parquet(data_path, columns=columns).compute()) # pyright: ignore[reportPrivateImportUsage]
        self.X: list[str] = X
        self.y: list[str] = y

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        if len(self.y) > 0:
            return cast(tuple[Tensor, Tensor], (Tensor(self.df.loc[idx, self.X].values), Tensor(self.df.loc[idx, self.y].values)))
        return cast(tuple[Tensor], (Tensorrself.df.loc[idx, self.X].values, )))
