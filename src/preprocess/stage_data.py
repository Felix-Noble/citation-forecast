import os

def main(
    raw: str,
    out: str,
    start_date: int,
    end_date: int,
    field_id: int,
    drop_na_cols: list[str],
) -> None:
    
    client = Client(n_workers=4, threads_per_worker=8, memory_limit=f'{58/4}GB')

    ddf = dd.read_parquet(
        raw,
        engine='fastparquet'
    )
     
    ddf = ddf[(ddf['publication_date_int'] >= start_date) & (ddf['publication_date_int'] < end_date) & (ddf['field_id'] == field_id)] 

    if drop_na_cols:
        ddf = ddf.dropna(subset=drop_na_cols)

    ddf = ddf.repartition(npartitions=64)
    print('writing parquet')
    write_op = ddf.to_parquet(
        out,
        engine='fastparquet',
        compute=False,
        overwrite=True,
    )
    progress(client.compute(write_op))
    print('finished')

if __name__ == '__main__':
    import dask.dataframe as dd
    from dask.distributed import Client, progress
    from config.config import config
    import datetime
    start_date = datetime.date(1960, 1, 1).toordinal()
    end_date = datetime.date(2020, 1, 1).toordinal()
    field_id = 32
    
    if config.data.staged.exists():
        selection = input('output folder exists, overwrite? Y/n : ')
        if not selection.lower() == 'y':
            quit()

    print('starting main body')
    main(
        str(config.data.raw),
        str(config.data.staged / 'psychology'),
        start_date,
        end_date,
        field_id,
        ['id', 'abstract'],
    )
