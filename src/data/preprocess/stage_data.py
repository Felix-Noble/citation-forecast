import polars as pl
import os
import gc
# pyright: basic
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false

os.environ['POLARS_MAX_THREADS'] = '32'

def main(
    raw: str,
    out: str,
    start_date: int,
    end_date: int,
    field_id: int,
    drop_na_cols: list[str] = [],
    num_files:int = 64,
) -> None:
    
    df = pl.scan_parquet(raw, extra_columns='ignore')
    #df = df.filter((pl.col('publication_date_int') >= start_date) & (pl.col('publication_date_int') < end_date))
    #df = df.filter(pl.col('field_id') == field_id)
    #df = df.drop_nulls(subset=drop_na_cols)

    print('writing parquet')
    i = 0
    n = 1_000_000
    while True:
        
        df_part = df.slice(i*n, n)#.collect(engine='streaming')
        len = df_part.select(pl.len()).collect(engine='streaming').item()
        if len < 1:
            break
        df_part.sink_parquet(
            f'{out}/part{i}.parquet',
            statistics=True,
            compression='zstd',
            compression_level=8,
        )
        print(f'finished writing part{i}')
        i += 1
    print('finished')

if __name__ == '__main__':
    import os
    import sys
    sys.path.append('/home/fnoble/projects/citation-forecast/')
    from config import env
    import datetime
    start_date = datetime.date(1960, 1, 1).toordinal()
    end_date = datetime.date(2026, 1, 1).toordinal()
    field_id = 32
    
    field_name = 'all'
    os.makedirs(env.STAGED_LOC / field_name, exist_ok=True)

    print('starting main body')
    main(
        str(env.RAW_LOC),
        str(env.STAGED_LOC / field_name),
        start_date,
        end_date,
        field_id,
        ['id', 'title'],
    )
