import polars as pl 
from dataclasses import dataclass
import os

os.environ['POLARS_MAX_THREADS'] = '16'

@dataclass(frozen=True)
class Filter:
    exclude = ['keywords:', 'Keywords:' 'query=', 'http', 'Abstract', 'Abstract '
    'ADVERTISEMENT RETURN TO ISSUE', 'PAPER ACCEPTED FOR PUBLICATION'
    'Article Views', 'Altimetric-Citations', 
    'Copyright', 'copyright', '©'
    'reference to this paper', 'Google Scholar'
    # foreign connectives (remember to include space)
    'de ',
    
    # chinese characters
   # Particles and function words
    '的', '了', '在', '是', '和', '与', '及', '或', '为', '被',
    '有', '无', '以', '对', '对于', '根据', '按', '由',
    
    # Common academic connectives
    '因此', '而且', '然而', '但', '但是', '同时', '并', '并且',
    '此外', '另外', '进一步', '总之', '综上', '可见',
    
    # Common verbs (often semantically weak in abstracts)
    '表明', '显示', '证明', '说明', '指出', '认为', '发现',
    '提出', '方法', '研究', '分析', '讨论', '介绍', 
     # Common single char words
    '的','了','是','在','和','与','对','有','无','以','为','被','等',
        # Vowels with accents
    'à', 'á', 'â', 'ã', 'ä', 'å', 'æ',
    'è', 'é', 'ê', 'ë',
    'ì', 'í', 'î', 'ï',
    'ò', 'ó', 'ô', 'õ', 'ö', 'ø', 'œ',
    'ù', 'ú', 'û', 'ü',
    'ý', 'ÿ',
    
    # Uppercase versions
    'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ',
    'È', 'É', 'Ê', 'Ë',
    'Ì', 'Í', 'Î', 'Ï',
    'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ø', 'Œ',
    'Ù', 'Ú', 'Û', 'Ü',
    'Ý',
    
    # Consonants with diacriticals
    'ç', 'Ç',
    'ñ', 'Ñ',
    'ð', 'Ð',
    'þ', 'Þ',
    'ß',

    ]
    tails = [
    'English', 'english',
    '<' , '>', ';', '@', '?', '[', ']', '{', '}'
    '#', '~', '/', '-', '_', '+', '=', '\\', '`', '¬',
    '!', '£', '$', '%','^', '&', '*', '(', ')'
    ]
    remove = []

def main(
    origin: str,
    destination: str,
    field_id: int,
    ):
    if os.path.exists(destination):
        print(f'clear path before saving - {destination}')
        quit()
    os.makedirs(destination, exist_ok=True)
    lf = pl.scan_parquet(origin)
    
    # type filters 
    lf = lf.filter((pl.col('language') == 'en') & (pl.col('type') == 'article'))
    lf = lf.filter(pl.col('field_id') == field_id)

    # std filtering
    lf = lf.with_columns(
        pl.col('abstract').str.len_chars().alias('abstract_len')
    )
    stats = lf.select('abstract_len').collect(engine='streaming')
    mean: float = stats['abstract_len'].mean()
    std: float = stats['abstract_len'].std()
    high: float = mean + (std * 3)  
    low: float = max(mean - (std * 3), 300)
    lf = lf.filter(
        (pl.col('abstract_len') > low) & pl.col('abstract_len') < high
    )
    lf = lf.drop('abstract_len')

    lf = lf.with_columns(
        pl.col('title').str.len_chars().alias('title_len')
    )
    stats = lf.select('title_len').collect(engine='streaming')
    mean: float = stats['title_len'].mean()
    std: float = stats['title_len'].std()
    high: float = mean + (std * 3)  
    low: float = max(mean - (std * 3), 10)
    lf = lf.filter(
        (pl.col('title_len') > low) & pl.col('title_len') < high
    )
    lf = lf.drop('title_len')
   
    # content filtering
    lf = lf.filter(~pl.col('abstract').str.contains_any(Filter.exclude))
    lf = lf.filter(pl.col('abstract').str.ends_with('.'))
    lf = lf.filter(~pl.col('title').str.contains_any(Filter.exclude))
    
    i = 0
    n = 1_000_000
    while True:
        lf_slice = lf.slice(i*n, (i+1)*n)
        len = lf_slice.select(pl.len()).collect(engine='streaming').item()
        if len < 1:
            break
        print(f'writing part {i}, len={len}')
        lf_slice.sink_parquet(
            f'{destination}/part{i}.parquet',
            statistics=True,
            compression='zstd',
            compression_level=4
        ) 
        i += 1 

if __name__ == '__main__':
    main(
        origin = '/home/fnoble/data/staged/all',
        destination = '/home/fnoble/data/staged/psychology_2',
        field_id = 32,
    )
