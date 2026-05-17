import polars as pl 
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Filter:
    exclude_quality = ['keywords:', 'Keywords:' 'query=', 'http', 'Abstract', 'Abstract '
    'ADVERTISEMENT RETURN TO ISSUE', 'PAPER ACCEPTED FOR PUBLICATION'
    'Article Views', 'Altimetric-Citations', 
    'Copyright', 'copyright', '©'
    'reference to this paper', 'Google Scholar' ]

    exclude_lang = [
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
        lf: pl.LazyFrame,
        col: str,
        min_len:  int,
        level: int = 1, #low=1 medium=2 high=3
    ):
    ## LOW 

    # content contains filtering
    # dodo measure delta here 
    l1 = lf.select(pl.len()).collect(engine='streaming').item()
    lf = lf.with_columns(
            pl.when(pl.col(col).str.contains_any(Filter.exclude_quality))
            .then(pl.lit(None, dtype=lf.schema[col]))
            .otherwise(pl.col(col))
            .alias(col)
        )
    l2 = lf.drop_nulls(subset=col).select(pl.len()).collect(engine='streaming').item()
    print(f'Filled {l1 - l2} {col} rows with {None}')
    lf = lf.with_columns(
            pl.when(pl.col(col).str.contains_any(Filter.exclude_lang))
            .then(pl.lit('unknown', dtype=lf.schema[col]))
            .otherwise(pl.col('language'))
            .alias('language')
        )

    ## MEDIUM
    if level > 1:
    # std filtering
        lf = lf.with_columns(
            pl.col(col).str.len_chars().alias(f'{col}_len')
        )
        stats = lf.select(f'{col}_len').collect(engine='streaming')
        mean: float = stats[f'{col}_len'].mean()
        std: float = stats[f'{col}_len'].std()
        high: float = mean + (std * 3)  
        low: float = max(mean - (std * 3), min_len)
        lf = lf.with_columns(
            pl.when((pl.col(f'{col}_len') > low) & pl.col(f'{col}_len') < high)
            .then(pl.lit(None, dtype=lf.schema[col]))
            .otherwise(pl.col(col))
            .alias(col)
        )
        lf = lf.drop(f'{col}_len')

    ## HIGH
    # content ends filtering 
    if level > 2:
        lf = lf.filter(pl.col(col).str.ends_with('.'))

    return lf
