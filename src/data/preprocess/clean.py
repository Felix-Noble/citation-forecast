import polars as pl 
from dataclasses import dataclass
import os

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
        lf: pl.LazyFrame,
        columns: list[str],
        min_lens: dict[str, int] = {'abstract': 300}
    ):
     
    for col in columns:
        # std filtering
        lf = lf.with_columns(
            pl.col('abstract').str.len_chars().alias('abstract_len')
        )
        stats = lf.select(f'{col}_len').collect(engine='streaming')
        mean: float = stats[f'{col}_len'].mean()
        std: float = stats[f'{col}_len'].std()
        high: float = mean + (std * 3)  
        low: float = max(mean - (std * 3), min_lens.get(col, 0))
        lf = lf.filter(
            (pl.col(f'{col}_len') > low) & pl.col(f'{col}_len') < high
        )
        lf = lf.drop(f'{col}_len')
   
        # content filtering
        lf = lf.filter(~pl.col(col).str.contains_any(Filter.exclude))
        lf = lf.filter(pl.col(col).str.ends_with('.'))
    
    return lf
