from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

def build_epoch_progress(disable: bool=False):
    epoch_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TextColumn('<'), 
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 60.0 * 10, # hours
        disable = disable,
    )     
    epoch_progress.start()
    return epoch_progress

def build_example_progress(disable: bool=False):
    example_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TextColumn('<'), 
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 30, # mins
        disable= disable,
    )   
    return example_progress

def build_eval_example_progress(disable: bool=False):
    eval_example_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TextColumn('<'), 
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 30, # mins
        disable= disable,
    )
    return eval_example_progress
    
def build_mem_util_progress(disable: bool=False):
    mem_util_progress = Progress(
        TextColumn('[yellow] {task.description}', justify='right'),
        BarColumn(bar_width=30),
        TextColumn('[task.completed]{task.completed:.1f}/{task.total:.1f} MiB'),
        disable= disable,
    )
    return mem_util_progress

def build_progress_bars(disable=False) -> tuple[Progress, ...]:
    return build_epoch_progress(disable), \
            build_example_progress(disable), \
            build_eval_example_progress(disable)
