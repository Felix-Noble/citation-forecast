from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

def build_progress_bars(disable=False) -> tuple[Progress, ...]:
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
    mem_util_progress = Progress(
        TextColumn('[yellow] {task.description}', justify='right'),
        BarColumn(bar_width=30),
        TextColumn('[task.completed]{task.completed:.1f}/{task.total:.1f} MiB'),
        disable= disable,
    )

    epoch_progress.start()
    example_progress.start()
    eval_example_progress.start()
    #mem_util_progress.start() # no report on strix halo
    return epoch_progress, example_progress, eval_example_progress, mem_util_progress
