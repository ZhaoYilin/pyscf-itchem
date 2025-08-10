import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps

def parallel_grid_integration(n_processes=4, batch_size=8000):
    """
    Decorator to parallelize grid-based integration functions.

    Parameters
    ----------
    n_processes : int, optional
        Number of processes for parallelization, by default 4.
    batch_size : int, optional
        Batch size for each process, by default 8000.

    Returns
    -------
    callable
        A decorated function that performs grid-based integration in parallel.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract grid coordinates from the function arguments
            # Assumes the grid coordinates are the second argument (index 1)
            grids_coords = args[1]
            ngrids = grids_coords.shape[0]  # Number of grid points

            # Split grid points into batches
            batches = [slice(i, min(i + batch_size, ngrids)) for i in range(0, ngrids, batch_size)]

            # Function to process a single batch
            def process_batch(batch_slice):
                # Create new args with the batch slice applied to the grid coordinates
                new_args = list(args)
                new_args[1] = grids_coords[batch_slice]
                return func(*new_args, **kwargs)

            # Use ProcessPoolExecutor to parallelize batch processing
            results = []
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    results.append(future.result())

            # Combine results from all batches
            if isinstance(results[0], np.ndarray):
                return np.concatenate(results, axis=-1)  # Combine along the last axis
            else:
                return sum(results)  # For scalar results, sum them up

        return wrapper
    return decorator