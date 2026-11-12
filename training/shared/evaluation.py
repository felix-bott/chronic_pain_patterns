"""Evaluation helpers shared by the classical trainers (FBR, FBS, HCF, CPN, final-layer models)."""
import numpy as np
from scipy.stats import pearsonr


# Define a custom Pearson correlation scorer
def pearson_corr(y_true, y_pred, tol=1e-10):
    # Handle cases where one of the inputs is constant
    if np.std(y_pred) < tol or np.std(y_true) < tol:
        return 0 # by setting this to -1, one avoids all zero coefficients in the final model
    return pearsonr(y_true.ravel(), y_pred.ravel())[0]

def aggregate_by_ids(X, ids, ids_master, aggregation_method):
    """
    Reorder rows of X and ids such that ids matches ids_master, and
    aggregate rows of X by averaging based on unique values in ids_master.

    Parameters:
        X (numpy.ndarray): 2D array where rows correspond to data points.
        ids (numpy.ndarray): 1D array of IDs associated with rows of X.
        ids_master (numpy.ndarray): 1D array of unique IDs in the desired order.

    Returns:
        X_reordered (numpy.ndarray): Reordered 2D array with aggregated rows.
        ids_reordered (numpy.ndarray): Reordered 1D array matching ids_master.
    """
    # Create a mapping of ids to rows in X
    id_to_indices = {id_: np.where(ids == id_)[0] for id_ in ids_master}

    # Reorder X and ids to match the order in ids_master
    X_reordered = []
    ids_reordered = []

    for id_ in ids_master:
        indices = id_to_indices[id_]
        id_to_append = id_
        # Compute the aggregated value for the rows corresponding to the current ID
        if aggregation_method == 'mean':
            X_aggregated = np.mean(X[indices], axis=0)
        elif aggregation_method == 'median':
            X_aggregated = np.median(X[indices], axis=0)
        elif aggregation_method == 'min':
            X_aggregated = np.min(X[indices], axis=0)
        elif aggregation_method == 'max':
            X_aggregated = np.max(X[indices], axis=0)
        elif aggregation_method == 'none':
            X_aggregated = X[indices]
            id_to_append = np.repeat(id_, len(indices))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Append aggregated values to the reordered list
        X_reordered.append(X_aggregated)
        ids_reordered.append(id_to_append)

    # Convert lists to numpy arrays
    if aggregation_method == 'none':
        X_reordered = np.vstack(X_reordered)
        ids_reordered = np.hstack(ids_reordered)
    else:
        X_reordered = np.array(X_reordered)
        ids_reordered = np.array(ids_reordered)

    return X_reordered, ids_reordered
