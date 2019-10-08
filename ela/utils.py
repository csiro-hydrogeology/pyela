
def flip(m, axis):
    """Reverse the order of elements in an array along the given axis. Only use if numpy < 1.12, otherwise use np.flip

    The shape of the array is preserved, but the elements are reordered.
    If using numpy < 1.12, np.flip is not available (stemmed from Anaconda for windows).
    # see https://stackoverflow.com/a/45707308/2752565

    Args:
        m (array_like): Input array.
        axis (None or int, optional): Axis or axes along which to flip over.
            The default, axis=None, will flip over all of the axes of the input array. If axis is negative it counts from the last to the first axis. If axis is a tuple of ints, flipping is performed on all of the axes specified in the tuple.
    Returns:	
        array_like: flipped array. A view of m with the entries of axis reversed. Since a view is returned, this operation is done in constant time.
    """
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

