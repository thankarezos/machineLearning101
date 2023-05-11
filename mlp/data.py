import createDataset as cd

def linearSeparated(n, model):
    X = cd.linearSeparated(n)
    return X

def nonlinearSeparatedAngle(n, model):
    X = cd.nonLinearAngle(n)
    return X

def nonlinearSeparatedCenter(n, model):
    X = cd.nonLinearCenter(n)
    return X

def nonlinearSeparatedXOR(n, model):
    X = cd.nonLinearXOR(n)
    return X

def nonlinearSeparated(n, model):
    X = cd.nonLinear(n)

    return X