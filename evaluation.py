import numpy as np 

def trainTestEvaluate(model, X, y, costFunc, test_ratio = 0.3):
    data = np.hstack((X, y[:, np.newaxis]))
    np.random.shuffle(data)
    _X = data[:, 0:-1]
    _y = np.ravel(data[:, -1])

    r = int((1-test_ratio) * _X.shape[0])
    train_X = _X[0:r] 
    test_X = _X[r:]
    train_y = _y[0:r]
    test_y = _y[r:]

    model.fit(train_X, train_y)
    return costFunc(model, test_X, test_y)

def crossValidationEvaluate(model, X, y, costFunc, folds=5):
    data = np.hstack((X, y[:, np.newaxis]))
    np.random.shuffle(data)
    _X = data[:, 0:-1]
    _y = np.ravel(data[:, -1])

    scores = []
    r = _X.shape[0] // folds
    for i in range(folds):
        train_mask = np.ones_like(y, dtype=bool)
        if i == folds-1:
            train_mask[i*r:] = 0 
        else:    
            train_mask[i*r : (i+1)*r] = 0 

        train_X = _X[train_mask] 
        test_X = _X[np.logical_not(train_mask)]
        train_y = _y[train_mask]
        test_y = _y[np.logical_not(train_mask)]

        model.fit(train_X, train_y)
        scores.append(costFunc(model, test_X, test_y))
    
    return scores

#Testing
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_boston
    from metrics import R2

    np.set_printoptions(precision=2)

    model = LinearRegression()
    boston = load_boston()
    X = boston.data
    y = boston.target

    #Variance difference
    scoresTT = []
    scoresCV = []
    for i in range(20):
        scoresTT.append(trainTestEvaluate(model, X, y, R2))
        scoresCV.append(np.mean(crossValidationEvaluate(model, X, y, R2)))

    
    print('mean, variance')
    print("Train_test: ", np.mean(scoresTT), np.var(scoresTT))
    print("Cross validation:", np.mean(scoresCV), np.var(scoresCV))
    print("var(TT) / var(CV) = ", np.var(scoresTT) / np.var(scoresCV))