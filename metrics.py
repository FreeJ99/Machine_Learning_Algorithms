import numpy as np

def MSE(model, X, y):
    y_pred = model.predict(X)
    return np.mean((y_pred - y)**2)

def R2(model, X, y):
    return 1 - MSE(model, X, y) / np.var(y)

def ACC(model, X, y):
    return np.mean(model.predict(X) == y)

#Tests functions
if __name__ == "__main__":
    #Preparation
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.datasets import load_boston

    cModel = LogisticRegression(solver='lbfgs', multi_class='auto')
    iris = load_iris()
    cX = iris.data
    cy = iris.target
    cModel.fit(cX, cy)

    rModel = LinearRegression()
    boston = load_boston()
    rX = boston.data
    ry = boston.target
    rModel.fit(rX, ry)

    #Testing
    from sklearn.metrics import mean_squared_error
    if(MSE(rModel, rX, ry) == mean_squared_error(ry, rModel.predict(rX))):
        print('MSE: VALID')
    else:
        print('MSE: ERROR')


    from sklearn.metrics import r2_score
    if(R2(rModel, rX, ry) == r2_score(ry, rModel.predict(rX))):
        print('R2: VALID')
    else:
        print('R2: ERROR')

    if(ACC(cModel, cX, cy) == cModel.score(cX, cy)):
        print('ACC: VALID')
    else:
        print('ACC: ERROR')
