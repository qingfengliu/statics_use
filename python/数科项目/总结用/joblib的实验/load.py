from sklearn.metrics import accuracy_score
import joblib





my_model_loaded= joblib.load("my_model.pkl")
x_test= joblib.load("x_test.pkl")
y_test= joblib.load("y_test.pkl")
y_pred = my_model_loaded.predict(x_test)  #在训练集上预测

print('测试集准确率:',accuracy_score(y_test, y_pred))