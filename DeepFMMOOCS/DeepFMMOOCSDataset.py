import pandas as pd
from sklearn.metrics import log_loss, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model
from keras.optimizers import SGD
if __name__ == "__main__":
    data = pd.read_csv('./DeepFMDummy.csv')
    data.head()
    print(data.head())
    sparse_features = [ 'gender', 'education',  'cluster_label', 'course_category']
    dense_features = ['all#count', 'session#count', 'seek_video#num', 'play_video#num',   
     'pause_video#num', 'stop_video#num', 'load_video#num','age']

    dense_features = ['all#count', 'session#count', 'seek_video#num', 'play_video#num', 
        'pause_video#num', 'stop_video#num', 'load_video#num',   'problem_get#num', 
        'problem_check#num', 'problem_save#num','reset_problem#num', 'problem_check_correct#num',
        'problem_check_incorrect#num', 'create_thread#num', 'create_comment#num', 'delete_thread#num', 
        'delete_comment#num','click_info#num', 'click_courseware#num', 'click_about#num','click_forum#num', 
        'click_progress#num', 'close_courseware#num', 'age']

    target = ['truth']   
    
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=32 )
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    
    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy'], )
    
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=32, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    pyplot.grid(True)
    pyplot.gca().set_ylim(0,1)
    pyplot.show()


    model.summary()

    plot_model(model, to_file='C:/Users/zunai/OneDrive/Desktop/FYP_MOOCS/WritingThesis/submodelDFM.jpg')
    score1 = pred_ans
    f, t , threshold = roc_curve(test[target].values, score1)
    pyplot.title('DeepFM:ROC')
    pyplot.plot(f, t)
    pyplot.plot([0, 1], ls="--")
    pyplot.plot([0, 0], [1, 0] , c=".7"), pyplot.plot([1, 1] , c=".7")
    pyplot.ylabel('True Positive Rate')
    pyplot.xlabel('False Positive Rate')
    pyplot.show()






    pyplot.subplot(211) 
    pyplot.title('Training')
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['binary_crossentropy'])
    pyplot.xlabel("Learning rate")
    pyplot.ylabel('Logloss')
    pyplot.legend()
    pyplot.show()

    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Testing')
    pyplot.plot(history.history['val_loss'])
    pyplot.plot(history.history['val_binary_crossentropy'])
    pyplot.xlabel("Learning rate")
    pyplot.ylabel('Logloss')
    pyplot.legend()
    pyplot.show()









    
