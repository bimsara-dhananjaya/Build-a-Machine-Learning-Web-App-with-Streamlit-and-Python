import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are Your Mushrooms Healthy?")
    st.sidebar.markdown("Are Your Mushrooms Healthy?")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('C:\\Users\\User\\Downloads\\mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, x_test, y_test):
        if "Confusion Matrix" in metrics_list:
            # For Confusion Matrix
            cm = confusion_matrix(y_test, model.predict(x_test))
            st.subheader("Confusion Matrix")
            st.write(cm)

            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=['Predicted ' + name for name in class_names],
                   yticklabels=['Actual ' + name for name in class_names],
                   title="Confusion Matrix",
                   ylabel='True label',
                   xlabel='Predicted label')

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            # For ROC Curve
            fpr, tpr, thresholds = roc_curve(
                y_test, model.predict_proba(x_test)[:, 1])
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label='ROC Curve')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            # For Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(
                y_test, model.predict_proba(x_test)[:, 1])
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label='Precision-Recall Curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            st.pyplot(fig)

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio(
            "Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix",
                                      "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider(
            "Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix",
                                      "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, x_test, y_test)

    # if classifier == "Random Forest":
    #     st.sidebar.subheader("Model Hyperparameters")
    #     n_estimators = st.sidebar.number_input(
    #         "The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    #     max_depth = st.sidebar.number_input(
    #         "The maximum depth of the tree", 1, 20, step=1, key='max_depth')
    #     bootstrap = st.sidebar.radio(
    #         "Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
    #     metrics = st.sidebar.multiselect(
    #         "What metrics to plot?", ("Confusion Matrix",
    #                                   "ROC Curve", "Precision-Recall Curve")
    #     )

    #     if st.sidebar.button("Classify", key='classify'):
    #         st.subheader("Random Forest Results")
    #         model = RandomForestClassifier(
    #             n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy: ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(
    #             y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(
    #             y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics, model, x_test, y_test)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input(
        "The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    max_depth = st.sidebar.number_input(
        "The maximum depth of the tree", 1, 20, step=1, key='max_depth')
    bootstrap = st.sidebar.radio(
        "Bootstrap samples when building trees", (True, False), key='bootstrap')
    metrics = st.sidebar.multiselect(
        "What metrics to plot?", ("Confusion Matrix",
                                  "ROC Curve", "Precision-Recall Curve")
    )

    if st.sidebar.button("Classify", key='classify'):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Precision: ", precision_score(
            y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(
            y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics, model, x_test, y_test)


if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Mushroom Data Set (Classification)")
    st.write(df)


if __name__ == '__main__':
    main()
