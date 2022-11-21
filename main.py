import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, silhouette_score
import pickle

plot_colors = px.colors.sequential.YlOrRd[::-2]

st.set_page_config(layout="wide")
st.title("Auto Machine Learning Web Application")

with st.sidebar:
    st.title("Automated Machine Learning Pipeline Web Application")
    st.write("1. Upload your dataset.")
    st.write("2. View a summary of your dataset.")
    st.write("3. Visualize correlation matrix and values distribution.")
    st.write("4. Clean your data.")
    st.write("5. Apply data preprocessing techniques.")
    st.write("6. Train regression/classification models.")
    st.write("7. Evaluate your trained models.")
    st.write("8. Download your models.")
    st.write("Made by [Youssef CHAFIQUI](https://www.ychafiqui.com)")
    st.write("Code on [Github](https://www.github.com/ychafiqui/automl_webapp)")

df = None
if 'df' not in st.session_state:
    st.session_state.df = df
else:
    df = st.session_state.df

st.session_state.eval = False

with st.expander("Upload your data", expanded=True):
    file = st.file_uploader("Upload a dataset", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write(df)
        st.write("Dataset shape:", df.shape)

with st.expander("Data Summary"):
    if df is not None:
        st.write(df.describe())
        st.write("Dataset shape:", df.shape)
        st.write("Number of Nan values across columns:", df.isna().sum())
        st.write("Total number of Nan values:", df.isna().sum().sum())

with st.expander("Data Visualization"):
    if df is not None:
        st.subheader("Correlation Heatmap")
        fig = px.imshow(df.corr(numeric_only=True), width=980)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig)

        st.subheader("Value Counts")
        all_cols_less_40 = [col for col in df.columns if df[col].nunique() < 40]
        cols_to_show = st.multiselect("Select columns to show", all_cols_less_40, default=all_cols_less_40[0])
        for col in cols_to_show:
            st.write(col)
            st.bar_chart(df[col].value_counts())

        all_cols_more_40 = [col for col in df.columns if df[col].nunique() >= 40]
        if len(all_cols_more_40) > 1:
            st.subheader("Scatter plot between two columns")
            x_col = st.selectbox("Select x column", all_cols_more_40, index=0)
            y_col = st.selectbox("Select y column", all_cols_more_40, index=1)
            fig = px.scatter(df, x=x_col, y=y_col, width=980, color_discrete_sequence=plot_colors[1:])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig)

with st.expander("Data Cleaning"):
    if df is not None:
        st.write("Select cleaning options")
        drop_na0 = st.checkbox("Drop all rows with Nan values")
        drop_na1 = st.checkbox("Drop all columns with Nan values")
        drop_duplicates = st.checkbox("Remove duplicates")
        drop_colmuns = st.checkbox("Drop specific columns")
        if drop_colmuns:
            columns = st.multiselect("Select columns to drop", df.columns)
        if st.button("Apply data cleaning"):
            if drop_na0:
                df = df.dropna(axis=0)
            if drop_na1:
                df = df.dropna(axis=1)
            if drop_duplicates:
                df = df.drop_duplicates()
            if drop_colmuns:
                df = df.drop(columns, axis=1)
            st.session_state.df = df
            st.write(df)
            st.write("Dataset shape:", df.shape)
            st.write("Total number of Nan values remaining:", df.isna().sum().sum())

with st.expander("Data Preprocessing"):
    if df is not None:
        if st.session_state.df is not None:
            df = st.session_state.df
        st.write("Select preprocessing options")
        balance = st.checkbox("Balance your dataset")
        if balance:
            sample_tech = st.selectbox("Select sampling technique", ["Over Sampling", "Under Sampling", "Combined"])
            target = st.selectbox("Select target column to be balanced", df.columns, index=len(df.columns)-1)
        pca = st.checkbox("Apply PCA")
        if pca:
            n_components = st.number_input("Number of PCA components", min_value=1, max_value=min(df.shape[0], df.shape[1]), value=2)
            target = st.selectbox("Select target column to be excluded from PCA", df.columns, index=len(df.columns)-1)

        if st.button("Apply data preprocessing"):
            if balance:
                X = df.drop(target, axis=1)
                y = df[target]
                if sample_tech == "Over Sampling":
                    X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
                elif sample_tech == "Under Sampling":
                    X_resampled, y_resampled = RandomUnderSampler(random_state=0).fit_resample(X, y)
                elif sample_tech == "Combined":
                    X_resampled, y_resampled = SMOTEENN(random_state=0).fit_resample(X, y)
                df = pd.DataFrame(X_resampled, columns=df.columns[:-1])
                df[target] = y_resampled
                st.bar_chart(df[target].value_counts())
            if pca:
                columns = df.columns
                df = OrdinalEncoder().fit_transform(df)
                df = pd.DataFrame(df, columns=columns)
                try:
                    pca = PCA(n_components=n_components)
                    X = df.drop(target, axis=1)
                    y = df[target]
                    df = pca.fit_transform(X)
                    df = pd.DataFrame(df)
                    df[target] = y
                except Exception as e:
                    if df.isna().sum().sum() > 0:
                        st.error("Your dataset contains NaN values. Please remove them from the Data Cleaning section and try again")
                    else:
                        st.error(e)

            st.session_state.df = df
            st.write(df)
            st.write("Dataset shape:", df.shape)
        

with st.expander("Model Training"):
    if df is not None:
        if st.session_state.df is not None:
            df = st.session_state.df
        columns = df.columns
        df = OrdinalEncoder().fit_transform(df)
        df = pd.DataFrame(df, columns=columns)
        st.write("Dataset to be used for training")
        st.write(df)
        st.write("Dataset shape:", df.shape)
        prob = st.radio("Select problem type", ["Classification", "Regression", "Clustering"])

        if prob != "Clustering":
            target = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
            test_data_size = st.slider("Test data size (%)", 10, 90, 20, 5)
            models_list = ["Logistic/Linear Regression", "Random Forest", "XGBoost", "CatBoost", "SVM", "KNN"]
            models = st.multiselect("Select ML models to train", models_list)
        else:
            models_list = ["K-Means", "Hierarchical Clustering"]
            models = st.multiselect("Select Clustering algorithms", models_list)

        if "Logistic/Linear Regression" in models:
            st.subheader("Logistic/Linear Regression Parameters")
            max_iter = st.number_input("Maximum number of iterations", min_value=100, max_value=10000, value=1000)
        if "Random Forest" in models:
            st.subheader("Random Forest Parameters")
            n_estimators = st.number_input("Number of estimators", key=1, min_value=100, max_value=10000, value=1000)
        if "XGBoost" in models:
            st.subheader("XGBoost Parameters")
            n_estimators = st.number_input("Number of estimators", key=2, min_value=100, max_value=10000, value=1000)
        if "CatBoost" in models:
            st.subheader("CatBoost Parameters")
            n_estimators = st.number_input("Number of estimators", key=3, min_value=100, max_value=10000, value=1000)
        if "SVM" in models:
            st.subheader("SVM Parameters")
            kernel = st.selectbox("Select kernel", ["linear", "poly", "rbf", "sigmoid"])
            c = st.number_input("C parameter value", min_value=0.1, max_value=10.0, value=1.0)
        if "KNN" in models:
            st.subheader("KNN Parameters")
            n_neighbors = st.number_input("Number of neighbors k", min_value=1, max_value=100, value=5)

        if "K-Means" in models:
            st.subheader("K-Means Parameters")
            n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=4, key=1)
        if "Hierarchical Clustering" in models:
            st.subheader("Hierarchical Clustering Parameters")
            n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=4, key=2)
            linkage = st.selectbox("Select linkage", ["ward", "complete", "average", "single"])

        if st.button("Train models"):
            if len(models) == 0:
                st.error("Please select at least one model to train")
            else:
                st.session_state.models = []
                st.session_state.model_names = models
                if prob != "Clustering":
                    X = df.drop(target, axis=1)
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_size/100)
                try:
                    if "Logistic/Linear Regression" in models:
                        with st.spinner("Training Logistic/Linear Regression model..."):
                            lr_model = LogisticRegression(max_iter=max_iter) if prob == "Classification" else LinearRegression()
                            lr_model.fit(X_train, y_train)
                            st.success("Logistic/Linear Regression model training complete!")
                            st.session_state.models.append(lr_model)
                    if "Random Forest" in models:
                        with st.spinner("Training Random Forest model..."):
                            rf_model = RandomForestClassifier(n_estimators=n_estimators) if prob=="Classification" else RandomForestRegressor(n_estimators=n_estimators)
                            rf_model.fit(X_train, y_train)
                            st.success("Random Forest model training complete!")
                            st.session_state.models.append(rf_model)
                    if "XGBoost" in models:
                        with st.spinner("Training XGBoost model..."):
                            xgb_model = XGBClassifier(n_estimators=n_estimators) if prob=="Classification" else XGBRegressor(n_estimators=n_estimators)
                            xgb_model.fit(X_train, y_train)
                            st.success("XGBoost model training complete!")
                            st.session_state.models.append(xgb_model)
                    if "CatBoost" in models:
                        with st.spinner("Training CatBoost model..."):
                            cat_model = CatBoostClassifier(n_estimators=n_estimators, allow_writing_files=False) if prob=="Classification" else CatBoostRegressor(n_estimators=n_estimators, allow_writing_files=False)
                            cat_model.fit(X_train, y_train, verbose=False)
                            st.success("CatBoost model training complete!")
                            st.session_state.models.append(cat_model)
                    if "SVM" in models:
                        with st.spinner("Training SVM model..."):
                            svm_model = SVC(kernel=kernel, C=c) if prob=="Classification" else SVR(kernel=kernel, C=c)
                            svm_model.fit(X_train, y_train)
                            st.success("SVM model training complete!")
                            st.session_state.models.append(svm_model)
                    if "KNN" in models:
                        with st.spinner("Training KNN model..."):
                            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors) if prob=="Classification" else KNeighborsRegressor(n_neighbors=n_neighbors)
                            knn_model.fit(X_train, y_train)
                            st.success("KNN model training complete!")
                            st.session_state.models.append(knn_model)

                    if "K-Means" in models:
                        with st.spinner("Running K-Means algorithm..."):
                            km_model = KMeans(n_clusters=n_clusters)
                            km_model.fit(df)
                            st.success("K-Means clustering complete!")
                            st.session_state.models.append(km_model)
                    if "Hierarchical Clustering" in models:
                        with st.spinner("Running Hierarchical Clustering algorithm..."):
                            hc_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                            hc_model.fit(df)
                            st.success("Hierarchical Clustering complete!")
                            st.session_state.models.append(hc_model)

                    st.session_state.eval = True
                except Exception as e:
                    if df.isna().sum().sum() > 0:
                        st.error("Your dataset contains NaN values. Please remove them from the Data Cleaning section and try again")
                    else:
                        st.error(e)

with st.expander("Model Evaluation", expanded=True):
    if 'models' in st.session_state and len(st.session_state.models) != 0 and st.session_state.eval:
        st.subheader("Evaluation Metrics")
        if prob == "Classification":
            eval_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
            if df[target].nunique() > 2:
                avg = 'micro'
            else:
                avg = 'binary'
            for model, name in zip(st.session_state.models, st.session_state.model_names):
                y_pred = model.predict(X_test)
                row_df = pd.DataFrame({
                    "Model": [name],
                    "Accuracy": [accuracy_score(y_test, y_pred)],
                    "Precision": [precision_score(y_test, y_pred, average=avg)],
                    "Recall": [recall_score(y_test, y_pred, average=avg)],
                    "F1 Score": [f1_score(y_test, y_pred, average=avg)]
                })
                eval_df = pd.concat([eval_df, row_df], ignore_index=True)
            st.table(eval_df)

            fig = px.bar(
                eval_df.set_index("Model"), 
                orientation='h', 
                width=980,
                color_discrete_sequence=plot_colors
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'})
            st.write(fig)

            st.subheader("Confusion Matrix")
            cols = st.columns(len(st.session_state.models))
            for i, col in enumerate(cols):
                col.write(st.session_state.model_names[i])
                y_pred = st.session_state.models[i].predict(X_test)
                col.table(confusion_matrix(y_test, y_pred))

        elif prob == "Regression":
            eval_df1 = pd.DataFrame(columns=["Model", "Mean Absolute Error", "Root Mean Squared Error"])
            eval_df2 = pd.DataFrame(columns=["Model", "R2 Score"])
            for model, name in zip(st.session_state.models, st.session_state.model_names):
                y_pred = model.predict(X_test)
                row_df1 = pd.DataFrame({
                    "Model": [name],
                    "Mean Absolute Error": [mean_absolute_error(y_test, y_pred)],
                    "Root Mean Squared Error": [np.sqrt(mean_squared_error(y_test, y_pred))]
                })
                row_df2 = pd.DataFrame({
                    "Model": [name],
                    "R2 Score": [r2_score(y_test, y_pred)],
                })
                eval_df1 = pd.concat([eval_df1, row_df1], ignore_index=True)
                eval_df2 = pd.concat([eval_df2, row_df2], ignore_index=True)
            st.table(eval_df1)
            fig = px.bar(
                eval_df1.set_index("Model"), 
                orientation='h', 
                width=980,
                color_discrete_sequence=plot_colors
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total descending'})
            st.write(fig)

            st.table(eval_df2)
            fig = px.bar(
                eval_df2.set_index("Model"), 
                orientation='h', 
                width=980,
                color_discrete_sequence=plot_colors
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'})
            st.write(fig)
        
        elif prob == "Clustering":
            eval_df = pd.DataFrame(columns=["Model", "Silhouette Score"])
            for model, name in zip(st.session_state.models, st.session_state.model_names):
                row_df = pd.DataFrame({
                    "Model": [name],
                    "Silhouette Score": [silhouette_score(df, model.labels_)]
                })
                eval_df = pd.concat([eval_df, row_df], ignore_index=True)
            st.table(eval_df)

            fig = px.bar(
                eval_df.set_index("Model"), 
                orientation='h', 
                width=980,
                color_discrete_sequence=plot_colors
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'})
            st.write(fig)


with st.expander("Model Download", expanded=True):
    if 'models' in st.session_state and len(st.session_state.models) != 0:
        st.write("Download your trained models")
        for m, model_name in zip(st.session_state.models, st.session_state.model_names):
            st.download_button(label="Download " + model_name + " model", data=pickle.dumps(m), file_name=model_name+"_model.pkl", mime="application/octet-stream")