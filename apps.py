import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from PIL import Image
from babel.numbers import format_currency
import pickle
import joblib
import numpy as np


data = pd.read_csv('dataset/telco_data_clustered.csv')

st.set_page_config(layout="wide")

image = Image.open('image/TELCO INSIGHT HUB.png')

col1, col2, col3 = st.columns([1, 11, 1])

with col1:
    st.image(image, use_column_width=True)

with col2:
    st.title('Telco Insight Hub')

with col3:
    st.write('')
    st.write('')
    st.button("Logout")


st.write('')

selected = option_menu(
    menu_title=None,
    options=['Home', 'Customer Segmentation','Customer Churn Predictor', 'Smart Product Recommender'],
    icons=['house-door', 'people-fill', 'motherboard', 'file-earmark-bar-graph'],
    orientation='horizontal',
)

if selected == 'Home':

    cluster_names = {
    0: "💲 Cost-Conscious Regulars",
    1: "♛ Loyal High Value",
    2: "☯ Basic Users",
    3: "☻ New Enthusiasts"
    }

    data['Cluster Label'] = data['Cluster Label'].map(cluster_names)

    col1, col2 = st.columns(2)

    with col1:
        location = st.multiselect('Filter Location', options=data['Location'].unique())

        if location:
            data = data[data['Location'].isin(location)]

    with col2:
        cust_seg = st.multiselect('Filter Customer Segment', options=data['Cluster Label'].unique())

        if cust_seg:
            data = data[data['Cluster Label'].isin(cust_seg)]


    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customers 👥", value=data['Customer ID'].nunique())
    with col2:
        st.metric(label="Churn Rate 📉", value=f"{data['Churn Label'].value_counts(normalize=True)['Yes']:.2%}")
    with col3:
        st.metric(label="Average Tenure 📅", value=f"{data['Tenure Months'].mean():.2f} Months")
    with col4:
        average_cltv = data['CLTV (Predicted Thou. IDR)'].mean() * 1000  # Convert to IDR
        formatted_cltv = format_currency(average_cltv, 'IDR ', locale='en_US')
        st.metric(label="Average CLTV 💰", value=formatted_cltv)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.header('Transaction Tools')

    col5, col6, col7 = st.columns(3)

    colors = {
        'Mid End': '#fb8d03',
        'High End': '#fe1d1d',
        'Low End': '#ffc900',
        'Pulsa': '#fe1d1d',
        'Digital Wallet': '#fb5c0f',
        'Debit':'#fb8d03',
        'Credit':'#ffc900'
    }

    with col5:
        device_class_counts = data['Device Class'].value_counts()
        fig_device_class = px.pie(device_class_counts, values=device_class_counts.values, names=device_class_counts.index,
                                hole=0.5, color=device_class_counts.index, color_discrete_map=colors)
        fig_device_class.update_traces(textinfo='percent+label')
        fig_device_class.update_layout(title='Device Class', showlegend=False, width=400, height=400)
        st.plotly_chart(fig_device_class)

    with col6:
        payment_method_counts = data['Payment Method'].value_counts()
        fig_payment_method = px.pie(payment_method_counts, values=payment_method_counts.values,
                                    names=payment_method_counts.index, hole=0.5, color=payment_method_counts.index,
                                    color_discrete_map=colors)
        fig_payment_method.update_traces(textinfo='percent+label')
        fig_payment_method.update_layout(title='Payment Method', showlegend=False, width=400, height=400)
        st.plotly_chart(fig_payment_method)

    data['Internet Service'] = data.apply(lambda row: 'Not Available' if row['Games Product'] == 'No internet service' else 'Available', axis=1)

    with col7:
        internet_service_counts = data['Internet Service'].value_counts()
        fig_internet_service = px.pie(internet_service_counts, values=internet_service_counts.values,
                                    names=internet_service_counts.index, hole=0.5,
                                    color=internet_service_counts.index, color_discrete_map={'Available': '#fe1d1d', 'Not Available': 'grey'})
        fig_internet_service.update_traces(textinfo='percent+label')
        fig_internet_service.update_layout(title='Internet Service', showlegend=False, width=400, height=400)
        st.plotly_chart(fig_internet_service)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.header('Support Services')

    col8, col9 = st.columns(2)

    with col8:
        call_center_counts = data['Call Center'].value_counts()
        call_center_counts = call_center_counts.reindex(index=['Yes', 'No'])
        fig_call_center = px.bar(call_center_counts, x=call_center_counts.index, y=call_center_counts.values,
                                color=call_center_counts.index, color_discrete_map={'Yes': '#fe1d1d', 'No': '#fb5c0f'})
        fig_call_center.update_layout(title='Call Center', xaxis_title='Call Center', yaxis_title='Count', showlegend=False, width=600, height=400)
        st.plotly_chart(fig_call_center)


    with col9:
        call_center_counts = data['Use MyApp'].value_counts()
        call_center_counts = call_center_counts.reindex(index=['Yes', 'No', 'No internet service'])
        fig_call_center = px.bar(call_center_counts, x=call_center_counts.index, y=call_center_counts.values,
                                color=call_center_counts.index, color_discrete_map={'Yes': '#fe1d1d', 'No': '#fb5c0f', 'No internet service':'grey'})
        fig_call_center.update_layout(title='Use MyApp', xaxis_title='Use MyApp', yaxis_title='Count', showlegend=False, width=600, height=400)
        st.plotly_chart(fig_call_center)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.header('Product Usage')

    col10, col11, col12, col13 = st.columns(4)

    with col10:
        games_product_counts = data['Games Product'].value_counts()
        games_product_counts = games_product_counts.reindex(index=['Yes', 'No', 'No internet service'])
        fig_games_product = px.bar(games_product_counts, x=games_product_counts.index, y=games_product_counts.values,
                                color=games_product_counts.index, color_discrete_map={'Yes': '#fe1d1d', 'No': '#fb5c0f', 'No internet service': 'grey'})
        fig_games_product.update_layout(title='Games Product', xaxis_title='Games Product', yaxis_title='Count', showlegend=False, width=300, height=300)
        st.plotly_chart(fig_games_product)

    with col11:
        games_product_counts = data['Music Product'].value_counts()
        games_product_counts = games_product_counts.reindex(index=['Yes', 'No', 'No internet service'])
        fig_games_product = px.bar(games_product_counts, x=games_product_counts.index, y=games_product_counts.values,
                                color=games_product_counts.index, color_discrete_map={'Yes': '#fe1d1d', 'No': '#fb5c0f', 'No internet service': 'grey'})
        fig_games_product.update_layout(title='Music Product', xaxis_title='Music Product', yaxis_title='Count', showlegend=False, width=300, height=300)
        st.plotly_chart(fig_games_product)

    with col12:
        games_product_counts = data['Education Product'].value_counts()
        games_product_counts = games_product_counts.reindex(index=['Yes', 'No', 'No internet service'])
        fig_games_product = px.bar(games_product_counts, x=games_product_counts.index, y=games_product_counts.values,
                                color=games_product_counts.index, color_discrete_map={'Yes': '#fe1d1d', 'No': '#fb5c0f', 'No internet service': 'grey'})
        fig_games_product.update_layout(title='Education Product', xaxis_title='Education Product', yaxis_title='Count', showlegend=False, width=300, height=300)
        st.plotly_chart(fig_games_product)

    with col13:
        games_product_counts = data['Video Product'].value_counts()
        games_product_counts = games_product_counts.reindex(index=['Yes', 'No', 'No internet service'])
        fig_games_product = px.bar(games_product_counts, x=games_product_counts.index, y=games_product_counts.values,
                                color=games_product_counts.index, color_discrete_map={'Yes': '#fe1d1d', 'No': '#fb5c0f', 'No internet service': 'grey'})
        fig_games_product.update_layout(title='Video Product', xaxis_title='Video Product', yaxis_title='Count', showlegend=False, width=300, height=300)
        st.plotly_chart(fig_games_product)

if selected == 'Customer Segmentation':

    st.write('')

    col14, col15 = st.columns(2)

    st.write('')

    with col14:
        total_customers = len(data)
        cluster_percentages = data['Cluster Label'].value_counts(normalize=True).reset_index()
        cluster_percentages.columns = ['Cluster Label', 'Percentage']
        cluster_percentages['Percentage'] = cluster_percentages['Percentage'] * 100 
        cluster_percentages['Percentage'] = cluster_percentages['Percentage'].round(2)

        cluster_names = {
            0: "Cost-Conscious Regulars",
            1: "Loyal High Value",
            2: "Basic Users",
            3: "New Enthusiasts"
        }

        cluster_percentages['Cluster Label'] = cluster_percentages['Cluster Label'].map(cluster_names)

        custom_colors = ['#fe1d1d', '#d10b0b', '#fb5c0f', '#fb8d03', '#ffc900']

        fig = px.treemap(cluster_percentages, 
                        path=['Cluster Label'], 
                        values='Percentage',
                        title='Customer Segment Treemap',
                        color_discrete_sequence=custom_colors)

        fig.update_traces(textinfo="label+percent entry")

        fig.update_layout(
            title='Customer Segment Treemap', title_x=0.32,
            margin=dict(t=37, l=25, r=35, b=25), height=530
            )

        st.plotly_chart(fig, use_container_width=True)

    with col15:
        st.write('')
        st.write('')

        cluster_characteristics = {
            "💲 Cost-Conscious Regulars": {
                "Avg. Tenure": "25.37 Months",
                "Avg. Monthly Spending": "69,120 IDR",
                "Usage Pattern": "Regular but Cost-Conscious",
                "Service Packages": "Basic Services",
                "Price Sensitivity": "Moderate Response",
                "Device Preference": "Mostly Mid-End Devices",
                "MyApp Usage": "Mostly Not Using MyApp",
                "Internet Service": "Yes"
            },
            "♛ Loyal High Value": {
                "Avg. Tenure": "54.23 Months",
                "Avg. Monthly Spending": "119,370 IDR",
                "Usage Pattern": "Loyal and Valuable Customers",
                "Service Packages": "Premium services",
                "Price Sensitivity": "Low Response",
                "Device Preference": "Mostly High-End Devices",
                "MyApp Usage": "Mostly Using MyApp",
                "Internet Service": "Yes"
            },
            "☯ Basic Users": {
                "Avg. Tenure": "30.55 Months",
                "Avg. Monthly Spending": "27,400 IDR",
                "Usage Pattern": "Using Only Essential Services",
                "Service Packages": "Basic & Essential Services",
                "Price Sensitivity": "High Response",
                "Device Preference": "All Low-End Devices",
                "MyApp Usage": "Not Using MyApp",
                "Internet Service": "No"
            },
            "☻ New Enthusiasts": {
                "Avg. Tenure": "18.76 Months",
                "Avg. Monthly Spending": "106,680 IDR",
                "Usage Pattern": "Trying Out Various Services & Enthusiastic About Offerings",
                "Service Packages": "Exploring Different Services",
                "Price Sensitivity": "Moderate Response",
                "Device Preference": "Mostly High-End Devices",
                "MyApp Usage": "Using MyApp",
                "Internet Service": "Yes"
            }
        }

        selected_cluster = st.selectbox("Select Customer Segment", list(cluster_characteristics.keys()))

        if selected_cluster:
            st.header(selected_cluster)
            cluster_data = cluster_characteristics[selected_cluster]
            cluster_df = pd.DataFrame(cluster_data.items(), columns=["Characteristic", "Description"])
            st.table(cluster_df)

    if selected_cluster == '💲 Cost-Conscious Regulars':
        st.markdown("<h1 style='text-align: center;'>Customer Engagement Strategies</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Cost-Conscious Regulars</h3>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        image_paths = [
            "image/ccr/1.png",
            "image/ccr/2.png",
            "image/ccr/3.png",
            "image/ccr/4.png"
        ]

        with col1:
            st.image(image_paths[0], use_column_width=True)

        with col2:
            st.image(image_paths[1], use_column_width=True)

        with col3:
            st.image(image_paths[2], use_column_width=True)

        with col4:
            st.image(image_paths[3], use_column_width=True)
    
    if selected_cluster == '♛ Loyal High Value':
        st.markdown("<h1 style='text-align: center;'>Customer Engagement Strategies</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Loyal High Value</h3>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        image_paths = [
            "image/lhv/1.png",
            "image/lhv/2.png",
            "image/lhv/3.png",
            "image/lhv/4.png"
        ]

        with col1:
            st.image(image_paths[0], use_column_width=True)

        with col2:
            st.image(image_paths[1], use_column_width=True)

        with col3:
            st.image(image_paths[2], use_column_width=True)

        with col4:
            st.image(image_paths[3], use_column_width=True) 

    if selected_cluster == '☯ Basic Users':
        st.markdown("<h1 style='text-align: center;'>Customer Engagement Strategies</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Basic Users</h3>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        image_paths = [
            "image/basus/1.png",
            "image/basus/2.png",
            "image/basus/3.png",
            "image/basus/4.png"
        ]

        with col1:
            st.image(image_paths[0], use_column_width=True)

        with col2:
            st.image(image_paths[1], use_column_width=True)

        with col3:
            st.image(image_paths[2], use_column_width=True)

        with col4:
            st.image(image_paths[3], use_column_width=True) 
    
    if selected_cluster == '☻ New Enthusiasts':
        st.markdown("<h1 style='text-align: center;'>Customer Engagement Strategies</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>New Enthusiasts</h3>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        image_paths = [
            "image/newen/1.png",
            "image/newen/2.png",
            "image/newen/3.png",
            "image/newen/4.png"
        ]

        with col1:
            st.image(image_paths[0], use_column_width=True)

        with col2:
            st.image(image_paths[1], use_column_width=True)

        with col3:
            st.image(image_paths[2], use_column_width=True)

        with col4:
            st.image(image_paths[3], use_column_width=True) 

def display_product_recommendations(segment, product_rankings, product_usage):
    st.write(f'To elevate your experience, we recommend exploring our diverse range of product(s). Here are smart product recommendations curated just for you.')
    st.subheader(f'Product Recommendations')
    
    col1, col2, col3, col4 = st.columns(4)

    index = 0

    for icon, caption in product_rankings[segment]:
        if caption not in product_usage:
            icon_path = f"image/{icon}"

            if index % 4 == 0:
                with col1:
                    st.write(f'### #1 {caption}')
                    st.image(icon_path)
            elif index % 4 == 1:
                with col2:
                    st.write(f'### #2 {caption}')
                    st.image(icon_path)
            elif index % 4 == 2:
                with col3:
                    st.write(f'### #3 {caption}')
                    st.image(icon_path)
            else:
                with col4:
                    st.write(f'### #4 {caption}')
                    st.image(icon_path)

            index += 1

if selected == 'Smart Product Recommender':
    st.header("Input Customer Data")

    with st.form(key='user_input_form'):
        tenure_months = st.number_input("Tenure Months", min_value=0)
        location = st.selectbox("Location", ["Jakarta", "Bandung"])
        device_class = st.selectbox("Device Class", ["High End", "Mid End", "Low End"])
        product_usage = st.multiselect("Product Usage", ["Games", "Music", "Education", "Video"])
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            submit_button = st.form_submit_button(label='Generate Smart Product Recommender')
        with col2:
            reset_button = st.form_submit_button(label='Reset')

    with open('models/rec_sys_telco.sav', 'rb') as model_file:
        classification_model = pickle.load(model_file)

    segment_mapping = {
        0: "💲 Cost-Conscious Regulars",
        1: "♛ Loyal High Value",
        2: "☯ Basic Users",
        3: "☻ New Enthusiasts"
    }

    product_rankings = {
        "💲 Cost-Conscious Regulars": [('games-icon.png', 'Games'), ('music-icon.png', 'Music'), ('edu-icon.png', 'Education'), ('video-icon.png', 'Video')],
        "♛ Loyal High Value": [('video-icon.png', 'Video'), ('edu-icon.png', 'Education'), ('music-icon.png', 'Music'), ('games-icon.png', 'Games')],
        "☯ Basic Users": [('video-icon.png', 'Video'), ('music-icon.png', 'Music'), ('edu-icon.png', 'Education'), ('games-icon.png', 'Games')],
        "☻ New Enthusiasts": [('video-icon.png', 'Video'), ('edu-icon.png', 'Education'), ('music-icon.png', 'Music'), ('games-icon.png', 'Games')],
    }

    if submit_button:
        user_data_dict = {"Tenure Months": [tenure_months], "Location": [location], "Device Class": [device_class]}
        user_data = pd.DataFrame(user_data_dict)
        predictions = classification_model.predict(user_data)
        predicted_segment = segment_mapping.get(predictions[0], "Unknown Segment")
        st.markdown(f'<h3>Hello, <strong style="color:#fe1d1d">{predicted_segment}</strong></h3>', unsafe_allow_html=True)

        display_product_recommendations(predicted_segment, product_rankings, product_usage)

    if reset_button:
        st.experimental_rerun()

if selected == 'Customer Churn Predictor':

    st.header("Input Customer Data")

    with st.form(key='user_input_form'):

        col1, col2, col3 = st.columns([5, 5, 3])

        with col1:
            tenure_months = st.number_input("Tenure Months", min_value=0)
            location = st.selectbox("Location", ["Jakarta", "Bandung"])
            device_class = st.selectbox("Device Class", ["High End", "Mid End", "Low End"])
            payment_method = st.selectbox("Payment Method", ['Digital Wallet', 'Pulsa', 'Debit', 'Credit'])
            monthly_purchase = st.number_input("Monthly Purchase (Thou. IDR)", min_value=0)
            monthly_purchase /= 1000

        with col2:
            cltv_predicted = st.number_input("CLTV (Thou. IDR)", min_value=0)
            cltv_predicted /= 1000
            games_product = st.selectbox("Games Product", ["Yes", "No", 'No internet service'])
            music_product = st.selectbox("Music Product", ["Yes", "No", 'No internet service'])
            education_product = st.selectbox("Education Product", ["Yes", "No", 'No internet service'])
            video_product = st.selectbox("Video Product", ["Yes", "No", 'No internet service'])

        with col3:
            call_center = st.selectbox("Call Center", ["Yes", "No"])
            use_myapp = st.selectbox("Use MyApp", ["Yes", "No", 'No internet service'])
            internet_service = st.selectbox("Internet Service", ["Available", "Not Available"])
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            clicked_submit = st.form_submit_button(label='Generate Customer Churn Prediction')
        with col2:
            reset_button = st.form_submit_button(label='Reset')
    
    if clicked_submit:
        
        user_data_dict = {
            "Tenure Months": [tenure_months],
            "Location": [location],
            "Device Class": [device_class],
            "Payment Method": [payment_method],
            "Monthly Purchase (Thou. IDR)": [monthly_purchase],
            "CLTV (Predicted Thou. IDR)": [cltv_predicted],
            "Games Product": [games_product],
            "Music Product": [music_product],
            "Education Product": [education_product],
            "Video Product": [video_product],
            "Call Center": [call_center],
            "Use MyApp": [use_myapp],
            "Internet Service": [internet_service]
        }

        user_data = pd.DataFrame(user_data_dict)

        le_location = joblib.load("models/le_location.joblib")
        le_device_class = joblib.load("models/le_device_class.joblib")
        le_payment_method = joblib.load("models/le_payment_method.joblib")
        le_games_product = joblib.load("models/le_games_product.joblib")
        le_music_product = joblib.load("models/le_music_product.joblib")
        le_education_product = joblib.load("models/le_education_product.joblib")
        le_video_product = joblib.load("models/le_video_product.joblib")
        le_call_center = joblib.load("models/le_call_center.joblib")
        le_use_myapp = joblib.load("models/le_use_myapp.joblib")
        le_internet_service = joblib.load("models/le_internet_service.joblib")
        scaler_Tenure = joblib.load("models/scaler_Tenure Months.joblib")
        scaler_CLTV = joblib.load("models/scaler_CLTV (Predicted Thou. IDR).joblib")
        scaler_Monthly = joblib.load("models/scaler_Monthly Purchase (Thou. IDR).joblib")

        categorical_columns = ['Location', 'Device Class', 'Payment Method', 'Games Product', 'Music Product',
                            'Education Product', 'Video Product', 'Call Center', 'Use MyApp', 'Internet Service']
        numerical_columns = ['Tenure Months', 'Monthly Purchase (Thou. IDR)',
                            'CLTV (Predicted Thou. IDR)']


        def data_preprocessing(data):
            """PPreprocessing data

            Args:
                data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
                
            return:
                Pandas DataFrame: Dataframe that contain all the preprocessed data
            """
            data = data.copy()
            df = pd.DataFrame(columns=['Tenure Months', 'Location', 'Device Class', 'Payment Method', 'Monthly Purchase (Thou. IDR)',
                                        'CLTV (Predicted Thou. IDR)', 'Games Product', 'Music Product',
                                        'Education Product', 'Video Product', 'Call Center', 'Use MyApp', 'Internet Service'])
            
            df['Tenure Months'] = scaler_Tenure.transform(
                np.asanyarray(data['Tenure Months']).reshape(-1, 1)).reshape(1, -1)[0]
            df['CLTV (Predicted Thou. IDR)'] = scaler_CLTV.transform(
                np.asanyarray(data['CLTV (Predicted Thou. IDR)']).reshape(-1, 1)).reshape(1, -1)[0]
            df['Monthly Purchase (Thou. IDR)'] = scaler_Monthly.transform(
                np.asanyarray(data['Monthly Purchase (Thou. IDR)']).reshape(-1, 1)).reshape(1, -1)[0]
            
            df["Location"] = le_location.transform(data["Location"])
            df["Device Class"] = le_device_class.transform(
                data["Device Class"])
            df["Payment Method"] = le_payment_method.transform(
                data["Payment Method"])
            df["Games Product"] = le_games_product.transform(
                data["Games Product"])
            df["Music Product"] = le_music_product.transform(
                data["Music Product"])
            df["Education Product"] = le_education_product.transform(
                data["Education Product"])
            df["Video Product"] = le_video_product.transform(
                data["Video Product"])
            df["Call Center"] = le_call_center.transform(data["Call Center"])
            df["Use MyApp"] = le_use_myapp.transform(data["Use MyApp"])
            df["Internet Service"] = le_internet_service.transform(
                data["Internet Service"])
            
            return df
            
        preprocessed_df = data_preprocessing(user_data)

        model = joblib.load("models/model_xgb.joblib")
        result_target = joblib.load("models/le_churn_label.joblib")


        def prediction(data):
            """Making prediction

            Args:
                data (Pandas DataFrame): Dataframe that contain all the preprocessed data

            Returns:
                numpy.ndarray: Prediction result (Churn or No)
            """
            result = model.predict(data)
            final_result = result_target.inverse_transform(result)
            return final_result

        predictions = prediction(preprocessed_df)

        segment_mapping = {
            'No': ("Customer Not Churning", "green"),
            'Yes': ("Customer Churning ⚠️", "#fe1d1d")
        }

        predicted_segment, color = segment_mapping.get(predictions[0], ("Unknown Segment", "black"))

        st.markdown(f'<h3>Prediction Result: <strong style="color:{color}">{predicted_segment}</strong></h3>', unsafe_allow_html=True)

    if clicked_submit and predicted_segment == 'Customer Churning ⚠️':
        with st.form(key='action_form'):
            if 'selected_option' not in st.session_state:
                st.session_state.selected_option = "Follow up by Whatsapp"

            selected_option = st.selectbox("Select Action", ["Follow up by Whatsapp", "Follow up by Email"], key='action_selectbox', index=0)
            take_action_button = st.form_submit_button("Take Action")

            if take_action_button:
                st.markdown(f"Performing action: {selected_option}")
                st.session_state.selected_option = selected_option
                   
    if reset_button:
        st.experimental_rerun()

footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    margin-top: 0; /* Reduce or remove the margin-top property to close the gap */
}
</style>
<div class="footer">
<p>COPYRIGHT © 2023 DATA CHALLENGE DSW - THE POWERPUFF GIRLS. ALL RIGHTS RESERVED.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)