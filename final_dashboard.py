import streamlit as st
import pandas as pd
import os
import sys
import subprocess

# Ensure this is the first Streamlit command
st.set_page_config(
    page_title="Technical Indicator Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clone repository if it doesn't exist
repo_url = "https://github.com/AmaanRai1/DS-440---Group-2.git"
repo_dir = "DS-440---Group-2"

# Add the repo directory to the system path
sys.path.append(os.path.abspath(repo_dir))

try:
    from final_backend import *  # Adjust this based on what you need from the file
    st.write("Successfully imported `final_backend.py`!")
except ImportError as e:
    st.error(f"Error importing final_backend: {e}")
    st.write("sys.path:", sys.path)


# Sidebar for user input
with st.sidebar:
    st.title('Technical Indicator Dashboard')

    # Stock ticker selection
    stock_ticker_lst = ['DJI', 'GDAXI', 'IBEX']
    selected_stock_ticker = st.selectbox('Select a stock', stock_ticker_lst)

    # Technical indicators selection
    st.markdown("### Select Technical Indicators:")
    selected_technical_indicator = st.radio(
        'Choose a technical indicator',
        ('RSI (Relative Strength Index)', 'MACD (Moving Average Convergence Divergence)', 'TEMA (Triple Exponential Moving Average)')
    )
    
    # Selectbox to decide whether to pair indicator with ML model
    st.markdown("### Pair with Machine Learning Model?")
    pair_option = st.selectbox(
        "Would you like to pair the selected technical indicator with the chosen ML model?",
        ("No", "Yes")
    )

    # Conditional display of ML model selection and hybrid strategy
    if pair_option == "Yes":
        st.markdown("### Select Machine Learning Model:")
        selected_model = st.radio(
            'Choose a model to pair with indicators',
            ('Linear Regression', 'LSTM', 'ANN')
        )
        st.markdown("### Select Hybrid Strategy:")
        selected_strategy = st.radio(
            'Choose a hybrid strategy to run the machine learning',
            ('Emphasize Technical Indicators', 'Emphasize Machine Learning'),
            index=0  # Default to "Emphasize Technical Indicators"
        )
    else:
        selected_model = "Not Using Pair"
        selected_strategy = "Not Using Strategy"

    # Add the "Execute Hybrid Model" Button Below the Sidebar Section
    execute_hybrid = st.button('Execute Hybrid Model', key='execute_hybrid')

# Main logic to handle data loading and analysis
if execute_hybrid:
    try:
        # Step 1: Load and preprocess the data
        df = load_data_from_github(selected_stock_ticker)  # Load the stock data
        train_data_scaled, validation_data_scaled, test_data_scaled, scaler = split_data(df)

        # Initialize variables for hybrid results
        hybrid_results = None

        # Step 2: Handle technical indicators with ML pairing
        if pair_option == "Yes":
            # Step 2a: Generate predictions from the selected ML model
            y_pred = None  # Placeholder for predictions

            if selected_model == 'Linear Regression':
                st.write("Training and generating predictions using Linear Regression...")

                # Debug: Validate data
                train_data_scaled, validation_data_scaled, test_data_scaled, scaler = split_data(df)
                print("train_data_scaled sample:")
                print(train_data_scaled.head())

                # Step 2a-1: Extract features and targets
                x_train = train_data_scaled[['Open', 'High', 'Low']].values
                y_train = train_data_scaled['Close'].values
                x_test = test_data_scaled[['Open', 'High', 'Low']].values

                # Debug: Check data types and shapes
                print(f"x_train type: {type(x_train)}, shape: {x_train.shape}")
                print(f"y_train type: {type(y_train)}, shape: {y_train.shape}")
                print(f"x_test type: {type(x_test)}, shape: {x_test.shape}")

                # Step 2a-2: Train the Linear Regression model
                regression_model = LinearRegression()
                regression_model.fit(x_train, y_train)

                # Debug: Fit the scaler
                close_scaler = MinMaxScaler()
                close_scaler.fit(train_data_scaled[['Close']])

                # Step 2a-3: Generate predictions
                y_pred = get_y_pred_lr(regression_model, x_test, close_scaler, time_window)


            elif selected_model == "LSTM":
                y_pred = get_y_pred_lstm(lstm_model, x_test, test_data, close_scaler, time_window)

            elif selected_model == "ANN":
                y_pred = get_y_pred_ann(ann_model, x_test, test_data, close_scaler, time_window)

            else:
                st.error("No valid model selected for predictions!")
                raise ValueError("Invalid ML model selected")

            # Step 2b: Apply hybrid strategy based on emphasis
            if selected_strategy == "Emphasize Technical Indicators":
                st.write(f"Running hybrid strategy emphasizing technical indicators for {selected_stock_ticker}...")
                if selected_technical_indicator == 'TEMA (Triple Exponential Moving Average)':
                    hybrid_results = htema_strategy(
                        test_data, 'Close', best_tema_params[0], best_tema_params[1], best_tema_params[2], y_pred
                    )
                elif selected_technical_indicator == 'MACD (Moving Average Convergence Divergence)':
                    hybrid_results = hmacd_strategy(
                        test_data, 'Close', best_macd_params[0], best_macd_params[1], best_macd_params[2], y_pred
                    )
                elif selected_technical_indicator == 'RSI (Relative Strength Index)':
                    hybrid_results = hrsi_strategy(
                        test_data, 'Close', best_rsi_period, y_pred
                    )

            elif selected_strategy == "Emphasize Machine Learning":
                st.write(f"Running hybrid strategy emphasizing machine learning for {selected_stock_ticker}...")
                if selected_technical_indicator == 'TEMA (Triple Exponential Moving Average)':
                    hybrid_results = htema_strategy_2(
                        test_data, 'Close', best_tema_params[0], best_tema_params[1], best_tema_params[2], y_pred
                    )
                elif selected_technical_indicator == 'MACD (Moving Average Convergence Divergence)':
                    hybrid_results = hmacd_strategy_2(
                        test_data, 'Close', best_macd_params[0], best_macd_params[1], best_macd_params[2], y_pred
                    )
                elif selected_technical_indicator == 'RSI (Relative Strength Index)':
                    hybrid_results = hrsi_strategy_2(
                        test_data, 'Close', best_rsi_period, y_pred
                    )

            # Step 2c: Display hybrid results
            st.write(f"Hybrid {selected_technical_indicator} Results:")
            st.dataframe(hybrid_results.head())

            # Step 2d: Calculate and display metrics
            metrics = calculate_metrics(hybrid_results)
            st.write(f"Performance Metrics for {selected_technical_indicator} with {selected_model} (Emphasis: {selected_strategy}):")
            st.write(metrics)

            # Step 2e: Plot hybrid strategy performance
            st.markdown("### Hybrid Strategy Performance")
            st.line_chart(hybrid_results[['CStrategy', 'CLogReturns']])

        else:
            # Step 3: Handle standalone technical indicators
            st.write(f"Running {selected_technical_indicator} without ML pairing...")

            if selected_technical_indicator == 'TEMA (Triple Exponential Moving Average)':
                standalone_results = tema_strategy(
                    test_data, 'Close', best_tema_params[0], best_tema_params[1], best_tema_params[2]
                )
            elif selected_technical_indicator == 'MACD (Moving Average Convergence Divergence)':
                standalone_results = macd_strategy(
                    test_data, 'Close', best_macd_params[0], best_macd_params[1], best_macd_params[2]
                )
            elif selected_technical_indicator == 'RSI (Relative Strength Index)':
                standalone_results = rsi_strategy(
                    test_data, 'Close', best_rsi_period
                )

            # Step 3a: Display standalone results
            st.write(f"{selected_technical_indicator} Results (Standalone):")
            st.dataframe(standalone_results.head())

            # Step 3b: Calculate and display metrics
            metrics = calculate_metrics(standalone_results)
            st.write(f"Performance Metrics for {selected_technical_indicator} (Standalone):")
            st.write(metrics)

            # Step 3c: Plot standalone strategy performance
            st.markdown("### Standalone Strategy Performance")
            st.line_chart(standalone_results[['CStrategy', 'CLogReturns']])

    except ValueError as e:
        st.error(f"An error occurred while running the strategy: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


