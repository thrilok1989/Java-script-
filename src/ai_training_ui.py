"""
AI Training UI Component
Streamlit interface for managing AI model training and data collection
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

try:
    from src.training_data_collector import TrainingDataCollector
    from src.model_trainer_pipeline import ModelTrainerPipeline, run_training_pipeline
    TRAINING_MODULES_AVAILABLE = True
except ImportError as e:
    TRAINING_MODULES_AVAILABLE = False
    import_error = str(e)


def render_ai_training_dashboard():
    """
    Render the AI Training & Performance dashboard

    This tab allows users to:
    1. View training data statistics
    2. Record trade outcomes
    3. Retrain the AI model
    4. Monitor model performance
    """

    st.header("🤖 AI Training & Performance Dashboard")
    st.markdown("**Train your AI on YOUR actual trading data for personalized predictions**")

    st.divider()

    # Check if modules are available
    if not TRAINING_MODULES_AVAILABLE:
        st.error("❌ AI Training modules not available")
        st.error(f"Import error: {import_error}")
        st.info("""
        ### 📦 Installation Required

        Install ML dependencies:
        ```bash
        pip install xgboost scikit-learn joblib
        ```

        Then restart the app.
        """)
        return

    # Initialize
    try:
        collector = TrainingDataCollector(data_dir="data")
        trainer = ModelTrainerPipeline(data_dir="data", model_dir="models")
    except Exception as e:
        st.error(f"Failed to initialize training modules: {e}")
        return

    # =====================================================================
    # SECTION 1: TRAINING DATA STATISTICS
    # =====================================================================

    st.subheader("📊 Training Data Statistics")

    col1, col2, col3, col4 = st.columns(4)

    stats = collector.get_stats()

    with col1:
        st.metric(
            "Total Samples",
            stats.get('total_samples', 0),
            help="Number of trade outcomes recorded"
        )

    with col2:
        win_rate = stats.get('win_rate', 0.0)
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            help="Percentage of profitable trades"
        )

    with col3:
        avg_pnl = stats.get('avg_pnl', 0.0)
        st.metric(
            "Avg P&L",
            f"{avg_pnl:+.2f}%",
            delta=f"{avg_pnl:+.2f}%" if avg_pnl != 0 else None,
            help="Average profit/loss per trade"
        )

    with col4:
        total_pnl = stats.get('total_pnl', 0.0)
        st.metric(
            "Total P&L",
            f"{total_pnl:+.2f}%",
            delta=f"{total_pnl:+.2f}%" if total_pnl != 0 else None,
            help="Cumulative profit/loss"
        )

    # Progress bar for minimum samples
    min_samples = 50
    current_samples = stats.get('total_samples', 0)
    progress = min(current_samples / min_samples, 1.0)

    st.progress(progress)
    st.caption(f"Training readiness: {current_samples}/{min_samples} samples")

    if current_samples < min_samples:
        st.warning(f"⚠️ Need {min_samples - current_samples} more samples to train a reliable model")
    else:
        st.success(f"✅ Sufficient data for training! ({current_samples} samples)")

    st.divider()

    # =====================================================================
    # SECTION 2: RECENT PREDICTIONS
    # =====================================================================

    st.subheader("📝 Recent Predictions")

    try:
        log_df = collector.get_prediction_log()

        if len(log_df) > 0:
            # Show only recent predictions
            recent_log = log_df.tail(10).sort_values('timestamp', ascending=False)

            st.dataframe(
                recent_log[[
                    'timestamp', 'prediction_id', 'ml_prediction',
                    'ml_confidence', 'nifty_price_at_prediction',
                    'outcome_recorded'
                ]],
                use_container_width=True,
                hide_index=True
            )

            # Export button
            if st.button("📥 Export All Predictions"):
                csv = log_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key='download-predictions'
                )
        else:
            st.info("ℹ️ No predictions logged yet. Make some predictions in the main app!")

    except Exception as e:
        st.error(f"Error loading prediction log: {e}")

    st.divider()

    # =====================================================================
    # SECTION 3: RECORD TRADE OUTCOME
    # =====================================================================

    st.subheader("📋 Record Trade Outcome")
    st.markdown("After a trade closes, record the outcome to improve AI training")

    # Get unrecorded predictions
    try:
        log_df = collector.get_prediction_log()
        unrecorded = log_df[log_df['outcome_recorded'] == False]

        if len(unrecorded) > 0:
            col1, col2 = st.columns(2)

            with col1:
                # Select prediction
                prediction_options = unrecorded.apply(
                    lambda row: f"{row['prediction_id']} - {row['ml_prediction']} ({row['timestamp']})",
                    axis=1
                ).tolist()

                selected_pred = st.selectbox(
                    "Select Prediction",
                    options=prediction_options,
                    help="Choose the prediction to record outcome for"
                )

                if selected_pred:
                    pred_id = selected_pred.split(' - ')[0]
                    pred_row = unrecorded[unrecorded['prediction_id'] == pred_id].iloc[0]

                    st.info(f"""
                    **Prediction Details:**
                    - Prediction: {pred_row['ml_prediction']}
                    - Confidence: {pred_row['ml_confidence']:.1f}%
                    - Entry Price: ₹{pred_row['nifty_price_at_prediction']:.2f}
                    - Time: {pred_row['timestamp']}
                    """)

            with col2:
                st.markdown("### Actual Outcome")

                # Outcome inputs
                actual_direction = st.selectbox(
                    "Actual Market Direction",
                    options=["BUY (Price went up)", "HOLD (Stayed flat)", "SELL (Price went down)"],
                    help="What actually happened?"
                )

                was_profitable = st.checkbox("Trade was profitable", value=False)

                pnl_percent = st.number_input(
                    "P&L %",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    help="Actual profit/loss percentage"
                )

                exit_price = st.number_input(
                    "Exit Price",
                    value=float(pred_row['nifty_price_at_prediction']),
                    step=1.0,
                    format="%.2f"
                )

            # Submit button
            if st.button("💾 Save Outcome", type="primary"):
                # Map direction
                direction_map = {
                    "BUY (Price went up)": 2,
                    "HOLD (Stayed flat)": 1,
                    "SELL (Price went down)": 0
                }

                outcome = {
                    'actual_direction': direction_map[actual_direction],
                    'profitable': was_profitable,
                    'pnl_percent': pnl_percent,
                    'exit_price': exit_price,
                    'actual_move_1h': 0.0,  # Could calculate from exit price
                    'actual_move_1d': 0.0   # Could calculate from exit price
                }

                # Note: We need to get feature values from somewhere
                # For now, create a placeholder - in production, store features with prediction
                feature_values = {}

                success = collector.record_outcome(
                    prediction_id=pred_id,
                    feature_values=feature_values,
                    outcome=outcome
                )

                if success:
                    st.success("✅ Outcome recorded successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Failed to record outcome")

        else:
            st.info("ℹ️ No unrecorded predictions. All predictions have outcomes!")

    except Exception as e:
        st.error(f"Error in outcome recording: {e}")

    st.divider()

    # =====================================================================
    # SECTION 4: MODEL TRAINING
    # =====================================================================

    st.subheader("🤖 Train AI Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Training Your AI

        The AI model learns from YOUR actual trading outcomes:
        - **Step 1**: Make predictions using the app
        - **Step 2**: Record actual outcomes (above)
        - **Step 3**: When you have 50+ outcomes, retrain the model
        - **Step 4**: AI gets smarter and personalizes to YOUR style!

        **Model improves with:**
        - More training samples (100+ recommended)
        - Diverse market conditions
        - Accurate outcome recording
        - Regular retraining (weekly/monthly)
        """)

    with col2:
        # Check if model exists
        latest_model = os.path.join("models", "latest_model.pkl")
        model_exists = os.path.exists(latest_model)

        if model_exists:
            st.success("✅ Trained Model Found")
            st.caption("Using your personalized model")
        else:
            st.warning("⚠️ No Trained Model")
            st.caption("Using simulated data")

        # Model info
        try:
            if model_exists:
                import joblib
                import json

                metadata_path = os.path.join("models", "latest_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    st.metric("Features", metadata.get('num_features', 'Unknown'))
                    st.caption(f"Trained: {metadata.get('timestamp', 'Unknown')}")

        except Exception as e:
            st.caption(f"Could not load metadata: {e}")

    # Training controls
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        enable_tuning = st.checkbox(
            "Enable Hyperparameter Tuning",
            value=False,
            help="More accurate but slower (5-10 minutes)"
        )

    with col2:
        min_samples_input = st.number_input(
            "Min Samples Required",
            min_value=10,
            max_value=200,
            value=50,
            help="Minimum samples needed to train"
        )

    with col3:
        st.markdown("")  # Spacer

    # Train button
    if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
        if current_samples < min_samples_input:
            st.error(f"❌ Need at least {min_samples_input} samples. You have {current_samples}.")
        else:
            with st.spinner("🤖 Training model... This may take a few minutes..."):
                try:
                    # Run training pipeline
                    results = run_training_pipeline(
                        data_dir="data",
                        model_dir="models",
                        hyperparameter_tuning=enable_tuning,
                        min_samples=min_samples_input
                    )

                    if results:
                        st.success("✅ Model training complete!")

                        # Display results
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Train Accuracy", f"{results['train_accuracy']:.2%}")

                        with col2:
                            st.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")

                        with col3:
                            st.metric("CV Score", f"{results['cv_mean']:.2%}")

                        # Feature importance
                        st.markdown("### 📈 Top 10 Important Features")

                        importance = results['feature_importance']
                        top_features = list(importance.items())[:10]

                        if top_features:
                            df_importance = pd.DataFrame(top_features, columns=['Feature', 'Importance'])

                            fig = px.bar(
                                df_importance,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Feature Importance'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        st.balloons()
                    else:
                        st.error("❌ Training failed")

                except Exception as e:
                    st.error(f"❌ Training error: {e}")

    st.divider()

    # =====================================================================
    # SECTION 5: PERFORMANCE CHARTS
    # =====================================================================

    st.subheader("📈 Performance Analysis")

    try:
        training_df = collector.get_training_data()

        if len(training_df) > 0:
            tab1, tab2, tab3 = st.tabs(["Win Rate Over Time", "P&L Distribution", "Direction Accuracy"])

            with tab1:
                # Calculate cumulative win rate
                training_df['cumulative_wins'] = training_df['profitable'].cumsum()
                training_df['cumulative_trades'] = range(1, len(training_df) + 1)
                training_df['cumulative_win_rate'] = (
                    training_df['cumulative_wins'] / training_df['cumulative_trades'] * 100
                )

                fig = px.line(
                    training_df,
                    x='cumulative_trades',
                    y='cumulative_win_rate',
                    title='Win Rate Over Time',
                    labels={'cumulative_trades': 'Number of Trades', 'cumulative_win_rate': 'Win Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = px.histogram(
                    training_df,
                    x='pnl_percent',
                    title='P&L Distribution',
                    nbins=20,
                    labels={'pnl_percent': 'P&L (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                # Direction accuracy
                if 'actual_direction' in training_df.columns:
                    direction_counts = training_df['actual_direction'].value_counts()
                    direction_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                    direction_labels = [direction_map.get(k, 'Unknown') for k in direction_counts.index]

                    fig = px.pie(
                        values=direction_counts.values,
                        names=direction_labels,
                        title='Actual Market Direction Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("ℹ️ No training data yet. Record some trade outcomes to see charts!")

    except Exception as e:
        st.error(f"Error creating charts: {e}")

    st.divider()

    # =====================================================================
    # SECTION 6: EXPORT & UTILITIES
    # =====================================================================

    st.subheader("🛠️ Utilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Export Training Data"):
            try:
                output_file = collector.export_for_analysis()
                st.success(f"✅ Exported to {output_file}")

                with open(output_file, 'r') as f:
                    csv_data = f.read()

                st.download_button(
                    "Download CSV",
                    csv_data,
                    os.path.basename(output_file),
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

    with col2:
        if st.button("📊 View Training Stats"):
            st.json(stats)

    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("💡 Tip: Retrain weekly for best results. More data = Better AI!")


# Standalone test
if __name__ == "__main__":
    render_ai_training_dashboard()
