import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

# ⚡ Page Configuration
st.set_page_config(
    page_title="Customer Retention AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Dark Glass Style (Consistent SaaS Design)
st.markdown("""
    <style>
    body { margin: 0; padding: 0; }
    
    .kpi-high {
        background: rgba(214, 40, 40, 0.25);
        border-left: 4px solid #d62828;
        padding: 12px 14px;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 600;
    }
    
    .kpi-medium {
        background: rgba(247, 127, 0, 0.25);
        border-left: 4px solid #f77f00;
        padding: 12px 14px;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 600;
    }
    
    .kpi-low {
        background: rgba(6, 167, 125, 0.25);
        border-left: 4px solid #06a77d;
        padding: 12px 14px;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 600;
    }
    
    .main-summary {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        border: 1px solid rgba(102, 126, 234, 0.5);
        padding: 16px 20px;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .summary-risk { font-size: 32px; font-weight: bold; margin: 4px 0; color: #ffffff; }
    .summary-prob { font-size: 20px; margin: 4px 0; font-weight: 600; color: #ffffff; }
    .summary-loss { font-size: 14px; opacity: 0.95; color: #ffffff; }
    
    .action-box {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.4);
        padding: 12px 14px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        color: #ffffff;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .reason-item {
        background: rgba(102, 126, 234, 0.15);
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
        border-left: 3px solid #667eea;
        font-size: 13px;
        color: #ffffff;
        font-weight: 500;
    }
    
    .signal-box {
        background: rgba(20, 30, 60, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 14px 16px;
        border-radius: 8px;
        font-size: 13px;
        line-height: 1.6;
        color: #ffffff;
    }
    
    .signal-box strong {
        color: #ffffff;
        display: block;
        margin-bottom: 8px;
        font-weight: 700;
        font-size: 14px;
    }
    
    .insight-box {
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.4);
        padding: 16px 18px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.7;
        color: #ffffff;
    }
    
    .insight-box strong {
        color: #ffffff;
        font-weight: 700;
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(6, 167, 125, 0.25) 0%, rgba(6, 167, 125, 0.15) 100%);
        border: 1px solid rgba(6, 167, 125, 0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, rgba(247, 127, 0, 0.25) 0%, rgba(247, 127, 0, 0.15) 100%);
        border: 1px solid rgba(247, 127, 0, 0.4);
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(214, 40, 40, 0.25) 0%, rgba(214, 40, 40, 0.15) 100%);
        border: 1px solid rgba(214, 40, 40, 0.4);
    }
    
    .section-title { 
        margin: 16px 0 12px 0; 
        font-size: 15px; 
        font-weight: bold; 
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Load pipeline
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load('models/churn_pipeline.pkl')
    return pipeline

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading pipeline: {str(e)}")
    st.stop()

# Load data for defaults
@st.cache_resource
def load_data_for_defaults():
    df = pd.read_csv('data/raw/telco.csv')
    return df

df_original = load_data_for_defaults()

# Main header
st.title("Customer Retention AI")
st.caption("AI-powered system to identify at-risk customers and prevent churn in real-time")

# ⚙️ SIDEBAR INPUTS WITH ORGANIZED SECTIONS
with st.sidebar:
    st.header("📋 Customer Profile")
    
    st.subheader(" Make Prediction")
    
    # Customer Basics
    with st.expander("📊 Basics", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        satisfaction = st.slider("Satisfaction (1-5 stars)", 1, 5, 3)
        num_referrals = st.slider("Number of Referrals", 0, 20, 0)
    
    # Billing
    with st.expander("💰 Billing", expanded=True):
        monthly_charge = st.slider("Monthly Charge ($)", 0, 200, 50)
        total_charges = st.slider("Total Charges ($)", 0, 10000, 600)
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
    
    # Contract & Services
    with st.expander("📄 Contract & Services", expanded=True):
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_type = st.selectbox("Internet Type", ["Fiber Optic", "Cable", "DSL", "None"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    predict_btn = st.button(" PREDICT CHURN", use_container_width=True, type="primary")
    
    st.divider()
    
    # Batch CSV upload
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV with customer data", type="csv", help="CSV should have the same columns as training data")
    batch_predict_btn = st.button("📊 PREDICT BATCH", use_container_width=True) if uploaded_file else False

# Main Dashboard
if predict_btn:
    # Build the complete input with 40 columns in the correct order
    input_dict = {
        'Gender': 'Male',
        'Age': 35,
        'Under 30': 'No',
        'Senior Citizen': 'No',
        'Married': 'No',
        'Dependents': 'No',
        'Number of Dependents': 0,
        'Latitude': 40.7128,
        'Longitude': -74.0060,
        'Population': 100000,
        'Quarter': 'Q1',
        'Referred a Friend': 'No',
        'Number of Referrals': num_referrals,
        'Tenure in Months': tenure,
        'Offer': 'None',
        'Phone Service': 'Yes',
        'Avg Monthly Long Distance Charges': 5.0,
        'Multiple Lines': 'No',
        'Internet Service': 'Yes' if internet_type != 'None' else 'No',
        'Internet Type': internet_type,
        'Avg Monthly GB Download': 50.0,
        'Online Security': online_security,
        'Online Backup': 'No',
        'Device Protection Plan': 'No',
        'Premium Tech Support': tech_support,
        'Streaming TV': 'No',
        'Streaming Movies': 'No',
        'Streaming Music': 'No',
        'Unlimited Data': 'No',
        'Contract': contract,
        'Paperless Billing': 'Yes',
        'Payment Method': payment_method,
        'Monthly Charge': float(monthly_charge),
        'Total Charges': float(total_charges),
        'Total Refunds': 0.0,
        'Total Extra Data Charges': 0.0,
        'Total Long Distance Charges': 50.0,
        'Total Revenue': float(total_charges),
        'Satisfaction Score': satisfaction,
        'CLTV': float(monthly_charge * 36)
    }
    
    try:
        input_df = pd.DataFrame([input_dict])
        churn_prob = pipeline.predict_proba(input_df)[0][1]
        
        # ═══════════════════════════════════════════════════════════
        # RISK DETERMINATION (CORRECT LOGIC)
        # ═══════════════════════════════════════════════════════════
        if churn_prob >= 0.7:
            risk_level = "🔴 HIGH RISK — Immediate Action"
            risk_color = "kpi-high"
            risk_class = "risk-high"
        elif churn_prob >= 0.4:
            risk_level = "🟡 MEDIUM RISK — Monitor Closely"
            risk_color = "kpi-medium"
            risk_class = "risk-medium"
        else:
            risk_level = "🟢 LOW RISK — Stable Customer"
            risk_color = "kpi-low"
            risk_class = "risk-low"
        
        # ═══════════════════════════════════════════════════════════
        # TOP ROW: 3 Clean KPIs (NO DUPLICATION)
        # ═══════════════════════════════════════════════════════════
        annual_loss = churn_prob * monthly_charge * 12
        
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        
        with kpi_col1:
            st.markdown(f"""
            <div class="main-summary {risk_class}">
                <div class="summary-risk">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col2:
            st.markdown(f"""
            <div class="main-summary">
                <div style="font-size: 14px; opacity: 0.9;">Churn Probability</div>
                <div class="summary-prob">{churn_prob:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col3:
            st.markdown(f"""
            <div class="main-summary">
                <div style="font-size: 14px; opacity: 0.9;">Revenue at Risk</div>
                <div class="summary-prob">${annual_loss:,.0f}/year</div>
            </div>
            """, unsafe_allow_html=True)
        
        
        # ═══════════════════════════════════════════════════════════
        # RECOMMENDATIONS: Clean & Concise
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">🎯 Recommended Actions</p>', unsafe_allow_html=True)
        
        # Determine main action
        if churn_prob >= 0.7:
            main_action = "⚠️ URGENT — Call within 24 hours"
        elif churn_prob >= 0.4:
            main_action = "⚠️ Monitor — Schedule check-in"
        else:
            main_action = "✅ Stable — Maintain engagement"
        
        st.markdown(f'<div class="action-box"><strong>{main_action}</strong></div>', unsafe_allow_html=True)
        
        # Additional targeted actions
        actions = []
        
        if tenure < 12:
            actions.append("• Strengthen onboarding (new customer in first year)")
        
        if satisfaction <= 2:
            actions.append("• Schedule support call immediately (low satisfaction)")
        elif satisfaction == 3:
            actions.append("• Improve satisfaction with proactive support")
        
        if contract == "Month-to-month":
            actions.append("• Convert to annual plan with discount incentive")
        
        if monthly_charge > 70:
            actions.append("• Offer loyalty discount on high monthly charges")
        
        if online_security == "No" and internet_type != "None":
            actions.append("• Bundle online security for added value")
        
        if tech_support == "No" and internet_type != "None":
            actions.append("• Include tech support in service bundle")
        
        if actions:
            for action in actions:
                st.markdown(f'<div class="reason-item">{action}</div>', unsafe_allow_html=True)
        
        
        # ═══════════════════════════════════════════════════════════
        # WHY THIS PREDICTION: Key Factors Only
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">🔍 Why this prediction?</p>', unsafe_allow_html=True)
        
        reasons = []
        if tenure < 12:
            reasons.append("New customer (short tenure increases risk)")
        if monthly_charge > 70:
            reasons.append("High monthly charges (price sensitivity)")
        if satisfaction <= 3:
            reasons.append(f"Moderate satisfaction ({satisfaction}/5) — opportunity to improve")
        if contract == "Month-to-month":
            reasons.append("Month-to-month contract (easy exit option)")
        if online_security == "No" and internet_type != "None":
            reasons.append("Missing security protection")
        if tech_support == "No" and internet_type != "None":
            reasons.append("No dedicated tech support")
        
        if reasons:
            reason_text = "<br>".join([f"• {r}" for r in reasons])
            st.markdown(f'<div class="insight-box">{reason_text}</div>', unsafe_allow_html=True)
        else:
            st.info("✅ No major churn risk factors detected.")
        
        
        # ═══════════════════════════════════════════════════════════
        # CUSTOMER SNAPSHOT: Clean & Simple
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">📋 Customer Snapshot</p>', unsafe_allow_html=True)
        
        snapshot_col1, snapshot_col2, snapshot_col3 = st.columns(3)
        
        with snapshot_col1:
            st.markdown(f"""
            <div class="signal-box">
            <strong>Engagement</strong><br>
            Tenure: {tenure} months<br>
            Monthly: ${monthly_charge}<br>
            Satisfaction: {satisfaction}/5 ⭐
            </div>
            """, unsafe_allow_html=True)
        
        with snapshot_col2:
            st.markdown(f"""
            <div class="signal-box">
            <strong>Terms & Payment</strong><br>
            Contract: {contract}<br>
            Payment: {payment_method}<br>
            Internet: {internet_type}
            </div>
            """, unsafe_allow_html=True)
        
        with snapshot_col3:
            security_status = "✅" if online_security == "Yes" else "❌"
            support_status = "✅" if tech_support == "Yes" else "❌"
            st.markdown(f"""
            <div class="signal-box">
            <strong>Add-ons & Engagement</strong><br>
            {security_status} Security<br>
            {support_status} Tech Support<br>
            Referrals: {num_referrals}
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Please check the input values and try again.")

elif batch_predict_btn and uploaded_file:
    # ═══════════════════════════════════════════════════════════
    # BATCH PREDICTION WITH ADVANCED ANALYTICS
    # ═══════════════════════════════════════════════════════════
    try:
        df_batch = pd.read_csv(uploaded_file)
        
        # Make predictions
        predictions = pipeline.predict_proba(df_batch)[:, 1]
        df_batch['Churn Probability'] = predictions
        
        # Add risk classification
        df_batch['Risk Level'] = df_batch['Churn Probability'].apply(
            lambda x: "🔴 HIGH" if x >= 0.7 else "🟡 MEDIUM" if x >= 0.4 else "🟢 LOW"
        )
        
        # Add Customer ID (use existing ID column if available, otherwise create from index)
        if 'customerID' in df_batch.columns:
            df_batch['Customer ID'] = df_batch['customerID']
        elif 'Customer ID' not in df_batch.columns:
            df_batch.insert(0, 'Customer ID', range(1, len(df_batch) + 1))
        
        # Add intelligent recommendations
        def get_recommendation(row):
            prob = row['Churn Probability']
            if prob >= 0.7:
                return "🎯 URGENT: Call within 24hrs - Offer discount"
            elif prob >= 0.4:
                return "📞 Schedule check-in - Review satisfaction"
            else:
                return "✅ Maintain - No action needed"
        
        df_batch['Recommendation'] = df_batch.apply(get_recommendation, axis=1)
        
        st.markdown('<p class="section-title">📊 Batch Prediction Analysis</p>', unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════
        # 1. KPI SECTION: Overview Metrics
        # ═══════════════════════════════════════════════════════════
        high_risk = len(df_batch[df_batch['Churn Probability'] >= 0.7])
        med_risk = len(df_batch[(df_batch['Churn Probability'] >= 0.4) & (df_batch['Churn Probability'] < 0.7)])
        low_risk = len(df_batch[df_batch['Churn Probability'] < 0.4])
        
        # Calculate total revenue at risk
        monthly_charge_col = 'Monthly Charge' if 'Monthly Charge' in df_batch.columns else 'MonthlyCharges'
        if monthly_charge_col not in df_batch.columns:
            # Fallback for different column names
            for col in df_batch.columns:
                if 'charge' in col.lower() and 'monthly' in col.lower():
                    monthly_charge_col = col
                    break
        
        total_revenue_at_risk = 0
        if monthly_charge_col in df_batch.columns:
            high_risk_df = df_batch[df_batch['Churn Probability'] >= 0.7]
            total_revenue_at_risk = (high_risk_df[monthly_charge_col].sum() * 12) if len(high_risk_df) > 0 else 0
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
            <div class="main-summary">
                <div style="font-size: 13px; opacity: 0.9;">Total Customers</div>
                <div class="summary-prob">{len(df_batch)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col2:
            st.markdown(f"""
            <div class="main-summary risk-high">
                <div style="font-size: 13px; opacity: 0.9;">High Risk</div>
                <div class="summary-prob">{high_risk}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col3:
            st.markdown(f"""
            <div class="main-summary risk-medium">
                <div style="font-size: 13px; opacity: 0.9;">Medium Risk</div>
                <div class="summary-prob">{med_risk}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col4:
            st.markdown(f"""
            <div class="main-summary">
                <div style="font-size: 13px; opacity: 0.9;">Revenue at Risk</div>
                <div class="summary-prob">${total_revenue_at_risk:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════
        # 2. RISK DISTRIBUTION CHART
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">📈 Risk Distribution</p>', unsafe_allow_html=True)
        
        risk_counts = pd.Series({
            '🔴 High Risk': high_risk,
            '🟡 Medium Risk': med_risk,
            '🟢 Low Risk': low_risk
        })
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#d62828', '#f77f00', '#06a77d']
            bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            ax.set_ylabel('Number of Customers', color='#ffffff', fontsize=11)
            ax.set_title('Customer Risk Distribution', color='#ffffff', fontsize=12, fontweight='bold')
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#0f111d')
            ax.tick_params(colors='#ffffff')
            ax.spines['bottom'].set_color('#ffffff')
            ax.spines['left'].set_color('#ffffff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', color='#ffffff', fontweight='bold')
            
            st.pyplot(fig, use_container_width=True)
        
        with col_chart2:
            risk_pct = pd.Series({
                'High': (high_risk/len(df_batch)*100),
                'Medium': (med_risk/len(df_batch)*100),
                'Low': (low_risk/len(df_batch)*100)
            })
            st.markdown(f"""
            <div class="signal-box">
            <strong>Risk %</strong><br>
            🔴 High: {risk_pct['High']:.1f}%<br>
            🟡 Medium: {risk_pct['Medium']:.1f}%<br>
            🟢 Low: {risk_pct['Low']:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════
        # 3. TOP RISK CUSTOMERS
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">🚨 Top Risk Customers (Immediate Action)</p>', unsafe_allow_html=True)
        
        top_customers = df_batch.nlargest(5, 'Churn Probability')[['Customer ID', 'Churn Probability', 'Risk Level', 'Recommendation']]
        top_customers_display = top_customers.copy()
        top_customers_display['Churn Probability'] = top_customers_display['Churn Probability'].apply(lambda x: f"{x:.0%}")
        
        st.dataframe(top_customers_display, use_container_width=True, hide_index=True)
        
        # ═══════════════════════════════════════════════════════════
        # 4. ACTION SUMMARY
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">🎯 Action Plan</p>', unsafe_allow_html=True)
        
        action_text = []
        if high_risk == 0:
            action_text.append("✅ <strong>No urgent cases detected</strong> — monitor medium risk customers")
        elif high_risk == 1:
            action_text.append(f"⚠️ <strong>1 customer</strong> needs immediate retention call within 24 hours")
        else:
            action_text.append(f"🚨 <strong>{high_risk} customers</strong> require urgent intervention — prioritize by churn probability")
        
        if med_risk > 0:
            action_text.append(f"📋 <strong>Schedule check-ins</strong> with {med_risk} medium-risk customers this week")
        
        # Find common patterns
        if 'Contract' in df_batch.columns:
            high_risk_df = df_batch[df_batch['Churn Probability'] >= 0.7]
            if len(high_risk_df) > 0:
                month_to_month = len(high_risk_df[high_risk_df['Contract'] == 'Month-to-month'])
                if month_to_month > 0:
                    pct = (month_to_month / len(high_risk_df) * 100)
                    action_text.append(f"📅 <strong>{pct:.0f}% of high-risk</strong> customers are on month-to-month — offer annual contract discounts")
        
        if monthly_charge_col in df_batch.columns:
            high_risk_df = df_batch[df_batch['Churn Probability'] >= 0.7]
            if len(high_risk_df) > 0:
                avg_charge = high_risk_df[monthly_charge_col].mean()
                action_text.append(f"💰 <strong>High spenders</strong> (avg ${avg_charge:.0f}/mo) — consider loyalty discounts")
        
        for action in action_text:
            st.markdown(f'<div class="action-box">{action}</div>', unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════
        # 5. SEGMENT INSIGHTS
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">🧩 Segment Insights</p>', unsafe_allow_html=True)
        
        insights = []
        
        # Contract insight
        if 'Contract' in df_batch.columns:
            high_risk_df = df_batch[df_batch['Churn Probability'] >= 0.7]
            if len(high_risk_df) > 0:
                m2m_pct = (len(high_risk_df[high_risk_df['Contract'] == 'Month-to-month']) / len(high_risk_df) * 100)
                insights.append(f"<strong>Contract Type:</strong> {m2m_pct:.0f}% of churners use month-to-month (vs annual)")
        
        # Payment method insight
        if 'Payment Method' in df_batch.columns:
            high_risk_df = df_batch[df_batch['Churn Probability'] >= 0.7]
            if len(high_risk_df) > 0:
                echeck = len(high_risk_df[high_risk_df['Payment Method'] == 'Electronic check'])
                if echeck > 0:
                    pct = (echeck / len(high_risk_df) * 100)
                    insights.append(f"<strong>Payment Method:</strong> {pct:.0f}% of high-risk use electronic check")
        
        # Internet type insight
        if 'Internet Type' in df_batch.columns:
            high_risk_df = df_batch[df_batch['Churn Probability'] >= 0.7]
            if len(high_risk_df) > 0:
                fiber = len(high_risk_df[high_risk_df['Internet Type'] == 'Fiber Optic'])
                if fiber > 0:
                    pct = (fiber / len(high_risk_df) * 100)
                    insights.append(f"<strong>Internet Type:</strong> {pct:.0f}% of high-risk have Fiber Optic service")
        
        if insights:
            insights_html = "<br>".join([f"• {ins}" for ins in insights])
            st.markdown(f'<div class="insight-box">{insights_html}</div>', unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════
        # 6. FULL RESULTS TABLE & DOWNLOAD
        # ═══════════════════════════════════════════════════════════
        st.markdown('<p class="section-title">📋 Full Predictions</p>', unsafe_allow_html=True)
        
        st.dataframe(df_batch[['Customer ID', 'Churn Probability', 'Risk Level', 'Recommendation']], use_container_width=True)
        
        # Download button - clean version without emojis in CSV
        df_export = df_batch[['Customer ID', 'Churn Probability', 'Risk Level', 'Recommendation']].copy()
        df_export['Risk Level'] = df_export['Risk Level'].str.replace('🔴 ', '').str.replace('🟡 ', '').str.replace('🟢 ', '')
        df_export['Recommendation'] = df_export['Recommendation'].str.replace('🎯 ', '').str.replace('📞 ', '').str.replace('✅ ', '')
        csv = df_export.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv;charset=utf-8"
        )
        
    except Exception as e:
        st.error(f"❌ Batch Prediction Error: {str(e)}")
        st.info("Ensure CSV has the same columns as the training data.")

else:
    # Welcome message on first load
    st.info("""
     **Welcome to Churn AI Prediction Dashboard**
    
    **Two modes available:**
    1. **Individual Prediction** - Enter customer details in sidebar & click "Predict Churn"
    2. **Batch Prediction** - Upload CSV file for group analysis
    
    Get instant churn predictions, risk assessment, and AI-driven retention actions.
    """)


