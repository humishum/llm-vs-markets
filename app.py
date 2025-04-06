import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import time
import random
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load LLM release dates
llm_releases = pd.read_csv('data/model_release_dates.csv')
# Convert date strings to datetime objects
llm_releases['Release Date'] = pd.to_datetime(llm_releases['Release Date'])

# List of tickers to track
COMPANIES = {
    'NVDA': 'NVIDIA',
    'AMD': 'AMD',
    'AVGO': 'Broadcom',
    'MSFT': 'Microsoft',
    'GOOG': 'Google',
    'META': 'Meta',
    'AMZN': 'Amazon'
}

# Default date range (1 year back from today)
default_end_date = datetime.now()
default_start_date = default_end_date - timedelta(days=365)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Market Trends vs LLM Releases", className="text-center my-4"),
            html.P("A Quick look at LLM Releases vs Major Ticker Performance", 
                   className="text-center mb-4"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Settings"),
                dbc.CardBody([
                    html.H5("Select Companies"),
                    dcc.Checklist(
                        id='company-checklist',
                        options=[{'label': name, 'value': ticker} for ticker, name in COMPANIES.items()],
                        value=['NVDA', 'AMD', 'AVGO'],  # Default selected companies
                        inline=True
                    ),
                    
                    html.H5("Date Range", className="mt-3"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=default_start_date.date(),
                        end_date=default_end_date.date(),
                        max_date_allowed=default_end_date.date()
                    ),
                    
                    html.H5("Chart Type", className="mt-3"),
                    dcc.RadioItems(
                        id='chart-type',
                        options=[
                            {'label': 'Line Chart', 'value': 'line'},
                            {'label': 'Candlestick', 'value': 'candle'}
                        ],
                        value='line',
                        inline=True
                    ),
                    
                    dbc.Button("Update Chart", id="update-button", color="primary", className="mt-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Spinner([
                dcc.Graph(id='stock-chart', style={'height': '600px'})
            ], color="primary", type="border")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("LLM Release Timeline", className="mt-4"),
            html.Div(id='llm-release-table')
        ], width=12)
    ])
], fluid=True)

# Add this function to generate demo data if all API calls fail
def generate_demo_data(ticker, start_date, end_date):
    """Generate synthetic stock data for demo purposes"""
    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Generate random price data
    np.random.seed(hash(ticker) % 10000)  # Use ticker as seed for consistency
    
    # Start with a base price based on the ticker
    base_price = 100 + hash(ticker) % 900  # Between 100 and 1000
    
    # Generate daily returns with some randomness
    daily_returns = np.random.normal(0.0005, 0.015, size=len(date_range))
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    # Calculate prices
    prices = base_price * cumulative_returns
    
    # Create a dataframe
    data = {
        'Open': prices * np.random.uniform(0.99, 1.0, size=len(date_range)),
        'High': prices * np.random.uniform(1.0, 1.03, size=len(date_range)),
        'Low': prices * np.random.uniform(0.97, 0.99, size=len(date_range)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, size=len(date_range))
    }
    
    df = pd.DataFrame(data, index=date_range)
    return df

# Modify the get_stock_data function to use demo data as a last resort
def get_stock_data(ticker, start_date, end_date, max_retries=3, use_demo_fallback=False):
    """
    Get stock data with retry logic and better error handling
    """
    # Add a buffer to the date range to ensure we get data
    # # Sometimes Yahoo Finance needs a bit of padding on the dates
    buffered_start = start_date - timedelta(days=5)
    buffered_end = end_date + timedelta(days=5)
    
    for attempt in range(max_retries):
        try:
            # Add a small random delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 2.0))
            
            # Try to download the data
            stock_data = yf.download(
                ticker, 
                start=buffered_start, 
                end=buffered_end,
                progress=False  # Disable progress bar to reduce console output
            )
            
            if not stock_data.empty:
                # Filter to the actual date range requested
                stock_data = stock_data[(stock_data.index >= start_date) & 
                                       (stock_data.index <= end_date)]
                if not stock_data.empty:
                    return stock_data
            
            # If we got an empty dataframe, wait and try again
            time.sleep(2)
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed for {ticker}: {str(e)}")
            time.sleep(2)  # Wait before retrying

    # If all methods fail and demo fallback is enabled
    if use_demo_fallback:
        print(f"Using demo data for {ticker}")
        return generate_demo_data(ticker, start_date, end_date)
    
    # Return empty dataframe if all methods fail and no fallback
    return pd.DataFrame()

@app.callback(
    [Output('stock-chart', 'figure'),
     Output('llm-release-table', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('company-checklist', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('chart-type', 'value')]
)
def update_chart(n_clicks, selected_companies, start_date, end_date, chart_type):
    if not selected_companies:
        selected_companies = ['NVDA']  # Default to NVIDIA if nothing selected
    
    # Convert string dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Ensure end_date is not in the future
    today = datetime.now().date()
    if end_date.date() > today:
        end_date = pd.Timestamp(today)
    
    # Filter LLM releases within the selected date range
    filtered_releases = llm_releases[
        (llm_releases['Release Date'] >= start_date) & 
        (llm_releases['Release Date'] <= end_date)
    ].sort_values('Release Date')
    
    # Create subplots: one for each company
    fig = make_subplots(
        rows=len(selected_companies), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{COMPANIES[ticker]} ({ticker})" for ticker in selected_companies]
    )
    
    # Add stock data for each company
    for i, ticker in enumerate(selected_companies):
        # Get stock data using our improved function
        stock_data = get_stock_data(ticker, start_date, end_date)
        
        if stock_data.empty:
            fig.add_annotation(
                text=f"No data available for {ticker}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                row=i+1, col=1
            )
            continue
            
        if chart_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'].values.flatten(),
                    name=ticker,
                    line=dict(width=2),
                    mode='lines',  # Explicitly set mode to lines
                    opacity=0.8    # Make slightly transparent to see markers better
                ),
                row=i+1, col=1
            )
        else:  # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'].values.flatten(),
                    high=stock_data['High'].values.flatten(),
                    low=stock_data['Low'].values.flatten(), 
                    close=stock_data['Close'].values.flatten(),
                    name=ticker,
                    increasing=dict(line=dict(color='#26a69a')),
                    decreasing=dict(line=dict(color='#ef5350'))
                ),
                row=i+1, col=1
            )
        
        # Add vertical lines for LLM releases
        for _, release in filtered_releases.iterrows():
            release_date = release['Release Date']
            # Find the closest trading day if the release date is not a trading day
            closest_date = None
            if release_date in stock_data.index:
                closest_date = release_date
            else:
                # Look for the next trading day
                for day_offset in range(1, 5):  # Check up to 5 days ahead
                    next_day = release_date + timedelta(days=day_offset)
                    if next_day in stock_data.index:
                        closest_date = next_day
                        break
                
                # If not found, look for the previous trading day
                if closest_date is None:
                    for day_offset in range(1, 5):  # Check up to 5 days before
                        prev_day = release_date - timedelta(days=day_offset)
                        if prev_day in stock_data.index:
                            closest_date = prev_day
                            break
            
            if closest_date is not None:
                # Use add_vline instead of add_shape
                fig.add_vline(
                    x=release_date,  # Use datetime object directly
                    line=dict(color="rgba(0, 128, 0, 0.7)", width=2, dash="dash"),
                    row=i+1, col=1
                )
                
                # Add annotation for the model name with additional info
                tooltip_text = f"{release['Model Name']}"
                
                # Handle multi-level columns if they exist
                if isinstance(stock_data.columns, pd.MultiIndex):
                    # For multi-level columns from yfinance
                    y_position = stock_data.loc[closest_date, ('High', ticker)] * 1.05
                else:
                    # For single-level columns from synthetic data
                    y_position = stock_data.loc[closest_date, 'High'] * 1.05
                
                fig.add_annotation(
                    x=release_date,
                    y=y_position,
                    text=tooltip_text,
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="rgba(0, 128, 0, 0.7)",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="green",
                    borderwidth=1,
                    row=i+1, col=1,
                    hovertext=f"{release['Model Name']} by {release['Company']}<br>"
                              f"Size: {release['Parameter Size']}<br>"
                              f"Open Source: {release['Open Source']}<br>"
                              f"Notes: {release['Notes']}"
                )
    
    # Update layout
    fig.update_layout(
        height=300 * len(selected_companies),
        showlegend=False,
        title_text="Market Trends vs LLM Releases",
        xaxis_rangeslider_visible=False,
        hovermode="closest",
        plot_bgcolor='rgba(240, 240, 240, 0.8)',  # Light gray background
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update y-axes to include some padding
    for i in range(1, len(selected_companies) + 1):
        fig.update_yaxes(
            title_text="Price ($)",
            autorange=True,
            row=i, col=1
        )
    
    # Create LLM release table
    if len(filtered_releases) > 0:
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Model Name"), 
                html.Th("Company"), 
                html.Th("Release Date"),
                html.Th("Parameter Size"),
                html.Th("Open Source"),
                html.Th("Notes")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(release['Model Name']),
                    html.Td(release['Company']),
                    html.Td(release['Release Date'].strftime('%Y-%m-%d')),
                    html.Td(release['Parameter Size']),
                    html.Td(release['Open Source']),
                    html.Td(release['Notes'])
                ]) for _, release in filtered_releases.iterrows()
            ])
        ], bordered=True, hover=True, striped=True, responsive=True, className="mt-3")
    else:
        table = html.P("No LLM releases found in the selected date range.")
    
    return fig, table

if __name__ == '__main__':
    app.run_server(debug=True) 