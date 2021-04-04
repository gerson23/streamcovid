import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import gzip
import csv
from datetime import date, timedelta
from math import log

# pylint: disable=no-value-for-parameter
def main():
    st.title("Stream Covid BR")

    data = pd.read_csv('cases-brazil-states.csv', index_col=['date'],
                    converters={'date': lambda x: date(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2]))})

    data_city = pd.read_csv('cases-brazil-cities.csv', index_col=['date'])

    # Trending chart
    show_trending(data, data_city)

    # Daily notifications
    # show_daily(data)

def show_daily(data):
    st.header("Daily notifications")
    st.sidebar.header("Daily notifications")

    today = date.today()
    d = st.sidebar.date_input("Selected day", date(today.year, today.month, today.day-1))
    opt = st.sidebar.selectbox("Type of view", ['Map', 'Chart'], index=0)
    try:
        filtered_data = data.loc[d, ['newDeaths', 'newCases', 'state']]
    except KeyError:
        st.info("No data for this day")
    else:
        if opt == 'Map':
            st.subheader("Red column is number of new notifications. Black column new deaths")
            show_column_map(filtered_data)
        elif opt == 'Chart':
            show_bar_chart(filtered_data)

def show_trending(data, data_city):
    st.header("Trending")
    st.sidebar.header("Trending")

    st.markdown("Daily new cases with rolling averages")

    opt = st.sidebar.selectbox("Select state or total", data.state.unique())
    data_city_hist = None

    # Handle city selector if a state was selected
    if opt != "TOTAL":
        data_city_filtered = data_city.query(f"state == '{opt}'")
        cities = data_city_filtered.city.unique()
        city = st.sidebar.selectbox("Select city or total", np.insert(cities, 0, 'TOTAL'))
        if city != "TOTAL":
            data_city_hist = lazy_read_data_city(opt, city)
    col = st.sidebar.radio("Data", ['newCases', 'newDeaths'])

    if data_city_hist is None:
        total = data.query(f"state == '{opt}'")
        fore_total = _model_and_fit(total.loc[:, col])
    else:
        total = data_city_hist
        fore_total = _model_and_fit(data_city_hist.loc[:, col])
    total = total.append(fore_total, sort=False)

    total.loc[:, 'Avg60'] = total[col].rolling(window=60).mean()
    total.loc[:, 'Avg30'] = total[col].rolling(window=30).mean()
    total.loc[:, 'AvgExp30'] = total[col].rolling(window=30, win_type='exponential').mean(tau=1)
    st.line_chart(total.loc[:, [col, 'Avg60', 'Avg30', 'AvgExp30', 'predicted']]) 

    st.markdown("Data source: [Number of confirmed cases of COVID-19 in Brazil](https://github.com/wcota/covid19br)")

    st.header("Reproduction number (R0)")
    R0 = _calculate_r0(total)
    R7 = _calculate_r0(total, 8)
    arrow = ":small_red_triangle:" if R0 > R7 else ":small_red_triangle_down:"
    st.markdown(f"### R0 is {R0:.2f} {arrow} (1 week ago was {R7:.2f})")

    st.subheader("The trends")
    st.markdown("* Long term trends are represented by 30 and 60-day averages, the later being the strongest\n"
                "* Short term trend is strongly related to the exponential 30-day average\n"
                "* In few words, for the trend to be really going down, AvgExp30 should cross Avg60 down\n"
                "* Predicted line is made by a basic variational inference using Adam optimization. "
                "This could give a sense of where it is heading, not exact number")
    
    st.subheader("The R0")
    st.markdown("* R0 indicate the average number of people a single infected person infects\n"
                "* There are different ways to model R0. Currently using AIDS model from from "
                "[DIETZ, K. The estimation of the basic reproduction for infectious diseases]"
                "(https://doi.org/10.1177%2F096228029300200103)")
    st.latex(r'''R_0 = 1 + \frac{D \ln(2)} {t_d}''')
    st.markdown("* D = duration of the infectious period\n"
                "* td = number of days to double cases")


def show_bar_chart(data):
    c = alt.Chart(data).mark_bar().encode(x='newCases', y='state', tooltip=['newCases'], color='state')
    st.altair_chart(c, use_container_width=True)

def show_column_map(data):
    data['lat'] = data['state'].apply(_get_coord, args=('lat',))
    data['lon'] = data['state'].apply(_get_coord, args=('lon',))
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=-23.901,
            longitude=-46.525,
            zoom=5,
            pitch=50
        ),
        layers=[
            pdk.Layer(
                'ColumnLayer',
                data=data,
                get_position='[lon, lat]',
                get_elevation='[newCases+newDeaths]',
                radius=20000,
                auto_highlight=True,
                elevation_scale=100,
                elevation_range=[0, 5000],
                pickable=True,
                extruded=True,
                get_color="[10 10 10 255]"
            ),
            pdk.Layer(
                'ColumnLayer',
                data=data,
                get_position='[lon, lat]',
                get_elevation='[newCases]',
                radius=20000,
                auto_highlight=True,
                elevation_scale=100,
                elevation_range=[0, 5000],
                pickable=True,
                extruded=True,
                get_color="[128 10 10 255]"
            )
        ]
    ))

@st.cache(allow_output_mutation=True)
def lazy_read_data_city(state, city):
    city_data = []
    with gzip.open('cases-brazil-cities-time.csv.gz', 'rt') as gfile:
        reader = csv.DictReader(gfile)
        for line in reader:
            if line['state'] == state and line['city'] == city:
                date_split = line['date'].split('-')
                line['date'] = date(int(date_split[0]), int(date_split[1]), int(date_split[2]))
                city_data.append(line)
    
    df = pd.DataFrame.from_records(city_data, index=['date']).loc[:, ['newCases', 'newDeaths']]
    df['newCases'] = df['newCases'].astype('int64')
    df['newDeaths'] = df['newDeaths'].astype('int64')

    return df

@st.cache
def _model_and_fit(total):
    data_tf = tf.convert_to_tensor(total[-70:-14], tf.float64)
    # using last two weeks as target for fit function
    target_tf = tf.convert_to_tensor(total[-14:], tf.float64)
    trend = tfp.sts.LocalLinearTrend(observed_time_series=data_tf)
    seasonal = tfp.sts.Seasonal(num_seasons=7,
                                num_steps_per_season=1,
                                observed_time_series=data_tf)
    model = tfp.sts.Sum([trend, seasonal], observed_time_series=data_tf)

    variational_post = tfp.sts.build_factored_surrogate_posterior(model=model)
    num_variational_steps = 200
    optimizer = tf.optimizers.Adam(learning_rate=.1)
    tfp.vi.fit_surrogate_posterior(target_log_prob_fn=model.joint_log_prob(observed_time_series=target_tf),
                                   surrogate_posterior=variational_post,
                                   optimizer=optimizer,
                                   num_steps=num_variational_steps)
    samples = variational_post.sample(100)

    forecasted_num = 30
    days = pd.date_range(start=date.today(), periods=forecasted_num)
    forescast_dist = tfp.sts.forecast(model=model,
                                      observed_time_series=data_tf,
                                      parameter_samples=samples,
                                      num_steps_forecast=forecasted_num)
    forecasted = forescast_dist.mean().numpy()[..., 0]
    fore_total = pd.DataFrame(forecasted, columns=['predicted'], index=days)

    return fore_total

def _calculate_r0(data, days_ago=1):
    """
    Estimate R0 for the given `days_ago` getting td(doubling time) from 1-week older
    data from the 30-day rolling average.
    """
    window = 7
    D = 14
    target_column = 'Avg30'

    cases_t0 = float(data.loc[date.today() - timedelta(days=days_ago), target_column])
    cases_t7 = float(data.loc[date.today() - timedelta(days=days_ago+window), target_column])

    td = window * cases_t0 / (cases_t0 - cases_t7)

    return 1 + (D * log(2) / td)


def _get_coord(state, orientation):
    try:
        return STATES_COORD[state][orientation]
    except:
        return 0

STATES_COORD = {
    'AC': {'lat': -9.59, 'lon': -70.09},
    'AL': {'lat': -9.63, 'lon': -36.11},
    'AM': {'lat': -4.52, 'lon': -62.76},
    'AP': {'lat': 0.94, 'lon': -51.33},
    'BA': {'lat': -12.93, 'lon': -40.97},
    'CE': {'lat': -5.27, 'lon': -39.18},
    'DF': {'lat': -15.86, 'lon': -47.88},
    'ES': {'lat': -19.83, 'lon': -40.29},
    'GO': {'lat': -15.69, 'lon': -49.81},
    'MA': {'lat': -4.71, 'lon': -44.57},
    'MG': {'lat': -19.21, 'lon': -44.18},
    'MS': {'lat': -20.65, 'lon': -54.75},
    'MT': {'lat': -13.07, 'lon': -56.61},
    'PA': {'lat': -5.89, 'lon': -52.42},
    'PB': {'lat': -7.26, 'lon': -36.04},
    'PE': {'lat': -8.75, 'lon': -37.66},
    'PI': {'lat': -6.86, 'lon': -42.94},
    'PR': {'lat': -24.85, 'lon': -51.11},
    'RJ': {'lat': -22.51, 'lon': -42.67},
    'RO': {'lat': -11.56, 'lon': -62.47},
    'RR': {'lat': -1.12, 'lon': -61.25},
    'RS': {'lat': -29.64, 'lon': -52.89},
    'SC': {'lat': -27.33, 'lon': -50.02},
    'SE': {'lat': -10.71, 'lon': -37.35},
    'SP': {'lat': -22.49, 'lon': -48.15},
    'TO': {'lat': -10.21, 'lon': -47.91}
}

if __name__ == "__main__":
    main()