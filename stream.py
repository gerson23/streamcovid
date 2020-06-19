import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

from datetime import date

# pylint: disable=no-value-for-parameter
def main():
    st.title("Stream Covid BR")

    data = pd.read_csv('cases-brazil-states.csv', index_col=['date'],
                    converters={'date': lambda x: date(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2]))})

    # Trending chart
    show_trending(data)

    # Daily notifications
    show_daily(data)

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

def show_trending(data):
    st.header("Trending")
    st.sidebar.header("Trending")

    st.markdown("Daily new cases with rolling averages")

    opt = st.sidebar.selectbox("Select state or total", data.state.unique())
    col = st.sidebar.radio("Data", ['newCases', 'newDeaths'])
    total = data.query(f"state == '{opt}'")
    total.loc[:, 'Avg60'] = total[col].rolling(window=60).mean()
    total.loc[:, 'Avg30'] = total[col].rolling(window=30).mean()
    total.loc[:, 'AvgExp30'] = total[col].rolling(window=30, win_type='exponential').mean(tau=1)
    st.line_chart(total.loc[:, [col, 'Avg60', 'Avg30', 'AvgExp30']]) 

    st.subheader("The trends")
    st.markdown("* Long term trends are represented by 30 and 60-day averages, the later being the strongest\n"
                "* Short term trend is strongly related to the exponential 30-day average\n"
                "* In few words, for the trend to be really going down, AvgExp30 should cross Avg60 down")

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