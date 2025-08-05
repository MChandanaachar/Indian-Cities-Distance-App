#streamlit (web app)
#let us create our first streamlit webapp
#the streamlit library is not intalled default by google colab
#when we have to install it manually
#!pip install streamlit 

import streamlit as st 

# Sample data
data = {
    'Agra-Delhi': 240,
    'Agra-Lucknow': 334,
    # Add more city pairs here
}

# Get all cities from data keys
cities = set()
for route in data.keys():
    origin_city, dest_city = route.split('-')
    cities.add(origin_city)
    cities.add(dest_city)

# Streamlit app
st.title('Distance Calculator App')

# Dropdowns
origin = st.selectbox('Select Origin', sorted(cities))
destination = st.selectbox('Select Destination', sorted(cities))

# Display distance
key = f'{origin}-{destination}'
reverse_key = f'{destination}-{origin}'

if key in data:
    distance = data[key]
    st.write(f'Distance from {origin} to {destination}: {distance} km')
elif reverse_key in data:
    distance = data[reverse_key]
    st.write(f'Distance from {origin} to {destination}: {distance} km')
else:
    st.write(f'Distance from {origin} to {destination} not available.')
