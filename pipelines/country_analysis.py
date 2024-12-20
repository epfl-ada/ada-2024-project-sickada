import os
import os.path as op
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests as req
from bs4 import BeautifulSoup as Bs
from tqdm.notebook import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import content_categories, inverted_categories
import pycountry
from countryinfo import CountryInfo


custom_color_map = { # thanks gpt
    'Southern Europe': '#FFA07A',  # Light Salmon
    'Western Europe': '#FA8072',  # Salmon
    'Eastern Europe': '#E9967A',  # Dark Salmon
    'Northern Europe': '#F08080',  # Light Coral

    'Western Asia': '#87CEEB',    # Sky Blue
    'Southern Asia': '#4682B4',  # 
    'Eastern Asia': '#1E90FF',   # 

    'Antarctica': '#ADD8E6',      # Light Blue
    'Oceania': '#FFD700',  # Gold

    'South America': '#32CD32',   # Lime Green
    'Northern America': ' #2E8B57', 
    'Caribbean': '#66CDAA',      # Blue Violet
    'Central America': '#20B2AA',  # Light Sea Green
    
    'Northern Africa': '#BDB76B',  # Dark Khaki
    'Western Africa': '#DAA520',  # Goldenrod
}





def get_iso3(x):
    # pip install pycountry # https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes#
    return pycountry.countries.get(alpha_2=x).alpha_3


def get_country_fullname(x):
    return pycountry.countries.get(alpha_2=x).name


def get_region(country_name):
    country_name = country_name.split(',')[0]
    if country_name.lower() == 'andorra':
        return 'Southern Europe'
    if country_name.lower() == 'antarctica':
        return 'Antarctica'
    if country_name.lower() == 'bahamas':
        return 'Caribbean'
    if country_name.lower() == 'czechia':
        return 'Eastern Europe'
    if country_name.lower() == 'falkland islands (malvinas)':
        return 'South America'
    if country_name.lower() == 'korea':
        return 'Eastern Asia'
    if country_name.lower() == 'north macedonia':
        return 'Southern Europe'
    if country_name.lower() == 'türkiye':
        return 'Western Asia'
    if country_name.lower() == 'taiwan, province of china':
        return 'Eastern Asia'
    if country_name.lower() == 'viet nam':
        return 'South-Eastern Asia'
    
    
    country_info = CountryInfo(country_name).info()
    return country_info.get('subregion', 'Unknown')  



def filter_categories(edu, kids = False, letters= False):
    df= edu[(edu.category == '9') |~(edu.category.str.endswith('9'))]
    if not kids:
        df= df[df.category != '5']
    if not letters:
        df  = df[~df.category.isin(['a', 'android', 'q', 's', 'life'])]
    return df[df.category != 'unclass']


def plot_category_pie(countries, country_names, df, rows=1):
    cols = len(countries) // rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=country_names, specs=[[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)])
    
    df_sel = df[df.country.isin(countries)].groupby('subcategory')
    lengths = df_sel.country.value_counts().unstack().sum(axis=0)
    
    for i, country in enumerate(countries):
        country_data = df_sel.country.value_counts().unstack()[country]
        inverted =[inverted_categories[cat.lower()] for cat in country_data.index]
        fig.add_trace(
            go.Pie(
                labels=country_data.index,
                values=country_data.values,
                name=country,
                hole=0.1,  #donut chart
                customdata=inverted,  # Add category IDs as customdata
                hovertemplate=("<b>%{label}</b><br>  Category ID: %{customdata}<br> Count: %{value}<br>  <extra></extra>"),
                text = inverted,
                textinfo='text',
            ),
            row=(i // (len(countries) // rows)) + 1, 
            col=(i % (len(countries) // rows)) + 1,
        )
    
    fig.update_layout(
        # title_text="Category Distribution by Country",
        height=500 * rows, 
        showlegend=False,
        
    )

    fig.update_traces( # add percent
    hovertemplate="<b>%{label}: %{customdata}</b><br># videos: %{value}<br> Count: %{value}<br> ",
    )
    del df_sel
    return fig