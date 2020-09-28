import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import altair as alt
import seaborn as sns
import altair as alt
import plotly.figure_factory as ff
import plotly.offline as py
sns.set()


st.title("Covid19 Hong Kong Open Data")

#st.header('by Jayden Yuen')



st.markdown("## " + 'TotalCases/Deaths/Recoveries Trend')	
st.markdown("#### " +"What Trends would you like to see?")

selected_metrics = st.selectbox(
    label="Choose...", options=['TotalCases','Deaths','Recoveries']
)

DATA_URL = ('https://raw.githubusercontent.com/7ayden/covid19/master/cdf2020-09-24.csv')
@st.cache(allow_output_mutation=True)
def get_data():
    data = pd.read_csv(DATA_URL)
    data['Date'] = pd.to_datetime(data['Report date']).dt.strftime('%d-%m-%Y')
    return data

#load data from url
df = get_data()
#df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
#df.set_index("Case no.", inplace = True)
District = st.sidebar.multiselect('Show which districts did you travel', df['District'].unique())

#date = st.sidebar('Show Which date were you there?', df['Report date'].unique())

# Filter dataframe
#new_df = df[(df['District'].isin(District)) & (df['Report date'].isin(date))]

new_df = df[(df['District'].isin(District))]

#new_df.sort_values(by=['Report Date'], inplace=True, ascending=False)


# create figure using plotly express
fig = px.scatter(new_df, x ='Report date',y='Age',color='Gender')

#if st.checkbox('Plot by district and Age '):
# Plot!
st.plotly_chart(fig)

#if st.checkbox('Show Historical average data'):
  # write dataframe to screen
st.write(new_df)



#clubs = st.multiselect('Show Player for clubs?', df['Club'].unique())nationalities = st.multiselect('Show Player from Nationalities?', df['Nationality'].unique())# Filter dataframe
#new_df = df[(df['Club'].isin(clubs)) & (df['Nationality'].isin(nationalities))]# write dataframe to screen

#st.write(new_df)
#df.loc[:, ~df.columns.str.contains('^Unnamed: 0')]
st.markdown("## " + 'Historical data stats')
if st.checkbox('Show Historical average data stats'):
  st.table(df[['District','Age' ]].groupby('District').agg(['mean','min','max', 'count']).sort_values(by=[('Age','mean')]))

st.markdown("## " + 'last 14 days cases')
	
import datetime as dt
df['Report date']= pd.to_datetime(df['Report date'], dayfirst = True).dt.date
range_max = df['Report date'].max()
range_min = range_max - dt.timedelta(days=14)

# take slice with final week of data
sliced_df = df[(df['Report date'] >= range_min) & 
               (df['Report date'] <= range_max)]
if st.checkbox('Show last 14 days cases data by district'):
  st.write(pd.DataFrame(sliced_df.groupby(['Report date'], sort=False)['District'].counts()))

st.markdown("## " + ' Show by district')

if st.checkbox('Show last 14 days cases data'):
  st.dataframe(sliced_df)


st.markdown("## " + ' FacetGrid')

a = sns.FacetGrid(df, hue = 'Hospitalised/Discharged/Deceased', aspect=4, palette="Set1" )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , df['Age'].max()))
a.add_legend()


Hospitalised = df[df['Hospitalised/Discharged/Deceased']== 'Hospitalised']

#fig = px.bar(sliced_df, x="Report day", y="District", color="Age", barmode="group", facet_col="Gender"
#fig.show()




'''
st.markdown("## " + 'Normal Curve')
chart = px.histogram(sliced_df, x="District", y="Report date", color="Gender",
                   marginal="box", # or violin, rug
                   hover_data=sliced_df.Age)
chart
'''
st.markdown("## " + 'last 14 days cases by age and gender')



st.markdown("## " + 'last 14 days cases by age and gender')

chart2 = px.scatter(data_frame=sliced_df,
           x="Report date",
           y="Age",
           color="Gender",
           title="Last 14 days cases by age and gender")
chart2

st.markdown("## " + 'last 14 days total cases by gender')

#chart3 = px.histogram(data_frame=sliced_df,x="Report date",color="Gender", title="Last 14 days total cases count by gender")
'''
women_bins = df['Gender']== 'F'
men_bins = df['Gender']== 'M'

y = list(range(0, 100, 10))

layout = go.Layout(yaxis=go.layout.YAxis(title='Age'),
                   xaxis=go.layout.XAxis(
                       range=[0, 1200],
                       title='Number'),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=y,
               x=men_bins,
               orientation='h',
               name='Men',
               hoverinfo='x',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=y,
               x=women_bins,
               orientation='h',
               name='Women',
               text=-1 * women_bins,
               hoverinfo='text',
               marker=dict(color='seagreen')
               )]
fig = go.plot(dict(data=data))
fig
#group_labels= ['Gender']
#fig = ff.create_displot(sliced_df, group_labels)
#fig
'''
st.markdown("## " + ' plotbox last 14 days total cases by gender')

chart4 = px.box(data_frame=sliced_df,
           x="Report date",
           y="Age",
           color="Gender",
           title="plotbox on the last 14 days cases")
chart4
st.markdown("## " + ' scatterplot')

chart5 = px.scatter(data_frame=df, x="Age", y=categorical("Case classification*"), facet_col="Hospitalised/Discharged/Deceased", marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white")
chart5


st.markdown("## " + ' age distribution')
##trying to use count plot on age distribtuion 

fig = sns.FacetGrid(df, hue = 'Hospitalised/Discharged/Deceased', aspect=4, palette="Set1" ).map(sns.kdeplot, 'Age', shade= True ).set(xlim=(0 , df['Age'].max())).add_legend()
st.pyplot(fig)
#bar_chart(df['Gender'])



#.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))



## sidebar 
st.sidebar.title("District Data")
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)


# Create traces

Hospitalised = df[df["Hospitalised/Discharged/Deceased"] == "Hospitalised"]
df2 = df[df["Defect severity"] == "High"]

fig = go.Figure()
if selected_metrics == 'Hospitalised':
  fig.add_trace(go.Bar(x=Hospitalised['Report date'],y=Hospitalised['Hospitalised/Discharged/Deceased']))
if selected_metrics == 'Deaths':
	fig.add_trace(go.Bar(x=df['Report date'],y=df[df['Hospitalised/Discharged/Deceased']=='Deceased'],
	                    mode='markers', name='Deaths'))
if selected_metrics == 'Recoveries':
	fig.add_trace(go.Bar(x=df['Report date'], y=df[df['Hospitalised/Discharged/Deceased']=='Discharged'],
	                    mode='relative',
	                    name='Recoveries'))
st.plotly_chart(fig, use_container_width=True)


## By scatter plit 

go.Scatter(alt.Chart(df).mark_bar().encode(alt.X('Report data', bin=True),
	                                                          	y = 'count()'))



min_year = int(df['Age'].min())
max_year = int(df['Age'].max())

District = df['District'].unique()

'## By district'
District = st.selectbox( 'df.District',df[df['District'] == District])



filter_data = df[df['Report date'] >='2020-08-31'].set_index("Date") 
st.markdown(str(','.join(country_name_input)) + " daily Death cases from 1st April 2020")
# bar chart 
st.bar_chart(filter_data[['Deaths']])


#histogram
hist_values = np.histogram(
df['Report date'].dt.weekday_name, bins=7, range=(0,7))[0]
st.bar_chart(hist_values)

'## By month'
month = st.selectbox('Month', range(1, 13))
df[df['Report data'] == month]



#data_load_state = st.text('Loading data...')
#data_load_state.text("Done! (using st.cache)")

#st.dataframe(data2)
#st.dataframe(data3)
#st.dataframe(data4)

#cols = ["Case no.", "Report date", "Gender", "Age", "Hospitalised/Discharged/Deceased"]
#st_ms = st.multiselect("Columns", df.columns.tolist(), default=cols)

#st.table(df.groupby("Gender").Age.mean().reset_index()\
#.round(2).sort_values("Age", ascending=False)\
#.assign(avg_age=lambda x: x.pop("Age").apply(lambda y: "%.2f" % y)))

#values = st.sidebar.slider("Age range", df.Age.min()), 100, (0, 100))
#f = px.histogram(df.query(f”age.between{values}”), x=”Age”, nbins=5, title=”Age distribution”)
#f.update_xaxes(title=”Age”)
#f.update_yaxes(title=”No. of listings”)
#st.plotly_chart(f)
