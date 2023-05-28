import streamlit as st
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# For movie type filter -> Han Zhang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import re
from PIL import Image
import time



# for KNNregression -> Lei Liu
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud
import altair as alt
from vega_datasets import data
import altair_viewer


# Data and plots -> Jie Chang
st.set_option('deprecation.showPyplotGlobalUse', False)
# Import the drawtree map package
import squarify

# content:
def main():
    # set title
    st.title("All about Movies")
    
#     # set password
#     password_check = st.text_input("Please enter your password")
#     if password_check != st.secrets['password']:
#         st.stop()
    st.markdown("<h2 style='color:green;font-size:24px;'>Amazing movie data analysis</h2>", unsafe_allow_html=True)    
#     st.subheader("Amazing movie data analysis")
    # set pic
    #     image = Image.open('/Users/hanhan/project3/headerPic.png')
#     st.image(image)
#     st.image('headerPic.png')
    st.image('project3/headerPic.png')
    
#     ################################################################################# 
    
#                                     # Jie Chang
        
#     ################################################################################# 

    st.image('project3/JieBackground.jpeg', width = 800)
    st.header("Jie Chang")
#     # 1. Data scrapping
#     # header
    st.header('Data')
    st.subheader('Web Scratching')
    code = """"
    from selenium import webdriver
    from lxml import etree
    import time
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    import random
    from  openpyxl import  Workbook 
    driver_path = "chromedriver.exe"
    from selenium.webdriver.chrome.options import Options
    
    # Instantiate a startup parameter object.
    chrome_options = Options()
    chrome_options.add_argument('blink-settings=imagesEnabled=false') 
    chrome_options.add_argument('--disable-infobars')
    driver = webdriver.Chrome(executable_path=driver_path,chrome_options=chrome_options)

    for year in range(2010,2022):
    root_url = f"https://www.the-numbers.com/box-office-records/worldwide/all-movies/cumulative/released-in-{str(year)}"
    driver.get(root_url)
    tabs = driver.find_elements(By.XPATH,"//div[@class='pagination']/a")
    urls = [tab.get_attribute('href') for tab in tabs]
    name_url = {}
    first_list = []
    for url in urls:
        driver.get(url)
        titles = driver.find_elements(By.XPATH,"//div[@id='page_filling_chart']//table/thead/tr/th")
        tt = [ title.text for title in titles]
        rows =  driver.find_elements(By.XPATH,"//div[@id='page_filling_chart']//table/tbody/tr")
        for row in rows:
            datas = row.find_elements(By.XPATH,"./td")
            a_url = row.find_element(By.XPATH,"./td//a").get_attribute('href')
            r_data = [data.text for data in datas]
            first_list.append(r_data)
            name_url[r_data[1]] = a_url
            
    # subpage
    second_title = ['Domestic Releases:','Production Budget:','Theater counts:','Running Time:','Keywords:','Genre:','Production Countries:','Languages:']
    second_dic = {}
    for name,n_url in name_url.items():
        driver.get(n_url) 
        second_dic[name] = ['']*len(second_title)
        trs = driver.find_elements(By.XPATH,"//div[@class='content active']/table//tr")

        for tr in trs:
            tds = tr.find_elements(By.XPATH,"./td")
            if len(tds) == 2:
                if tds[0].text in second_title:
                    second_dic[name][second_title.index(tds[0].text)] = tds[1].text
    for i in range(len(first_list)):
        first_list[i].extend(second_dic[first_list[i][1]])
        
    
    wb = Workbook()
    ws = wb.active
    tt.extend(second_title)
    ws.append(tt)

    for row in first_list:
        ws.append(row)
    wb.save(f"movies{year}.xlsx")
    
    root_url = f"https://www.the-numbers.com/box-office-records/worldwide/all-movies/cumulative/released-in-2022"
    driver.get(root_url)
    tabs = driver.find_elements(By.XPATH,"//div[@class='pagination']/a")
    urls = [tab.get_attribute('href') for tab in tabs]
    name_url = {}
    first_list = []
    for url in urls:
        driver.get(url)
        titles = driver.find_elements(By.XPATH,"//div[@id='page_filling_chart']//table/thead/tr/th")
        tt = [ title.text for title in titles]
        rows =  driver.find_elements(By.XPATH,"//div[@id='page_filling_chart']//table/tbody/tr")
        for row in rows:
            datas = row.find_elements(By.XPATH,"./td")
            a_url = row.find_element(By.XPATH,"./td//a").get_attribute('href')
            r_data = [data.text for data in datas]
            first_list.append(r_data)
            name_url[r_data[1]] = a_url
            
    # subpage
    second_title = ['Domestic Releases:','Production Budget:','Theater counts:','Running Time:','Keywords:','Genre:','Production Countries:','Languages:']
    second_dic = {}
    for name,n_url in name_url.items():
        driver.get(n_url) 
        second_dic[name] = ['']*len(second_title)
        trs = driver.find_elements(By.XPATH,"//div[@class='content active']/table//tr")

        for tr in trs:
            tds = tr.find_elements(By.XPATH,"./td")
            if len(tds) == 2:
                if tds[0].text in second_title:
                    second_dic[name][second_title.index(tds[0].text)] = tds[1].text
    for i in range(len(first_list)):
        first_list[i].extend(second_dic[first_list[i][1]])

    wb = Workbook()
    ws = wb.active
    tt.extend(second_title)
    ws.append(tt)

    for row in first_list:
        ws.append(row)
    wb.save("movies2022.xlsx")

    
    """
    st.code(code, language='python')
    
    
    
    
 
    
    
    
    
    st.subheader('Data Wrangling')
    code1 = """
    file_paths = glob.glob('*.xlsx')
    merged_data = pd.DataFrame()
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        merged_data = pd.concat([merged_data, df], ignore_index=True)
    
    # movie budget 
    import re
    pattern = r'\(.*?\)'
    merged_data['Production Budget:'] = [re.sub(pattern, '', str(string)) for string in merged_data['Production Budget:']]
    
    # Movie theater broadcast volume cleaning, including the number of movie theaters and the number of playing weeks
    merged_data['Opening_theaters'] = merged_data['Theater counts:'].str.extract(r'([\d,]+) opening theaters')
    merged_data['Max_theaters'] = merged_data['Theater counts:'].str.extract(r'([\d,]+) max\. theaters')
    merged_data['Average_run(week)'] = merged_data['Theater counts:'].str.extract(r'([\d.]+) weeks average run per theater')
    
    # Extract showtimes
    merged_data['Domestic Releases:'] = merged_data['Domestic Releases:'].str.extract(r'(.*?)\s*\(', expand=False)
    
    # delect some column which I don't use
    merged_data= merged_data.drop('Theater counts:',axis=1)
    
    # change data type
    merged_data['Average_run(week)'] = merged_data['Average_run(week)'].astype(float)
    merged_data['Max_theaters'] = merged_data['Max_theaters'].str.replace(',', '').fillna(0).astype(int)
    merged_data['Opening_theaters'] = merged_data['Opening_theaters'].astype(str)
    merged_data['Running Time:'] = merged_data['Running Time:'].str.replace(' minutes', '').fillna(0).astype(int)
    merged_data['Production Budget:'] = merged_data['Production Budget:'].str.replace('$', '',regex=True).str.replace(',', '',regex=True).replace('nan', np.nan,regex=True)
    merged_data['Production Budget:'] =merged_data['Production Budget:'].fillna(0).astype(int)
    merged_data['Worldwide Box Office'] = merged_data['Worldwide Box Office'].str.replace('$', '',regex=True).str.replace(',', '',regex=True).replace('nan', np.nan,regex=True).fillna(0)
    merged_data['Worldwide Box Office']=merged_data['Worldwide Box Office'].astype(float)
    merged_data['Domestic Box Office'] = merged_data['Domestic Box Office'].str.replace('$', '',regex=True).str.replace(',', '',regex=True).astype(float).fillna(0)
    merged_data['Domestic Box Office'] = merged_data['Domestic Box Office'].astype(int)
    merged_data['International Box Office'] = merged_data['International Box Office'].str.replace('$', '',regex=True).str.replace(',', '',regex=True).replace('nan', np.nan,regex=True).fillna(0).astype(int)
    merged_data['Domestic\nShare'] = merged_data['Domestic\nShare'].str.strip('%').astype(float) / 100
    merged_data.columns = merged_data.columns.str.replace(':', '',regex=True)
    merged_data.rename(columns={'Domestic Releases': 'Date'}, inplace=True)
    merged_data.to_csv('wrangling_data.csv', index=False)
    
    """
    
    st.code(code1, language='python')
    

    ## data processing and plotting
    st.subheader('Plotting')
        # upload movie csv
    uploader_file3 = st.file_uploader(
        label = "Upolad your dataset"
        , key="uploader3"
    )
    
    data3 = None 
    
   # if the file is uploaded successfully, then go to train model
    if uploader_file3 is None:
     # if file is empty, display: please upload file
        st.error("Please upload a file！")
        return

    else:
        st.success('Successfully uploaded!')
        data3 = pd.read_csv(uploader_file3)
        data3.dropna(inplace=True)
        
        
    # data processing
    # Clean out the data rows with null values directly, and the result shows that there are 1998 rows left
    data3 = data3.dropna(how='any')
    data3.Date = data3.Date.str.replace(' ', '/',regex=True).str.replace(',', '',regex=True)
    data3.Date = data3.Date.str.replace("th", "")
    data3.Date = data3.Date.str.replace("st", "")
    data3.Date = data3.Date.str.replace("nd", "")
    data3.Date = data3.Date.str.replace("rd", "")
    data3['Date'] = data3['Date'].str.split('/').str[0].str[:3]+'/'+data3['Date'].str.split('/').str[1]+'/'+data3['Date'].str.split('/').str[2]
    data3 = data3.dropna(subset="Date")
    data3.Date = pd.to_datetime(data3.Date,format="%b/%d/%Y")
    data3['Year'] = data3['Date'].dt.year
    
    # create figure
    # figure 1
    st.write('fig 1 - Movie by Year')
    movie_years_count = data3.groupby('Year')['Movie'].count()
    plt.figure(figsize=(10, 8), dpi=80)
    movie_years_count.plot()
    plt.title('Movie by Year', fontsize=12)
    plt.xlabel('Year')
    plt.ylabel('Count')
    st.pyplot()
    
    # description
    st.write("""
    Description:
    
    Here I use the groupby statement to directly group the target label items, and count() is an aggregation function to summarize the statistics of the movie
    
    It can be seen that the number of movies has dropped sharply after 2018, and it is guessed that the reason is due to the covid-19
    
    
    """)
    


    
    # Movie box office trends over the years
    st.write('fig 2 - Movie box office trends over the years')
    movie_gross = data3.groupby('Year')['Worldwide Box Office'].sum()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
    x = movie_gross.index.tolist()
    y = movie_gross.values
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    ax.bar(x, y, color=colors)

    for i in range(len(x)):
        ax.text(x[i], y[i]+0.05, '%i' % y[i], ha='center', va='bottom', rotation=45)
    ax.set_xlabel('Year')
    ax.set_ylabel('World Box Office')
    ax.set_title('World Box Office by Year')
    st.pyplot(fig)
    st.write("""
    Description:
    
    It is evident that the global movie box office experienced a sharp decline starting from the outbreak of the pandemic in 2019. However,
    after the end of the pandemic in 2021, there has been a gradual
    recovery in box office revenue.
    
    
    """)
    
    
    # 'Worldwide Box Office by Average run(week)
    st.write('fig 3 - Worldwide Box Office by Average run(week)')
    run_week_count = data3.groupby('Average_run(week)')['Worldwide Box Office'].count()
    fig, ax = plt.subplots()
    run_week_count.plot(ax=ax)
    ax.set_title('Worldwide Box Office by Average run(week)', fontsize=12)
    ax.set_xlabel('Average run(week)')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.write("""
    Description:
    
    Let's analyze the impact of the number of weeks a movie is screened on its box office 
    performance. We can observe that initially, the box office revenue tends to 
    increase with the number of weeks a movie is screened. However, after around 8 weeks 
    of screening, the impact on the box office revenue becomes less significant.


    """)
    
    
    # Distribution of Movie Genres
    st.write('fig 4 - Distribution of Movie Genres')
    movie_type = data3['Genre'].str.split('/')
    # Convert List to Series
    movie_type = movie_type.apply(pd.Series)
    # Use the unstack function to rotate rows into columns and rearrange data:
    movie_type = movie_type.apply(pd.value_counts)
    # At this time, the data is a Series, remove the null value, and convert it to Dataframe by reset_index()
    movie_type = movie_type.unstack().dropna().reset_index()
    # Summary of movie genres
    movie_type.columns =['level_0','type','counts']
    amovie_type_m = movie_type.drop(['level_0'],axis=1).groupby('type').sum().sort_values(by=['counts'],ascending=False).reset_index()
    size = [1100, 900, 759, 416, 323, 300, 286, 180, 148, 136, 128, 105, 104, 103, 92]
    name = ['Drama', 'Action', 'Adventure', 'Comedy', 'Suspense', 'Thriller', 'Horror', 'Romantic Comedy',
            'Black Comedy', 'Documentary', 'Musical', 'Western', 'Concert', 'Performance', 'Multiple Genres', 'Reality']
    colors = ['steelblue', '#9999ff', 'red', 'indianred', 'green', 'yellow', 'orange']

    fig, ax = plt.subplots(figsize=(8, 6))

    squarify.plot(
        sizes=size,
        color=colors,
        label=name,
        value=size,
        alpha=0.5,
        edgecolor='white',
        linewidth=3,
        ax=ax
    )

    plt.rc('font', size=7)
    ax.set_title('Distribution of Movie Genres', fontdict={'fontsize': 20})
    ax.axis('off')
    ax.tick_params(top='off', right='off')

    st.pyplot(fig)
    
    
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('========================================================================================')
    



   
    

    
    
    
    
    
    ################################################################################# 
    
                                # Han Zhang

    ################################################################################# 
    
    st.image('project3/HanBackground.webp')
    st.header('Han Zhang')
    # 1. film filter model
    # header
    st.header('Movie Types')
    
    # upload movie csv
    uploader_file = st.file_uploader(
        label = "Upolad your dataset"
        , key="uploader1"
    )
    
    input_df = None 
    
   # if the file is uploaded successfully, then go to train model
    if uploader_file is None:
     # if file is empty, display: please upload file
        st.error("Please upload a file！")
        return

    else:
        st.success('Successfully uploaded!')
        input_df = pd.read_csv(uploader_file)
        input_df.dropna(inplace=True)
    

    # feature
    feature = input_df['Keywords'] # get feature
    feature_lists = feature.values.tolist()
    
    # target
    target = input_df['Genre']
    target_lists = target.values.tolist()
    
    # transfrom the feature to numeric features
    tf = TfidfVectorizer()


    n_feature = tf.fit_transform(feature_lists)
    
    # split data to training data and test data
    x_train, x_test, y_train, y_test = train_test_split(n_feature, target_lists, test_size=0.2, random_state=453)
    
    # model: MultinomialNB
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    score1 = nb.score(x_test,y_test)
    
  

    st.write("## Model score is: ", score1)
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    

    # display label names and apperance times
    targetvc = target.value_counts()
    chart_data=pd.DataFrame(targetvc)
    st.bar_chart(chart_data)
    
    
    
    

    # data display
    c = CountVectorizer()
    result = c.fit_transform(feature_lists)
    feature_names = c.get_feature_names_out()
    # use regular expression to deduct some numeric words
    filtered_feature_names = [name for name in feature_names if not re.search(r'\d', name)]
    filtered_feature_names = [name for name in filtered_feature_names if len(name) >= 4]

    selected_options = st.multiselect('Please select the description:', filtered_feature_names)
    st.write('The selected descriptions are:', selected_options)
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')

    st.write("#### Your current potential movie preferences list：")

    if selected_options:
        filtered_data_frames = []

        # based on user selection, store the filtered data frames
        for option in selected_options:
            filtered_rows = input_df[input_df['Keywords'].str.contains(option, case=False)]
            if not filtered_rows.empty:
                filtered_data_frames.append(filtered_rows)

        # Display the latest filtered data frame
        if filtered_data_frames:
            latest_filtered_frame = filtered_data_frames[-1]
            st.write(f"Length of the filtered data: {len(latest_filtered_frame)}")
            latest_filtered_frame = latest_filtered_frame.iloc[:,1:].reset_index(drop=True)
            st.write(latest_filtered_frame)
        



     # submission button:
    with st.form("latest_filtered_frame"):
        
        submitted=st.form_submit_button("Submit to predict your movie preference")
        if submitted:  
          # add progress bar
            import time
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(100):
                latest_iteration.text(f'Iteration{i+1}')
                bar.progress((i+5) % 101)
                time.sleep(0.02)

            # feature
            featureUser = latest_filtered_frame['Keywords']
            featureUser_list = featureUser.values.tolist()


            # transfrom the feature to numeric features
            n_feature = tf.transform(featureUser_list)

            # model: MultinomialNB
            target_userget = nb.predict(n_feature)
            if len(target_userget) > 0:
                st.write("### Your movie preference is: ", target_userget[0])
            
            
            
            
            
            
            # movie remoccendation
            st.subheader('Here are top 10 movies you might be interested in:')
            selected_rows = input_df[input_df['Genre'] == target_userget[0]]
            RecommendedMovies = selected_rows.sort_values('Worldwide Box Office', ascending= False)
            columns_to_keep = ['Movie', 'Genre','Languages', 'Worldwide Box Office', 'Running Time' , 'Production Countries','Keywords']
            RecommendedMovies = RecommendedMovies[columns_to_keep]
            RecommendedMovies_slice = RecommendedMovies.head(10).reset_index(drop=True)
            st.dataframe(RecommendedMovies_slice, width = 800)
            
            
            
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('                            ')
    st.write('========================================================================================')


            
            
            
            
            
            

    ################################################################################# 
    
                                    # Lei Liu
        
    ################################################################################# 
    
    
    st.image('project3/LeiBackground.jpg', width = 700)
    st.header('Lei Liu')
    # Data Wrangling
    st.subheader('Data Wrangling - Dealing with date and time')
    code2 = """
    from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    from wordcloud import WordCloud
    
    data.Date = data.Date.str.replace(' ', '/',regex=True).str.replace(',', '',regex=True)
    data.Date = data.Date.str.replace("th", "")
    data.Date = data.Date.str.replace("st", "")
    data.Date = data.Date.str.replace("nd", "")
    data.Date = data.Date.str.replace("rd", "")
    data['Date'] = data['Date'].str.split('/').str[0].str[:3]+'/'+data['Date'].str.split('/').str[1]+'/'+data['Date'].str.split('/').str[2]
    data = data.dropna(subset="Date")
    data.Date = pd.to_datetime(data.Date,format="%b/%d/%Y")

    """    
    st.code(code2, language='python')
    
    # 1. KNNregression model
    st.subheader('Worldwide Box Office Prediction')
    
    # upload movie csv
    uploader_file2 = st.file_uploader(
        label = "Upolad your dataset"
        , key="uploader2"
    )
    
    data = None 
    
   # if the file is uploaded successfully, then go to train model
    if uploader_file2 is None:
     # if file is empty, display: please upload file
        st.error("Please upload a file！")
        return

    else:
        st.success('Successfully uploaded!')
        data = pd.read_csv(uploader_file2)
        data.dropna(inplace=True)
        
    
    
    
    
    # data processing
    data.Date = data.Date.str.replace(' ', '/',regex=True).str.replace(',', '',regex=True)
    data.Date = data.Date.str.replace("th", "")
    data.Date = data.Date.str.replace("st", "")
    data.Date = data.Date.str.replace("nd", "")
    data.Date = data.Date.str.replace("rd", "")
    data['Date'] = data['Date'].str.split('/').str[0].str[:3]+'/'+data['Date'].str.split('/').str[1]+'/'+data['Date'].str.split('/').str[2]
    data = data.dropna(subset="Date")
    data.Date = pd.to_datetime(data.Date,format="%b/%d/%Y")
    data['Month'] = data['Date'].dt.month
    
    # data cleaning, remove the data have no Genre infomation 
    data.dropna(subset='Genre',inplace=True)
    
    
    # fig 1
    st.write("fig 1 - Top 10 Genres Movies Count ")
     # display label names and apperance times
    def plot_genre_counts(data):
        fig, ax = plt.subplots(figsize=(6, 4))
        genre_counts = data['Genre'].value_counts()[:10].sort_values(ascending=False)
        ax = sns.countplot(y='Genre', data=data[data['Genre'].isin(genre_counts.index)], order=genre_counts.index, palette='hls')
        for i, count in enumerate(genre_counts.values):
            ax.text(count, i, str(count), ha='right', va='center', color='white', fontweight='bold')
        ax.set_title('Top 10 Genres Movies Count')
        ax.set_xlabel('Count')
        ax.set_ylabel('Genre')
        return fig

    st.title('Top 10 Genres Movies Count')
    fig = plot_genre_counts(data)
    st.pyplot(fig)
    st.write('Through our analysis of the Worldwide Box Office, we have discovered that while Drama genre has the most movies produced, Action movies are the most popular, closely followed by Adventure genre.')
    


    
#   fig 2
    st.write('fig 2 - Top 10 Genre Worldwide Box Office (Log Scale)')
    data_group=data[['Worldwide Box Office','Genre']].groupby('Genre').sum()
    box_office_log = np.log10(data_group['Worldwide Box Office'])
    sns.set_style('whitegrid')

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=box_office_log, y=data_group.index, data=data_group)

    plt.title('Top 10 Genre Worldwide Box Office (Log Scale)')
    plt.xlabel('Box Office (Log Scale)')
    plt.ylabel('Genre')

    for i, v in enumerate(box_office_log):
        ax.text(v, i, '{:.2f}'.format(v), ha='right', va='center',color='white', weight='bold')
    
    st.title('Top 10 Genre Worldwide Box Office')
    st.pyplot(plt)

    
    


    
    
   # fig 3
    st.write('fig 3 - Scatter Plot Matrix')
#     sns.set_style('whitegrid')

#     # Create scatter plot matrix
    box_office_df = data[['Domestic Box Office','International Box Office','Production Budget','Running Time','Genre','Month']]
#     sns.pairplot(box_office_df, vars=['Domestic Box Office', 'International Box Office', 'Production Budget', 'Running Time', 'Month'], hue='Genre')

#     # Show the plot
#     plt.show()
    st.title('Scatter Plot Matrix')
#     st.pyplot(plt)
    st.image('project3/LeiLiufig 3 - Scatter Plot Matrix.png')

 
    
    st.write('Upon analyzing the pairplots, it becomes evidentno significant correlations betw that there are een the running time and other variables.')
    box_office_df = box_office_df.drop(columns=['Running Time','Month'])
    box_office_df = pd.get_dummies(box_office_df,columns=['Genre'])

    

    
    ### R2 and RMSE
    st.write('fig 4 - RSquare for the KNeighborsRegressorlots and RMSE for the KNeighborsRegressor')
    st.image("project3/LeiLiuRSquare_fig4.png")
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(box_office_df), columns=box_office_df.columns)
    X = scaled_df.drop('International Box Office', axis=1)
    y = box_office_df['International Box Office']

    # train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     knn = KNeighborsRegressor(n_neighbors=5, weights='uniform')
#     mod = knn.fit(X, y)

#     # predict
#     x = X + 0.0001
#     y_hat = mod.predict(x)

#     # calculate calculate_regression_goodness_of_fit
#     def calculate_regression_goodness_of_fit(ys, y_hat):
#         ss_total = np.sum(np.square(ys - np.mean(ys)))
#         ss_residual = np.sum(np.square(ys - y_hat))
#         ss_regression = np.sum(np.square(y_hat - np.mean(ys)))

#         r_square = ss_regression / ss_total
#         rmse = np.sqrt(ss_residual / float(len(ys)))

#         return r_square, rmse

#     calculate_regression_goodness_of_fit(y, y_hat)

#     rsquare_arr = []
#     rmse_arr = []

#     for k in range(2, 200):
#         knn = KNeighborsRegressor(n_neighbors=k)
#         y_hat = knn.fit(X, y).predict(x)
#         rsquare, rmse = calculate_regression_goodness_of_fit(y, y_hat)
#         rmse_arr.append(rmse)
#         rsquare_arr.append(rsquare)

#     # subplots
#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

#     # RSquare
#     axes[0].plot(range(2, 200), rsquare_arr, c='k', label='R Square')
#     axes[0].axis('tight')
#     axes[0].set_xlabel('k number of nearest neighbors')
#     axes[0].set_ylabel('RSquare')
#     axes[0].legend(loc='upper right')
#     axes[0].set_title("RSquare for the KNeighborsRegressor")

#     # RMSE
#     axes[1].plot(range(2, 200), rmse_arr, c='k', label='RMSE')
#     axes[1].axis('tight')
#     axes[1].set_xlabel('k number of nearest neighbors')
#     axes[1].set_ylabel('RMSE')
#     axes[1].legend(loc='upper left')
#     axes[1].set_title("RMSE for the KNeighborsRegressor")

#     # adjust space between each plot
#     plt.subplots_adjust(hspace=0.5)
    
    # show plot
#     st.pyplot(fig)
   

    st.write('Based on the above plots, it seems that a K value between 0 and 5 yields a higher RSquare score and a lower RMSE score, indicating a better fit for the model.')
    
    # fit the best K value by performing cross-validation for each K value
    k_values = range(1, 11)

    best_k = None
    best_mse = np.inf

    # Perform cross-validation for each K value
    for k in k_values:
        # Train a KNN regression model
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)

        # Predict on the testing set
        y_pred = model.predict(X_test)

        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)

        # Check if current K value is the best
        if mse < best_mse:
            best_k = k
            best_mse = mse

    st.write(f"Best K value: {best_k}")
    

    
    # fit model
    opt_knn = KNeighborsRegressor(n_neighbors=best_k)
    opt_knn.fit(X_train, y_train)
    y_pred = opt_knn.predict(X_test)
    accuracy = opt_knn.score(X_test, y_test)
    st.write("## Model accuracy: ", accuracy)
    
    

     # 1. Genre
    genre_options = data['Genre'].unique()
    genre = st.selectbox('Choose a genre:', genre_options)
    submitted = False

    with st.form('userinput2'):
        # 2. Production Budget
        productionBudget = st.number_input(label = 'Production Budget($)',min_value = 0)
        # 3. Domestic Box Office
        domesticBoxOffice = st.number_input(label = 'Domestic Box Office($)', min_value = 0)
        st.write("### User Input：{}".format([genre, productionBudget, domesticBoxOffice]))
 
        # submission button
        submitted2 = st.form_submit_button("Submit to predict International Box Office")
 
        
#        if submitted:
        if submitted2:
        # preprocess 'Genre' by performing ont-hot (14)
            genre_Action, genre_Adventure, genre_Drama, genre_Thriller_Suspense, genre_Comedy, genre_Western, genre_Musical, genre_BlackComedy, genre_Horror, genre_RomanticComedy, genre_Documentary, genre_Concert_Performance, genre_Reality, genre_MultipleGenres, = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
            if genre == 'Action':
                genre_Action = 1
            elif genre == 'Adventure':
                genre_Adventure = 1
            elif genre == 'Drama':
                genre_Drama = 1
            elif genre == 'genre_Thriller/Suspense':
                genre_Thriller_Suspense = 1
            elif genre == 'Comedy':
                genre_Comedy = 1
            elif genre == 'Western':
                genre_Western = 1
            elif genre == 'Musical':
                genre_Musical = 1
            elif genre == 'Black Comedy':
                 genre_BlackComedy = 1
            elif genre == 'Horror':
                genre_Horror = 1
            elif genre == 'Romantic Comedy':
                genre_RomanticComedy = 1
            elif genre == 'Documentary':
                genre_Documentary = 1
            elif genre == 'Concert/Performance':
                genre_Concert_Performance = 1
            elif genre == 'Reality':
                genre_Reality = 1
            elif genre == 'Multiple Genres':
                genre_MultipleGenres = 1

            # combie all features together
            temp_feature = [
                 productionBudget, 
                 domesticBoxOffice,
                 genre_Action, 
                 genre_Adventure,
                 genre_Drama, 
                 genre_Thriller_Suspense, 
                 genre_Comedy, 
                 genre_Western, 
                 genre_Musical,
                 genre_BlackComedy, 
                 genre_Horror,
                 genre_RomanticComedy,
                 genre_Documentary, 
                 genre_Concert_Performance, 
                 genre_Reality, 
                 genre_MultipleGenres]
            
            st.write(str(temp_feature))


            # predict
            # The International Box Office:
            predUser = opt_knn.predict([temp_feature])
            formatted_predUser = np.array2string(predUser, formatter={'float_kind': lambda x: '{:,.0f}'.format(x)})
            st.write('## The International Box Office is around($): ' , str(formatted_predUser))
           
            # The Global Box Office:
            globalBox_predUser = predUser + domesticBoxOffice
            formatted_globalBox_predUser = np.array2string(globalBox_predUser, formatter={'float_kind': lambda x: '{:,.0f}'.format(x)})
            st.write('## The Global Box Office is around($): ' , str(formatted_globalBox_predUser))
            
            
            # Return on Investment" -> "ROI"
            ROI = (globalBox_predUser -productionBudget)/productionBudget
            formatted_ROI = np.array2string(ROI, formatter={'float_kind': lambda x: '{:,.0f}'.format(x)})
            
            st.write('## Return on Investment(ROI):', str(formatted_ROI))
        


    



    




if __name__ == '__main__':
    main()
