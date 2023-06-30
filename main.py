import openai
import streamlit as st
from streamlit_chat import message
import numpy as np
import urllib.request
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import pandas as pd
import time
import cv2
import plotly.express as px
from io import BytesIO,StringIO
from pyxlsb import open_workbook as open_xlsb
import base64



st.set_page_config(page_icon="🤖",page_title='Chat_model',layout='wide')
openai.api_key = st.sidebar.text_input("Please Enter your API key",placeholder='Please visit openai website for apikey',type='password')
if openai.api_key:
    llm = OpenAI(api_token=openai.api_key)
    pandas_ai = PandasAI(llm,conversational=True)
    def text_to_image():
        st.title("DallE2 model 🤖")
        def url_to_image(url):
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        try:
            image_prompt = st.text_input("Enter your prompt to generate image")
            image_limit = st.slider("Enter the number of images you want to generate upto 10",0,10,1)
            image_size = st.text_input("Enter the size of image",placeholder='Eg : 1024x1024')
            with st.spinner("Please Wait for DallE to generate your image...."):
                response = openai.Image.create(
                prompt=image_prompt,
                n=image_limit,
                size=image_size
                        )
                image_url = response['data'][0]['url']
                image_ = url_to_image(image_url)
                st.image(image_,caption=image_prompt)
        except:
            pass
    
    def ChatGPT():
        st.title("ChatGPT")
        def chatgpt(prompt):
            completions = openai.Completion.create(
                engine = 'text-davinci-003',
                prompt = prompt,
                max_tokens = 1024,
                n = 1,
                stop = None,
                temperature = 0.7,
            )
            message = completions.choices[0].text
            return message
        
        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        def get_text():
            st.image("https://images.emojiterra.com/google/noto-emoji/unicode-15/animated/1f916.gif",width=100)
            input_text = st.text_input("User: ","Hello How are you doing? ",key="input")
            return input_text
        user_input = get_text()
        if user_input:
            output = chatgpt(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        if st.session_state['generated']:

            for i in range(len(st.session_state['generated'])-1,-1,-1):
                message(st.session_state['generated'][i],key=str(i))
                message(st.session_state['past'][i],is_user=True,key=str(i)+"_user")
                
    def PandasAI():
        st.title("PandasAI model 🐼")
        file = st.file_uploader("Please upload you dataframe here!!")
        if file:
            try:
                st.subheader("Your Dataframe")
                df = pd.read_csv(file)
                st.checkbox("Use container width", value=False, key="use_container_width")
                st.dataframe(df.head(7), use_container_width=st.session_state.use_container_width)
                user_prompt = st.text_input("Please ask your question about the dataframe!")
                if st.button("Generate"):
                    with st.spinner("Please wait for PandasAI to Generate....."):
                
                        st.write(pandas_ai.run(df,user_prompt))
                else:
                    st. error("Please enter your prompt")
            except:
                pass

    def Excel_plotter():
        st.title("Excel plotter 📈")
        st.subheader("Please upload your file here 👇")
        csv_file = st.file_uploader("**Please upload if you want to convert your csv file to excel file**")
        excel_file = st.file_uploader("**If you have a excel file or convert it into excel and upload here!**",type='xlsx')
        def generate_excel_download_link(df):
                # Credit Excel: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5
                towrite = BytesIO()
                df.to_excel(towrite, encoding="utf-8", index=False, header=True)  # write to BytesIO buffer
                towrite.seek(0)  # reset pointer
                b64 = base64.b64encode(towrite.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_download.xlsx">Download Excel File</a>'
                return st.markdown(href, unsafe_allow_html=True)
        def generate_html_download_link(fig):
                # Credit Plotly: https://discuss.streamlit.io/t/download-plotly-plot-as-html/4426/2
                towrite = StringIO()
                fig.write_html(towrite, include_plotlyjs="cdn")
                towrite = BytesIO(towrite.getvalue().encode())
                b64 = base64.b64encode(towrite.read()).decode()
                href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download Plot</a>'
                return st.markdown(href, unsafe_allow_html=True)
        if csv_file:
            dataframe = pd.read_csv(csv_file)
            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'}) 
                worksheet.set_column('A:A', None, format1)  
                writer.save()
                processed_data = output.getvalue()
                return processed_data
            excel_file = to_excel(dataframe)
            st.download_button(label='📥 Download your excel file here',
                                    data=excel_file ,
                                    file_name= 'data.xlsx')
        if excel_file:
            st.markdown('---')
            df = pd.read_excel(excel_file,engine='openpyxl')
            st.dataframe(df)
            get_col = list(pd.read_excel(excel_file,nrows=1, engine='openpyxl'))
            groupby_column = st.selectbox("What would you like to analyse?",get_col)
            output_columns = st.multiselect("What would you like to compare your data pick two columns",get_col)

            df_grouped = df.groupby(by=[groupby_column],as_index=False)[output_columns].sum()
            st.dataframe(df_grouped)
            try:
                fig = px.bar(
                    df_grouped,
                    x=groupby_column,
                    y=output_columns[0],
                    color=output_columns[1],
                    color_continuous_scale=['red','yellow','green'],
                    template='plotly_white',
                    title=f'<b>{output_columns[0]} & {output_columns[1]} by {groupby_column}</b>'


                )
                st.plotly_chart(fig)
                st.subheader("Downloads")
                generate_excel_download_link(df_grouped)
                generate_html_download_link(fig)
            except:
                st.error("Please Select any two the columns")
                
            

    users_choice = st.sidebar.selectbox("What feature you want to use?",options=['ChatGPT','DallE2','PandasAI','Excel_Plotter'])
    if users_choice=="ChatGPT":
        ChatGPT()
    if users_choice=="DallE2":
        text_to_image()
    if users_choice=="PandasAI":
        PandasAI()
    if users_choice=="Excel_Plotter":
        Excel_plotter()



else:
    st.error("Please enter your api key!!")