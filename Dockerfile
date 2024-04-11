FROM python:3.7

COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/

RUN rm -f /usr/local/lib/python3.7/site-packages/setuptools-27.2.0-py3.7.egg
RUN pip install --upgrade setuptools
RUN pip install scikit-build
RUN pip install -r requirements.txt

CMD streamlit run app.py
