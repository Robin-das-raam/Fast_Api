FROM python:3.10-slim

#Set the working directory in the container
WORKDIR /app

#copy the current directoy content into the container at /app
COPY . /app

#install any needed packages specified in requireents.txt
RUN pip install --no-cache-dir -r requirements.txt

#Make port 8000 available to the world outside this container
EXPOSE 8000

#Define enviroment variable
ENV PYTHONBUFFERED=1

#Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]