# Use a lightweight Python image
FROM python:3.9-slim
 
# Install necessary Python libraries
RUN pip install pandas scikit-learn joblib
 
# Create directories for data and model
RUN mkdir -p /app/data /app/model
 
# Copy all scripts and data files into the container
COPY preprocess.py /app/preprocess.py
COPY train.py /app/train.py
COPY evaluate.py /app/evaluate.py
COPY data /app/data
 
# Set the working directory
WORKDIR /app
 
# Set a default entrypoint to specify which stage to run
ENTRYPOINT ["python"]
CMD ["preprocess.py"]
