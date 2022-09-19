FROM python:3.10-slim-buster
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install st_btn_select

COPY hand-sign-detector/app.py .
COPY hand-sign-detector/utils.py .
COPY hand-sign-detector/checkpoints/ .
COPY hand-sign-detector/checkpoints/yolov4_tiny_1_3_416_416_static.onnx checkpoints/yolov4_tiny_1_3_416_416_static.onnx
COPY hand-sign-detector/HandSignModel/ . 
COPY hand-sign-detector/HandSignModel/__init__.py HandSignModel/__init__.py 
COPY hand-sign-detector/HandSignModel/HandSignModel.py HandSignModel/HandSignModel.py 

ENTRYPOINT [ "streamlit" ]
CMD ["run", "app.py" ]

EXPOSE 8501

