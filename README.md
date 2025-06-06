# Resnet18Cloud

# Installation

- Run setup.sh to create a virtual environment and install all the dependecies
- Activate virtual environment
- Run app.py from app directory
- Run dispatcher.py in another terminal from the dispatcher directory

# How to use

- Open a new terminal. To test the current state of implementation we will use curl to POST and GET response to the servers.
- Send image to the dispatcher

```bash
# User/Load Tester sends image to dispatcher
curl -X POST -F "image=@images/imagenet-sample-images/n01531178_goldfinch.JPEG" http://127.0.0.1:8080/predict_async
```

- Response from Server

```bash
{"message":"Request queued for processing","queue_size":0,"request_id":"<req_id>","status_url":"/result/<req_id>"}
```

```bash
{"message":"Request queued for processing","queue_size":0,"request_id":"9025f980-bc3a-4b70-a865-1d25494f5ac2","status_url":"/result/9025f980-bc3a-4b70-a865-1d25494f5ac2"}
```

- Get result from server

```bash
curl -X GET http://127.0.0.1:8080/result/<req_id>
```

```bash
{"completed_at":"2025-06-06T11:55:36.456400","filename":"n01531178_goldfinch.JPEG","processing_time":0.22717595100402832,"queued_at":"2025-06-06T11:55:36.228686","replica_used":"http://127.0.0.1:5000","request_id":"9025f980-bc3a-4b70-a865-1d25494f5ac2","result":{"filename":"n01531178_goldfinch.JPEG","inference_time":0.19434309005737305,"predictions":["goldfinch","bulbul","chickadee","magpie","junco"]},"status":"completed"}
```
