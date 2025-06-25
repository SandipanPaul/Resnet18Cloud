# Resnet18Cloud

# Steps to get started
## Pre requisites
- Docker  
- Minikube  
- Kubectl  
- Python 3.9+

### 1. Clone repo & enter
```bash
git clone https://github.com/your-org/ResNet18Cloud.git
cd ResNet18Cloud
```

### 2. Python virtual environment & dependencies
 Either do 
```bash
chmod +x ./setup.sh
```
or

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start minikube 
```bash
minikube start
```

### 4. Build & load images into minikube
```bash
# Dispatcher
cd dispatcher/
docker build -t dispatcher:latest .
minikube image load dispatcher:latest

# App (ResNet18 service)
cd app/
docker build -t resnet18-app:latest .
minikube image load resnet18-app:latest
```

### 5. Deploy Kubernetes Services
```bash
kubectl apply -f k8s/dispatcher.yaml   
kubectl apply -f k8s/deployment.yaml             
# kubectl apply -f k8s/hpa.yaml                     
```

### 6. Port forwarding & verifying the connection between dispatcher-replicas
```bash
kubectl port-forward svc/dispatcher-service 8080:8080
```

- On another terminal, check dispatcher health
```bash
curl -s http://127.0.0.1:8080/health | jq
```
which should return `{"status":"healthy"}`

- Send one image through dispatcher (it queues the request)
```bash
RESP=$(curl -s -X POST \
  -F "image=@images/imagenet-sample-images/n01531178_goldfinch.JPEG" \
  http://127.0.0.1:8080/predict_async)
```
- (Optional) In order to check the response the dispatcher, do 
```bash
echo "Dispatch response → $RESP"
```
to get
`Dispatch response → {"from_cache":false,"message":"Request queued for processing","queue_size":1,"request_id":"f18d7c0a-e37f-4f6c-b214-542dd751b0e3","status_url":"/result/f18d7c0a-e37f-4f6c-b214-542dd751b0e3"}`

- Extract the request ID
```bash
D=$(echo "$RESP" | jq -r .request_id)
echo "Request ID → $ID"
```

expected output will be something like `Request ID → f18d7c0a-e37f-4f6c-b214-542dd751b0e3`

- Poll for the result
```bash
for i in {1..5}; do
  echo -n "Attempt $i: "
  curl -s http://127.0.0.1:8080/result/"$ID" | jq
  sleep 1
done
```

Expected output will be this:
```bash
Attempt 1: {
  "completed_at": "2025-06-25T11:22:41.523565",
  "filename": "n01531178_goldfinch.JPEG",
  "from_cache": false,
  "processing_time": 0.38537049293518066,
  "queued_at": "2025-06-25T11:22:41.129970",
  "replica_used": "http://resnet18-service.default.svc.cluster.local:5000",
  "request_id": "f18d7c0a-e37f-4f6c-b214-542dd751b0e3",
  "result": {
    "filename": "n01531178_goldfinch.JPEG",
    "inference_time": 0.24497175216674805,
    "predictions": [
      "goldfinch",
      "bulbul",
      "chickadee",
      "magpie",
      "junco"
    ]
  },
  "status": "completed"
}
.
.
. 
}
```
Which means the dispatcher is forwarding the requests to the replicas and we are getting the predictions. 

### 7. Veritfy the load tester - replica connection
in one terminal do the port forwarding, andf in another terminal inside the virtual environment, run the load tester python file.
Expected output when running the `python3 load_tester.py`
```bash
2025-06-25 16:08:29,659 - INFO - Found 1000 image files
2025-06-25 16:08:29,659 - INFO - Starting load test - Images per second schedule: [4, 8, 11, 15, 19, 22, 26, 30, 30, 30, 30, 30, 30, 30, 10, 2, 2, 2, 2, 2, 30, 30, 30, 30, 27, 2, 2, 2, 2, 2, 30, 30, 30, 30, 30, 30, 1, 2, 2, 2, 2, 2, 30, 30, 30, 30, 30, 27, 30, 26, 23, 19, 16, 12, 9, 5]
2025-06-25 16:08:29,659 - INFO - Max workers: 50
2025-06-25 16:08:29,659 - INFO - Second 1: Sending 4 images
2025-06-25 16:08:30,660 - INFO - Second 2: Sending 8 images
2025-06-25 16:08:31,660 - INFO - Second 3: Sending 11 images
2025-06-25 16:08:32,661 - INFO - Second 4: Sending 15 images
.
.
.
```
In another terminal, tail the app pod logs. This will combine all the logs from the 3 replicas. 
```bash
kubectl logs -l app=resnet18-app --tail=20 --follow
```

expected output will be like:

```bash
INFO:__main__:Prediction completed in 0.086s for n01560419_bulbul.JPEG
INFO:__main__:Prediction completed in 0.162s for n01824575_coucal.JPEG
INFO:werkzeug:10.244.0.36 - - [25/Jun/2025 14:08:31] "POST /predict HTTP/1.1" 200 -
INFO:__main__:Prediction completed in 0.087s for n04141327_scabbard.JPEG
INFO:werkzeug:10.244.0.36 - - [25/Jun/2025 14:08:30] "POST /predict HTTP/1.1" 200 -
INFO:__main__:<FileStorage: 'n02808440_bathtub.JPEG' (None)>
INFO:__main__:Prediction completed in 0.107s for n02808440_bathtub.JPEG
.
.
.
```


To inquiry the dispatcher metrics (JSON with requests, throughput, queue size , cache hit etc) run: 
```bash
curl localhost:8080/status | jq
```
expected output:
```bash
 % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   607  100   607    0     0  14804      0 --:--:-- --:--:-- --:--:-- 14804
{
  "cache_metrics": {
    "cache_hit_rate": 55.39,
    "cache_hits": 1378,
    "hit_rate": 55.39,
    "hits": 1378,
    "max_size": 1000,
    "misses": 1110,
    "size": 887,
    "utilization": 88.7
  },
  "endpoint_url": "http://resnet18-service.default.svc.cluster.local:5000",
  "failed_requests": 110,
  "health_metrics": {
    "error_rate": 4.42,
    "last_request_timestamp": 1750860535.1879418,
    "uptime": 13857.83
  },
  "performance_metrics": {
    "avg_processing_time": 0.043,
    "avg_queue_time": 3.987,
    "throughput": 0.17
  },
  "queue_metrics": {
    "peak_queue_size": 200,
    "queue_capacity": 200,
    "queue_utilization": 0
  },
  "queue_size": 0,
  "stored_results": 2488,
  "successful_requests": 2378,
  "total_requests": 2488
}
```
It means load testetr is talking to the dispatcher, and dispatcher is fwding the requests to the replicas.





<!--NEEDS TO BE UPDATED-->
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
