# Financial Aid

## Dataset

Both raw and processed data is kept in the `datawarehouse` folder. Which however isn't committed due to data privacy.

## Run

### Environment

Install the required libraries using the following command

```python
python3 -m pip install -r requirements.txt
```

### Run Locally

You can run the dashboard locally after installing the libraries and visiting the hosted address (e.g. http://127.0.0.1:8050/) using a browser (Google Chrome preferred).

```python
gunicorn app:server --bind=0.0.0.0:8050
```

### AWS Deployment

To deploy this dashboard to a cloud server like AWS use the following steps:

1. Create an EC2 instance. Run and connect to it.
2. Install Git and clone this repository:

    ```dash
    git clone https://github.com/UVA-MLSys/Financial-Aid.git
    sudo yum update -y
    sudo yum install git -y
    ```

3. **Environment**: Create a virtual env and activate it. Install the required libraries:

    ```dash
    sudo apt install python3.12
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Data**: Upload the `Merged.csv` file in the `datawarehouse` folder. Since there is no drag and drop, once easy way is to upload to Google drive, get a share link and use the file id from there to download inside the `datawarehouse` folder: `gdown shared_file_id`.
5. In `app.py` replace host address `127.0.0.1` with `0.0.0.0`. The port 8050 is fine.
6. **Network**: Deploy the app with `python app.py`. This runs the app inside the instance and can be accessed though the `http://ec2_ip_address:port` as long as the EC2 network security group contains an inbound rule allows TCP connections to that port (8050) for incoming IPs.
7. **Scale**: To scale up use binding through gunicorn. For example, `gunicorn --workers 3 app:server --bind=0.0.0.0:8050` creates 3 worker processes for deployment (maximum 3 people will be able to connect at a time). To upgrade the EC2 instance, stop the instance first, then change the instance type.
8. **Persistence**: To keep the server running even when you are not connected to the EC2 console: `screen gunicorn --workers 3 app:server --bind=0.0.0.0:8050 &`. 
   1. As long as the EC2 is running, the server will run in background.
   2. Use `ps -X` to find the running server processes. `kill process_id` if you want to terminate the run and deploy a new version.
9.  If you want to host to an address without having to add port number `8050` at the end, you have to run it on default port 80.
    1.  This is not permitted by default and you'll receive permission errors.
    2.  Install `nginx` and configure it at a mirror proxy server to redirect port gunicorn `8050` traffic to nginx port `80`.
10. To deploy to an `https` server you'll need a 
    1. Domain that can give you a SSL certificate
    2. Target group that includes your EC2 instance and the port it must redirect.
    3. Load balancer for http and https listeners attached to the target group.
    4. Route 53 records for your domain to catch traffic for your load balancer and give a nice domain name as alias to the load-balancer address.
