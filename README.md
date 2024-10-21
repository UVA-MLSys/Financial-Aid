# Financial-Aid

## Dataset

Both raw and processed data is kept in the `datawarehouse` folder. Which however isn't committed due to data privacy. 

## Install

### AWS

If you are on an AWS EC2 instance, install a virtual python environment first.

```bash
sudo yum update -y
sudo yum install git -y
git clone https://github.com/UVA-MLSys/Financial-Aid.git

cd Financial-Aid
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the file [locally](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive)

```bash
# Upload the Merged.csv file locally in aws
# One way is to upload it is Google drive, then wget from there
# Go to the file share option and get the file id from that link
# if file size < 100MB
wget --no-check-certificate drive_link -O Merged.csv
# if file size is bigger than 100MB
gdown --id <put-the-file-ID>
```

sudo apt install python3.12

### Environment

Install the required libraries using the following command

```python
python3 -m pip install -r requirements.txt
```

## Run

```python
app.run_server(host= '0.0.0.0', port=8050)
gunicorn app:server --bind=0.0.0.0:8050
gunicorn --workers 3 app:server --bind=0.0.0.0:8050
screen gunicorn --workers 3 app:server --bind=0.0.0.0:8050 &
deactivate
```
